# coding=utf-8
"""
resnet101_mamba_ite_score.py  – **ResNet‑101 Backbone × ITE (short‑range) × Mamba (long‑range) × ScoreSoftmax + UNet shallow segmentation分支 + GCN/Graph分支 + 视觉泛化向量(VGT) + Domain/Region Learnable Tokens + Prototype Guidance + Pseudo‑Boundary Graph + Domain‑Adversarial(GRL)**
====================================================================================================================================================================================

> **最新变动（2025‑07‑05）**
> 1. **Irrelevant‑Feature Suppressor (IFS)**：通道‑Gate + Spatial‑Mask 双重抑制病灶无关纹理。
> 2. **Domain / Region Tokens (DT / RT)**：一组可学习向量插入视觉序列，动态吸收患者差异 & 图像背景噪声。
> 3. **Graph Prototype Embedding (GPE)**：对 UNet 分割得到的超像素 / k‑NN 区域产生 prototype，指导主干聚类式学习。
> 4. **Pseudo‑Boundary Graph (PBG)**：利用 segmentation mask 生成边界图，引入辅助对比损失 Boundary‑Graph‑Loss。
> 5. 仍保留 **VGT + GRL(DomainClassifier)**，并在融合阶段统一对齐。
"""

import random, math, jittor as jt
from jittor import nn
from jittor.models import resnet101
from nnUnet_jittor import nnUNet2D
import cv2
import numpy as np

# ---------------- Config -----------------
class Config:
    num_classes = 4
    score_levels = 5
    # gaussian label params
    score_mu_correct, score_sigma_correct = 0.8, 0.07
    score_mu_wrong,   score_sigma_wrong  = 0.2, 0.07
    # prototype / token dims
    token_dim  = 64
    proto_dim  = 64
    n_domain_token = 2   # e.g. source / target
    n_region_token = 4   # quadrants or learnable region hints

# ------------- Gaussian Label ------------

def gaussian_label_matrix(batch_lbl,num_classes, noise=True):
    lbls = batch_lbl.int32().numpy().tolist()
    B, C, G = len(lbls), num_classes, Config.score_levels
    Y = jt.zeros((B, C, G))
    table = jt.arange(1, G+1).float32()
    for b,l in enumerate(lbls):
        for c in range(C):
            mu,sg = (Config.score_mu_correct,Config.score_sigma_correct) if c==l else (Config.score_mu_wrong,Config.score_sigma_wrong)
            mu,sg = mu*G, sg*G
            if noise:
                mu += random.uniform(-.2,.2)*G
                sg = max(sg + random.uniform(-.1,.1)*G, 1e-2)
            dist = jt.exp(-(table-mu)**2/(2*sg**2))
            Y[b,c] = dist / jt.sum(dist)
    return Y

class ScoreSoftmaxLoss_KL(nn.Module):
    def execute(self, pred_scores, labels,num_classes=Config.num_classes):
        y_true = gaussian_label_matrix(labels,num_classes=num_classes)
        y_pred = jt.maximum(pred_scores, 1e-8)
        y_true = jt.maximum(y_true, 1e-8)
        kl = jt.sum(y_true * (jt.log(y_true)-jt.log(y_pred)), dims=[1,2])
        return jt.mean(kl)

# ------- CORE BUILDING BLOCKS ----------------------------------

# 1) Irrelevant‑Feature Suppressor
class IrrelevantFeatureSuppressor(nn.Module):
    """通道 Gate + Spatial Mask"""
    def __init__(self, in_ch):
        super().__init__()
        self.ch_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv(in_ch, in_ch//4,1), nn.ReLU(), nn.Conv(in_ch//4, in_ch,1), nn.Sigmoid())
        self.spa_gate = nn.Sequential(nn.Conv(in_ch, 1, 1), nn.Sigmoid())
    def execute(self, x):
        ch_w = self.ch_gate(x)
        spa_w = self.spa_gate(x)
        return x * ch_w * spa_w

# 2) Visual Generalization Token (unchanged but param dim from Config)
class VisualGeneralizationToken(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv(in_dim, Config.token_dim, 1), nn.GELU(), nn.Conv(Config.token_dim, in_dim, 1))
    def execute(self, x):
        token = self.gate(jt.mean(x, dims=[2,3], keepdims=True))
        return x + token

# 3) Domain / Region Learnable Tokens
class TokenPool(nn.Module):
    """returns learnable tokens broadcast to B and concat with patch tokens"""
    def __init__(self, n_token, dim):
        super().__init__()
        self.token = jt.randn(n_token, dim) * 0.02
    def execute(self, B):
        return self.token.unsqueeze(0).repeat(B,1,1)  # [B,T,dim]

# 4) Prototype Guidance
# class GraphPrototype(nn.Module):
#     def __init__(self, n_proto, feat_dim):
#         super().__init__()
#         self.proto = nn.Parameter(jt.randn(n_proto, feat_dim)*0.02)
#     def execute(self, feat):
#         # feat: [B,N,D] graph node features
#         sim = jt.bmm(feat, self.proto.transpose(0,1))  # [B,N,P]
#
#         attn = nn.softmax(sim, dim=-1)
#         proto_feat = jt.bmm(attn, self.proto.unsqueeze(0).repeat(feat.shape[0],1,1))  # [B,N,D]
#         return proto_feat.mean(dim=1)  # [B,D]

class GraphPrototype(nn.Module):
    """Generate a prototype vector per‑batch via attention over a learnable codebook.
    Expects `feat` as [B, N, D] (graph nodes). Returns [B, D] or [B, P] depending on head.
    We fix the dimensionality mismatch that caused the previous `bmm` assertion
    (both operands must be 3‑D).  The learnable prototype matrix is repeated along
    the batch dimension so the second operand has shape [B, D, P].
    """
    def __init__(self, n_proto: int, feat_dim: int):
        super().__init__()
        self.weight = nn.Parameter(jt.randn(n_proto, feat_dim) * 0.02)  # [P, D]

    def execute(self, feat):
        """feat : [B, N, D] graph/node features"""
        B, N, D = feat.shape
        # Prepare prototype matrix for batched matmul ➜ shape [B, D, P]
        proto = self.weight.transpose(0, 1)           # [D, P]
        proto = proto.unsqueeze(0).repeat(B, 1, 1)    # [B, D, P]

        # Similarity & soft‑assignment
        sim = jt.bmm(feat, proto)                     # [B, N, P]
        attn = nn.softmax(sim, dim=-1)                # attention over prototypes
        proto_feat = jt.bmm(attn, self.weight.unsqueeze(0).repeat(B, 1, 1))  # [B, N, D]

        # Mean pooling over nodes gives per‑sample prototype guidance vector [B, D]
        return proto_feat.mean(dim=1)


# 5) Gradient Reversal Layer & Domain Classifier
class GradReverse(jt.Function):
    @staticmethod
    def execute(x):
        return x
    @staticmethod
    def grad(x):
        return -x

def grl(x):
    return GradReverse.apply(x)

class DomainClassifier(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(feat_dim,128), nn.ReLU(), nn.Linear(128,2))
    def execute(self, x):
        return self.layer(grl(x))

# 6) Simplified GCN
class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.theta = nn.Linear(in_dim, hidden_dim)
        self.phi   = nn.Linear(in_dim, hidden_dim)
        self.proj  = nn.Linear(hidden_dim, in_dim)
    def execute(self, x):
        B,C,H,W = x.shape
        feat = x.reshape(B, C, -1).transpose(0,2,1)  # [B,N,C]
        t = self.theta(feat); p = self.phi(feat)
        adj = jt.bmm(t, p.transpose(0,2,1)) / math.sqrt(t.shape[-1])
        adj = nn.softmax(adj, dim=-1)
        out = jt.bmm(adj, t)          # [B,N,H]
        out = self.proj(out).transpose(0,2,1).reshape(B,C,H,W)
        return out

# ------------------- Backbone & Blocks --------------------------
# >>>>> Your existing ITEBlock, MambaBranch, UNetEncoder kept identical (omitted here for brevity)
# -------- ITE/Mamba blocks ----------
class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = jt.ones((1, dim))
        self.bias = jt.zeros((1, dim))
        self.eps = eps
    def execute(self, x):
        x = x.permute(0, 2, 3, 1)
        mean = x.mean(dim=-1, keepdims=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdims=True)
        x = (x - mean) / jt.sqrt(var + self.eps)
        x = x * self.weight.reshape(1, 1, 1, -1) + self.bias.reshape(1, 1, 1, -1)
        return x.permute(0, 3, 1, 2)

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=7):
        super().__init__()
        self.dim = dim
        self.heads = num_heads
        self.win = window_size
        self.scale = (dim//num_heads)**-0.5
        self.qkv = nn.Linear(dim,dim*3)
        self.proj = nn.Linear(dim,dim)
    def execute(self,x):
        B,C,H0,W0 = x.shape                 # save original size
        pad_h = (self.win - H0%self.win)%self.win
        pad_w = (self.win - W0%self.win)%self.win
        if pad_h or pad_w:
            x = nn.pad(x, [0,pad_w,0,pad_h])
        B,C,H,W = x.shape
        x = x.reshape(B,C,H//self.win,self.win,W//self.win,self.win).transpose(0,2,4,3,5,1)
        x = x.reshape(-1, self.win*self.win, C)
        qkv = self.qkv(x).chunk(3,dim=-1)
        q,k,v = [t.reshape(t.shape[0], t.shape[1], self.heads, C//self.heads).transpose(0,2,1,3) for t in qkv]
        attn = nn.softmax((q*self.scale) @ k.transpose(0,1,3,2), dim=-1)
        out = (attn @ v).transpose(0,2,1,3).reshape(x.shape[0], x.shape[1], C)
        out = self.proj(out)
        out = out.reshape(B, H//self.win, W//self.win, self.win, self.win, C).transpose(0,5,1,3,2,4).reshape(B,C,H,W)
        # crop back to original size
        if pad_h or pad_w:
            out = out[:,:,:H0,:W0]
        return out


class ITEBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__(); self.norm=LayerNorm2d(dim); self.attn=WindowAttention(dim,heads); self.ffn=nn.Sequential(nn.Conv(dim,dim*4,1),nn.GELU(),nn.Conv(dim*4,dim,1))
    def execute(self,x):
        x = x + self.attn(self.norm(x)); x = x + self.ffn(self.norm(x)); return x

class Mamba1DBlock(nn.Module):
    def __init__(self,dim):
        super().__init__();
        self.gate=nn.Linear(dim,dim);
        self.out=nn.Linear(dim,dim);
        self.conv=None;
        self.dim=dim
        C=dim
        self.conv = nn.Conv1d(C, C, 9, padding=4, groups=C)
    def execute(self,x):
        B,C,N=x.shape
        # print(B,C,N)
        # if self.conv is None: self.conv=nn.Conv1d(C,C,9,padding=4,groups=C)
        g=jt.sigmoid(self.gate(x.transpose(0,2,1))).transpose(0,2,1)
        y=self.conv(x*g); y=self.out(y.transpose(0,2,1)); return x+y.transpose(0,2,1)

class MambaBranch(nn.Module):
    def __init__(self,dim): super().__init__(); self.m=Mamba1DBlock(dim)
    def execute(self,x): B,C,H,W=x.shape; seq=self.m(x.reshape(B,C,H*W)); return seq.reshape(B,C,H,W)

# ----------- UNet shallow encoder -----------
class UNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv(4, 32, 3, padding=1), nn.BatchNorm(32), nn.ReLU())
        self.pool1 = nn.Pool(2, stride=2, op='maximum')
        self.block2 = nn.Sequential(nn.Conv(32, 64, 3, padding=1), nn.BatchNorm(64), nn.ReLU())
        self.pool2 = nn.Pool(2, stride=2, op='maximum')
        self.block3 = nn.Sequential(nn.Conv(64, 128, 3, padding=1), nn.BatchNorm(128), nn.ReLU())
        self.pool3 = nn.Pool(2, stride=2, op='maximum')
        self.block4 = nn.Sequential(nn.Conv(128, 256, 3, padding=1), nn.BatchNorm(256), nn.ReLU())
    def execute(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.block4(x)
        return x
class ResNet101Backbone(nn.Module):
    def __init__(self, pretrained=True,in_channels=3):
        super().__init__()
        net = resnet101(pretrained=pretrained)
        if in_channels != 3:
            old_conv = net.conv1
            new_conv = nn.Conv(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # 复制原来 3 通道权重
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # 第 4 通道用 3 通道均值初始化
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)

            net.conv1 = new_conv


        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
    def execute(self,x):
        x = self.stem(x)
        _  = self.layer1(x)
        c2 = self.layer2(_)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c2,c3,c4

class CrossAttentionFusion(nn.Module):
    """
    将 prototype 向量通过 Cross‑Attention 注入到全局 feat 表示中
    Q = feat          （[B, 256] -> [B, 1, D]）
    K,V = proto_vec   （[B, 128] -> proj -> [B, 1, D]）
    输出仍是 [B, D]，并带残差。
    """
    def __init__(self, q_dim=256, kv_dim=128, hidden_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (hidden_dim // num_heads) ** -0.5

        # 线性映射到统一 hidden_dim
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.k_proj = nn.Linear(kv_dim, hidden_dim)
        self.v_proj = nn.Linear(kv_dim, hidden_dim)

        # 输出映射 & 残差
        self.out_proj = nn.Linear(hidden_dim, q_dim)

    def execute(self, feat, proto_vec):
        """
        feat:      [B, 256]
        proto_vec: [B, 128]
        """
        B = feat.shape[0]

        # [B, 1, D]
        Q = self.q_proj(feat).unsqueeze(1)
        K = self.k_proj(proto_vec).unsqueeze(1)
        V = self.v_proj(proto_vec).unsqueeze(1)

        # 拆分多头 -> [B, h, 1, d_k]
        h = self.num_heads
        dk = Q.shape[-1] // h
        Q = Q.reshape(B, 1, h, dk).transpose(0,2,1,3)   # [B,h,1,d_k]
        K = K.reshape(B, 1, h, dk).transpose(0,2,1,3)   # [B,h,1,d_k]
        V = V.reshape(B, 1, h, dk).transpose(0,2,1,3)   # [B,h,1,d_k]

        attn = nn.softmax((Q * self.scale) @ K.transpose(0,1,3,2), dim=-1)   # [B,h,1,1]
        ctx  = (attn @ V).transpose(0,2,1,3).reshape(B, 1, h*dk)             # [B,1,D]
        ctx  = ctx.squeeze(1)                                                # [B,D]

        fused = feat + self.out_proj(ctx)    # 残差加回原 feat
        return fused




# --------------------------------------
# 结构性量化指标提取函数（可扩展）
# --------------------------------------

def extract_rich_region_features(image: np.ndarray, mask: np.ndarray):
    """
    提取区域统计和结构语义特征:
    - area, aspect_ratio, roundness, perimeter
    - edge_sharpness, lobulation_score, echo_attenuation
    - texture_entropy, halo_contrast, brightness_variation
    :param image: 原始灰度图像 (H, W)，取值 [0, 255]
    :param mask: 二值掩码 (H, W)，值为 0/1
    :return: [12维] numpy array 特征
    """
    if mask.sum() == 0:
        return np.zeros(12, dtype=np.float32)

    h, w = image.shape
    area = mask.sum()
    rows, cols = np.where(mask)
    height, width = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
    aspect_ratio = height / (width + 1e-5)
    perimeter = cv2.arcLength(cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], True)
    roundness = 4 * np.pi * area / (perimeter ** 2 + 1e-5)

    # 1. 边缘锐利度（图像梯度均值）
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    edge_sharpness = np.mean(np.sqrt(grad_x**2 + grad_y**2)[mask > 0])

    # 2. 分叶度估计（轮廓曲率近似）
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0][:, 0, :]
    diffs = np.diff(contour, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    lobulation_score = np.std(angles)

    # 3. 后方回声衰减（下方 ROI 平均亮度）
    y_max = rows.max()
    lower_band = image[min(h-1, y_max+10):min(h, y_max+30), :]
    echo_attenuation = 1.0 - np.mean(lower_band) / (np.mean(image[mask > 0]) + 1e-5)

    # 4. 回声纹理（熵）
    roi_vals = image[mask > 0]
    hist = np.histogram(roi_vals, bins=32, range=(0, 255))[0] + 1e-6
    p = hist / hist.sum()
    texture_entropy = -np.sum(p * np.log(p))

    # 5. 声晕对比度（边缘内外亮度差）
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    halo_ring = (dilated > 0) & (mask == 0)
    halo_contrast = np.mean(image[halo_ring]) - np.mean(image[mask > 0])

    # 6. 区域内亮度方差（回声不均匀性）
    brightness_variation = np.std(image[mask > 0])

    # 归一化
    norm_area = area / (h * w)
    norm_perimeter = perimeter / (2 * (h + w))

    return np.array([
        norm_area, aspect_ratio, roundness, norm_perimeter,
        edge_sharpness, lobulation_score, echo_attenuation,
        texture_entropy, halo_contrast, brightness_variation,
        np.mean(image[mask > 0]) / 255.0,  # 平均亮度
        np.max(image[mask > 0]) / 255.0   # 最大亮度
    ], dtype=np.float32)




class RichStructEncoder(nn.Module):
    """
    模块：提取 rich_struct_feat，并进行 nn.Linear(12 → 64) 投影。
    输入为 [B, 1, H, W] 的图像灰度图和 mask。
    """
    def __init__(self):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(12, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

    def execute(self, x, mask):
        B = x.shape[0]
        feats = []
        for i in range(B):
            img = (x[i, 0] * 255).numpy().astype("uint8")
            msk = mask[i, 0].numpy().astype("uint8")
            feat = extract_rich_region_features(img, msk)
            feats.append(feat)
        feats = jt.array(feats)
        return self.projector(feats)  # [B, 64]


class ROIEncoder(nn.Module):
    """
    模块：基于 mask 提取 ROI 图像区域并编码为特征。
    """
    def __init__(self, in_channels=1, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.out_proj = nn.Linear(32, out_dim)

    def execute(self, x, mask):
        B = x.shape[0]
        crops = []
        for i in range(B):
            img = x[i, 0].numpy()
            msk = mask[i, 0].numpy()
            if msk.sum() == 0:
                crop = np.zeros((64, 64), dtype=np.float32)
            else:
                ys, xs = np.where(msk > 0)
                if ys.size == 0 or xs.size == 0:
                    crop = np.zeros((64, 64), dtype=np.float32)
                else:
                    y1, y2 = max(0, ys.min() - 5), min(img.shape[0], ys.max() + 5)
                    x1, x2 = max(0, xs.min() - 5), min(img.shape[1], xs.max() + 5)
                    crop = img[y1:y2, x1:x2]
                    crop = cv2.resize(crop, (64, 64)) / 255.
            crops.append(crop)
        crops = jt.array(np.stack(crops)).unsqueeze(1)  # [B, 1, 64, 64]
        feat = self.encoder(crops).squeeze(-1).squeeze(-1)  # [B, 32]
        return self.out_proj(feat)  # [B, 128]


class MskCrossAttentionFusion(nn.Module):
    """
    模块：Cross-Attention 融合主图像、结构 DropMask 引导、ROI特征
    """
    def __init__(self, embed_dim=2048, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(32 + 64 + 128+128, embed_dim)
        self.v_proj = nn.Linear(32 + 64 + 128+128, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def execute(self, img_feat, drop_feat, roi_feat):
        # img_feat: [B, 2048], drop_feat: [B, 32 + 64 + 128], roi_feat: [B, 128]
        fusion_kv = jt.concat([drop_feat, roi_feat], dim=1)
        q = self.q_proj(img_feat).unsqueeze(1)     # [B, 1, D]
        k = self.k_proj(fusion_kv).unsqueeze(1)    # [B, 1, D]
        v = self.v_proj(fusion_kv).unsqueeze(1)    # [B, 1, D]
        attn = nn.bmm(q, k.transpose(1, 2)) / (img_feat.shape[1] ** 0.5)  # [B, 1, 1]
        attn = nn.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        fused = nn.bmm(attn, v).squeeze(1)  # [B, D]
        return self.out_proj(fused)



class StructGuidedAttention(nn.Module):
    def __init__(self, struct_dim=64, vis_dim=2048, hidden_dim=512):
        super().__init__()
        self.q_proj = nn.Linear(struct_dim, hidden_dim)
        self.k_proj = nn.Linear(vis_dim, hidden_dim)
        self.v_proj = nn.Linear(vis_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, vis_dim)

    def execute(self, struct_feat, vis_feat):
        # struct_feat: [B, 64], vis_feat: [B, 2048]
        q = self.q_proj(struct_feat).unsqueeze(1)    # [B, 1, D]
        k = self.k_proj(vis_feat).unsqueeze(1)       # [B, 1, D]
        v = self.v_proj(vis_feat).unsqueeze(1)
        attn = nn.bmm(q, k.transpose(1, 2)) / (k.shape[-1] ** 0.5)
        attn = nn.softmax(attn, dim=-1)
        out = nn.bmm(attn, v).squeeze(1)
        return self.out_proj(out)


class AAMHead(nn.Module):
    def __init__(self, in_dim, out_dim, m=0.35, s=30.0):
        super().__init__()
        self.weight = nn.Parameter(jt.randn(out_dim, in_dim))
        self.m = m
        self.s = s

    def execute(self, x, label):
        W = nn.normalize(self.weight, dim=1)
        x = nn.normalize(x, dim=1)
        logits = jt.matmul(x, W.transpose(0, 1))
        one_hot = jt.zeros_like(logits)
        one_hot.scatter_(1, label.unsqueeze(1), 1.0)
        logits_m = logits - one_hot * self.m
        return self.s * logits_m





# ------------------- Main Fusion Model --------------------------
class Res101_Mamba_ITE_UNet_GraphPlus(nn.Module):
    def __init__(self, num_classes=Config.num_classes,in_channels=3,freeze_seg=True, seg_ckpt="busi_seg_ckpt.pkl"):
        super().__init__()
        Config.num_classes=num_classes
        self.backbone = ResNet101Backbone(in_channels=in_channels)

        base_model = resnet101(pretrained=True)
        self.img_encoder = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )


        self.ite2, self.ite3 = ITEBlock(512), ITEBlock(1024)
        self.ite4 = ITEBlock(2048)
        self.m2, self.m3   = MambaBranch(512), MambaBranch(1024)
        self.m4 = nn.Sequential(
            nn.Conv(2048, 1024, 1),
            nn.BatchNorm(1024),
            nn.ReLU(),
            nn.Conv(1024, 512, 1),
            nn.BatchNorm(512),
            nn.ReLU(),
            nn.Conv(512, 256, 1),
            # nn.BatchNorm(256)
        )
        self.seg_model          = nnUNet2D(in_channels=1, n_classes=2)

        self.seg_model.load(str(seg_ckpt))
        if freeze_seg:
            for param in self.seg_model.parameters():
                param.stop_grad()

        self._d_feats = {}

        def save_hook(name):
            def fn(module, x, y):
                self._d_feats[name] = y

            return fn

        self.seg_model.dec1.register_forward_hook(save_hook("d1"))
        self.seg_model.dec2.register_forward_hook(save_hook("d2"))
        self.seg_model.dec3.register_forward_hook(save_hook("d3"))


        self.gcn           = SimpleGCN(128, 64)
        self.irrelevant    = IrrelevantFeatureSuppressor(128)
        self.vgt           = VisualGeneralizationToken(256)
        # tokens
        self.domain_token  = TokenPool(Config.n_domain_token, 128)
        self.region_token  = TokenPool(Config.n_region_token, 128)
        # prototype
        self.proto         = GraphPrototype(n_proto=16, feat_dim=128)
        # reduce
        self.red2, self.red3 = nn.Conv(512,256,1), nn.Conv(1024,256,1)
        self.fuse   = nn.Sequential(nn.Conv(256*3+128, 256, 3, padding=1, bias=False), nn.BatchNorm(256), nn.GELU())
        self.pool   = nn.AdaptiveAvgPool2d(1)
        # self.cls    = nn.Linear(256, num_classes*Config.score_levels)
        self.table  = jt.arange(1,Config.score_levels+1).float32()
        # domain classifier
        self.domain_cls = DomainClassifier(256)
        self.proto_proj = nn.Linear(128, 256)  # 把 128‑D prototype 提升到 256‑D

        self.cross_fusion = CrossAttentionFusion(
            q_dim=256, kv_dim=128, hidden_dim=256, num_heads=4)

        self.cls_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256)
        )

        self.proj_fused = nn.Linear(512, 256)

        self.rich_encoder = RichStructEncoder()
        self.roi_encoder = ROIEncoder()
        self.struct_attn = StructGuidedAttention()

        self.fusion_attn = MskCrossAttentionFusion()

        self.cls = nn.Linear(256, num_classes * Config.score_levels)
        # self.cls = AAMHead(256, num_classes)

        self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=64)  # 用于 rich_struct_feat

        self.rich_proj = nn.Linear(64, 256)

    # -------- Helper --------
    def score_softmax(self, logits):
        B = logits.shape[0]
        return nn.softmax(logits.reshape(B, Config.num_classes, Config.score_levels), dim=2)

    # -------- Forward --------
    def execute(self, x, return_domain=False):


        x_rgb = jt.concat([x, x, x], dim=1)
        img_feat = self.img_encoder(x_rgb).squeeze(-1).squeeze(-1)  # [B, 2048]
        mask_logits, _, _ = self.seg_model(x)
        # mask = mask_logits.argmax(dim=1, keepdims=True)  # [B,1,H,W]
        mask = mask_logits.argmax(dim=1, keepdims=True)[0]

        # DropFeat from seg decoder
        d1 = nn.AdaptiveAvgPool2d((1, 1))(self._d_feats["d1"]).squeeze(-1).squeeze(-1)
        d2 = nn.AdaptiveAvgPool2d((1, 1))(self._d_feats["d2"]).squeeze(-1).squeeze(-1)
        d3 = nn.AdaptiveAvgPool2d((1, 1))(self._d_feats["d3"]).squeeze(-1).squeeze(-1)
        drop_feat = jt.concat([d1, d2, d3], dim=1)  # [B, 32 + 64 + 128]

        rich_struct_feat = self.rich_encoder(x, mask)  # [B, 64]
        roi_feat = self.roi_encoder(x, mask)  # [B, 128]

        struct_feat = self.struct_attn(rich_struct_feat, img_feat)
        attn_feat = self.fusion_attn(img_feat, drop_feat, roi_feat)

        all_feat = attn_feat + struct_feat

        x = jt.concat([x_rgb, mask], dim=1)

        # backbone
        c2, c3, c4 = self.backbone(x)
        c2 = self.m2(self.ite2(c2))
        c3 = self.m3(self.ite3(c3))
        c4 = self.m4(self.ite4(c4))


        u = self._d_feats["d3"] # [B,128,H_unet,W_unet]
        h, w = c2.shape[2:]
        u = nn.interpolate(u, size=(h, w), mode='bilinear', align_corners=False)
        u = self.gcn(u)
        # u = self.irrelevant(u)
        # ensure UNet branch matches spatial size with backbone features
        u = nn.interpolate(u, size=(h, w), mode="bilinear", align_corners=False)
        # concat tokens (flatten spatial first) (flatten spatial first)
        B = x.shape[0]
        tok_dom = self.domain_token(B)  # [B,Td,D]
        tok_reg = self.region_token(B)
        # resize feature maps to same H,W
        c3 = nn.interpolate(c3, size=(h, w), mode="bilinear", align_corners=False)
        c4 = nn.interpolate(c4, size=(h, w), mode="bilinear", align_corners=False)
        # fuse features
        fusion = jt.concat([self.red2(c2), self.red3(c3), c4, u], dim=1)
        # fusion = jt.concat([self.red2(c2), self.red3(c3), c4], dim=1)
        fusion = self.fuse(fusion)
        fusion = self.vgt(fusion)
        feat = self.pool(fusion).reshape(B, -1)  # [B,256]
        # prototype guidance loss input
        graph_feat = u.reshape(B, 128, -1).transpose(0, 2, 1)  # [B,N,D]
        proto_vec = self.proto(graph_feat)  # [B,D]

        feat = self.cross_fusion(feat, proto_vec)  # Cross‑Attention 融合

        feat_msk=self.cls_head(all_feat)

        fused_feat = self.proj_fused(jt.concat([feat_msk, feat], dim=1))

        # proto_vec = self.proto_proj(proto_vec)
        # feat = feat + proto_vec  # additive guidance
        logits = self.cls(fused_feat)



        score_mat = self.score_softmax(logits)
        scores = jt.matmul(score_mat, self.table)
        if return_domain:
            dom_logits = self.domain_cls(feat.detach())
            return scores, score_mat, dom_logits
        # return fused_feat,scores, score_mat
        return fused_feat, scores, score_mat, rich_struct_feat, feat, mask


# ------------------ Losses --------------------
main_criterion  = nn.cross_entropy_loss
score_criterion = ScoreSoftmaxLoss_KL()
ce_loss = nn.cross_entropy_loss

def compute_loss(scores, score_mat, labels,num_classes=Config.num_classes, dom_logits=None, domain_labels=None, alpha=0.5, beta=0.1):
    loss = main_criterion(scores, labels) + alpha*score_criterion(score_mat, labels,num_classes=num_classes)
    if dom_logits is not None and domain_labels is not None:
        loss += beta*ce_loss(dom_logits, domain_labels)
    return loss


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(jt.randn(num_classes, feat_dim) * 0.1)
    def execute(self, features, labels):
        centers_batch = self.centers[labels]
        return ((features - centers_batch) ** 2).sum(dim=1).mean()

# center_loss=CenterLoss(Config.num_classes,256)

def compute_total_loss(model,scores, score_mat, labels,
                       rich_feat, fused_feat,
                       domain_logits=None, domain_labels=None,
                       mask=None, gcn_feat=None,num_classes=Config.num_classes):

    # 1. 分类主损失（Focal Loss）
    ce = nn.cross_entropy_loss(scores, labels, reduction="none")
    pt = jt.exp(-ce)
    loss_cls = (1.0 * (1 - pt) ** 2 * ce).mean()

    # 2. 模糊标签 KL
    loss_score = ScoreSoftmaxLoss_KL()(score_mat, labels,num_classes=num_classes)

    # 3. 一致性损失（结构 ↔ 分类）

    rich_feat_proj = model.rich_proj(rich_feat)
    loss_struct_align = nn.MSELoss()(rich_feat_proj, fused_feat.detach())

    # 4. SC-Loss（结构聚类）
    loss_struct_center = model.center_loss(rich_feat, labels)

    # 5. 不确定性建模（熵）
    if gcn_feat is not None:
        prob = nn.softmax(gcn_feat.reshape(gcn_feat.shape[0], gcn_feat.shape[1], -1), dim=2)
        entropy = -jt.sum(prob * jt.log(prob + 1e-6), dim=2).mean()
        loss_uncertainty = entropy
    else:
        loss_uncertainty = 0.0

    # 6. 域对抗损失（可选）
    loss_domain = nn.cross_entropy_loss(domain_logits, domain_labels) if domain_logits is not None else 0.0

    # 汇总
    loss = loss_cls + 1 * loss_score + 1 * loss_struct_align + \
           1 * loss_struct_center + 1 * loss_uncertainty + 1 * loss_domain

    # print(loss_cls,loss_score,loss_struct_align,loss_struct_center,loss_uncertainty,loss_domain)

    # return loss
    return loss, loss_cls, loss_score, loss_struct_align, loss_struct_center, loss_uncertainty, loss_domain


__all__ = ["Res101_Mamba_ITE_UNet_GraphPlus", "compute_loss","compute_total_loss"]
