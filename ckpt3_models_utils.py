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
import copy
import random, math, jittor as jt
from jittor import nn
from jittor.models import resnet101
from nnUnet_jittor import nnUNet2D
from EMCADNet import EMCADNet
import cv2
import numpy as np

# ---------------- Config -----------------
class Config:
    num_classes = 4
    score_levels = 10
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
        self.weight = jt.randn(n_proto, feat_dim) * 0.02  # [P, D]

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
        self.out = nn.AdaptiveAvgPool2d((1, 1))
    def execute(self,x):
        x = self.stem(x)
        _  = self.layer1(x)
        c2 = self.layer2(_)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        out = self.out(c4)
        return c2,c3,c4,out

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
def normalize_rich_features(feat: np.ndarray):
    assert feat.shape[-1] == 12
    norm_feat = np.zeros_like(feat)

    norm_feat[0] = np.log1p(feat[0])                          # area
    norm_feat[1] = np.log1p(feat[1])                          # aspect_ratio
    norm_feat[2] = feat[2]                                    # roundness
    norm_feat[3] = np.log1p(feat[3])                          # perimeter
    norm_feat[4] = np.log1p(feat[4])                            # edge_sharpness
    norm_feat[5] = np.tanh(feat[5])                           # lobulation_score
    norm_feat[6] = np.tanh(np.clip(feat[6], -1.0, 1.0))       # echo_attenuation
    norm_feat[7] = feat[7] / 6.0                              # entropy
    norm_feat[8] = np.tanh(feat[8] / 15.0)                    # halo_contrast
    norm_feat[9] = np.log1p(feat[9]) / np.log1p(64)           # brightness_var
    norm_feat[10] = feat[10]                                  # mean_intensity
    norm_feat[11] = feat[11]                                  # max_intensity

    return norm_feat.astype(np.float32)



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
        # empty_token = jt.randn(12) * 0.01
        return np.ones(12, dtype=np.float32)*(1e-5)
        # return empty_token.numpy()

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
    lobulation_score = 0
    # 检查样本数量
    if len(angles) > 1:
        lobulation_score = np.std(angles)
    # else:
    #     lobulation_score = 0

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
    # halo_contrast = np.log(halo_contrast+1e-6)


    # 6. 区域内亮度方差（回声不均匀性）
    brightness_variation = 0
    roi_vals = image[mask > 0]
    if len(roi_vals) > 1:
        brightness_variation = np.std(roi_vals)
    # else:
    #     brightness_variation = 0

    # 归一化
    norm_area = area / (h * w)
    norm_perimeter = perimeter / (2 * (h + w))

    reg_feat =  np.array([
        norm_area, aspect_ratio, roundness, norm_perimeter,
        edge_sharpness, lobulation_score, echo_attenuation,
        texture_entropy, halo_contrast, brightness_variation,
        np.mean(image[mask > 0]) / 255.0,  # 平均亮度
        np.max(image[mask > 0]) / 255.0   # 最大亮度
    ], dtype=np.float32)

    return normalize_rich_features(reg_feat)




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

        self.norm12 = nn.LayerNorm(12)  # or BatchNorm1d

    def execute(self, x, mask):
        B = x.shape[0]

        # x_np = (x[:, 0] * 255).sync()
        # mask_np = mask[:, 0].sync()
        # x_np = x_np.numpy().astype("uint8")  # shape: [B, H, W]
        # mask_np = mask_np.numpy().astype("uint8")

        feats = []
        for i in range(B):
            img = (x[i, 0] * 255).numpy().astype("uint8")
            msk = mask[i, 0].numpy().astype("uint8")
            feat = extract_rich_region_features(img, msk)
            # feat = extract_rich_region_features(img, msk)
            feats.append(feat)
        feats = jt.array(feats)  # [B, 12]
        feat_normed = self.norm12(feats)  # [B, 12]
        proj_feat = self.projector(feat_normed)  # [B, 64]
        fused_feat = jt.concat([feat_normed, proj_feat], dim=1)  # [B, 76]
        return fused_feat


# projector = nn.Sequential(
# nn.LayerNorm(12),
#     nn.Linear(12, 64),
#     nn.LayerNorm(64),
#     nn.ReLU()
# )
# sample_input = jt.array([[0.1]*12])
# print("Projected:", projector(sample_input))  # 看是否正常输出



class ROIEncoder(nn.Module):
    """
    模块：提取结构区域图像（等比例缩放+填充），编码为特征，并输出位置编码。
    输出：
        - roi_feat: 图像区域编码特征 [B, 128]
        - position_feat: 区域位置信息 [B, 4]，含中心坐标和尺寸（归一化）
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

    def resize_with_padding(self, crop, target_size=64):
        h, w = crop.shape
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(crop, (new_w, new_h))
        pad_top = (target_size - new_h) // 2
        pad_bottom = target_size - new_h - pad_top
        pad_left = (target_size - new_w) // 2
        pad_right = target_size - new_w - pad_left
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        return padded / 255.

    def execute(self, x, mask):
        B, _, H, W = x.shape
        crops, pos_feats = [], []

        for i in range(B):
            img = x[i, 0].numpy()
            msk = mask[i, 0].numpy()

            if msk.sum() == 0:
                crop = np.zeros((64, 64), dtype=np.float32)
                pos_feat = np.zeros(4, dtype=np.float32)  # [xc, yc, w, h]
            else:
                ys, xs = np.where(msk > 0)
                y1, y2 = max(0, ys.min() - 5), min(H, ys.max() + 5)
                x1, x2 = max(0, xs.min() - 5), min(W, xs.max() + 5)
                region = img[y1:y2, x1:x2]
                crop = self.resize_with_padding(region)

                xc = ((x1 + x2) / 2) / W
                yc = ((y1 + y2) / 2) / H
                ww = (x2 - x1) / W
                hh = (y2 - y1) / H
                pos_feat = np.array([xc, yc, ww, hh], dtype=np.float32)

            crops.append(crop)
            pos_feats.append(pos_feat)

        crops = jt.array(np.stack(crops)).unsqueeze(1)      # [B, 1, 64, 64]
        pos_feats = jt.array(np.stack(pos_feats))           # [B, 4]

        feat = self.encoder(crops).squeeze(-1).squeeze(-1)  # [B, 32]
        roi_feat = self.out_proj(feat)                      # [B, 128]

        return roi_feat, pos_feats  # [B,128], [B,4]



class StructGuidedAttention(nn.Module):
    def __init__(self, struct_dim=64+12+4, vis_dim=2048, hidden_dim=512):
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


class MskCrossAttentionFusion(nn.Module):
    """
    模块：Cross-Attention 融合主图像、结构 DropMask 引导、ROI特征
    """
    def __init__(self, embed_dim=2048, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(256 + 512 + 1024+128*3, embed_dim)
        self.v_proj = nn.Linear(256 + 512 + 1024+128*3, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def execute(self, img_feat, drop_feat, roi_feat):
        # img_feat: [B, 2048], drop_feat: [B, 32 + 64 + 128], roi_feat: [B, 128]
        fusion_kv = jt.concat([drop_feat, roi_feat, roi_feat, roi_feat], dim=1)
        q = self.q_proj(img_feat).unsqueeze(1)     # [B, 1, D]
        k = self.k_proj(fusion_kv).unsqueeze(1)    # [B, 1, D]
        v = self.v_proj(fusion_kv).unsqueeze(1)    # [B, 1, D]
        attn = nn.bmm(q, k.transpose(1, 2)) / (img_feat.shape[1] ** 0.5)  # [B, 1, 1]
        attn = nn.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        fused = nn.bmm(attn, v).squeeze(1)  # [B, D]
        return self.out_proj(fused)


class DirectionalAttention(nn.Module):
    def __init__(self, in_channels, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.query_conv = nn.Conv(in_channels, in_channels // 8, kernel_size=1, bias=False)
        self.key_conv = nn.Conv(in_channels, in_channels // 8, kernel_size=1, bias=False)
        self.value_conv = nn.Conv(in_channels, in_channels, kernel_size=1, bias=False)
        self.gamma = jt.array([0.0]).float32().stop_grad()  # 可训练的权重，初始化为0
        # self.gamma = nn.Parameter(self.gamma)

        # 方向编码（x,y坐标网格）
        xs = jt.arange(0, width).float32().unsqueeze(0).repeat(height, 1)   # [H, W]
        ys = jt.arange(0, height).float32().unsqueeze(1).repeat(1, width)   # [H, W]
        dir_map = jt.stack([xs, ys], dim=0)  # [2, H, W]
        self.register_buffer('direction_map', dir_map)  # 注册为buffer，不参与梯度

    def execute(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert H == self.height and W == self.width, "输入尺寸必须和初始化时一致"

        proj_query = self.query_conv(x).reshape(B, -1, H*W)     # [B, C', N]
        proj_key = self.key_conv(x).reshape(B, -1, H*W)         # [B, C', N]

        energy = jt.bmm(proj_query.transpose(0,2,1), proj_key)  # [B, N, N]

        # 计算方向相似度矩阵 N×N
        dir_vec = self.direction_map.reshape(2, -1)             # [2, N]
        norm = jt.norm(dir_vec, dim=0, keepdims=True) + 1e-6
        dir_normed = dir_vec / norm                              # 单位向量
        dir_sim = jt.matmul(dir_normed.transpose(1,0), dir_normed)  # [N, N]

        # 融合方向相似度与特征相似度
        energy = energy * dir_sim.unsqueeze(0).stop_grad()       # 防止方向相似度梯度传播

        attention = nn.softmax(energy, dim=-1)                   # [B, N, N]

        proj_value = self.value_conv(x).reshape(B, -1, H*W)      # [B, C, N]
        out = jt.bmm(proj_value, attention.transpose(0,2,1))    # [B, C, N]
        out = out.reshape(B, C, H, W)

        out = self.gamma * out + x
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        b, c, _, _ = x.shape
        avg = self.avg_pool(x).view(b, c)
        max_ = self.max_pool(x).view(b, c)
        avg_out = self.fc(avg)
        max_out = self.fc(max_)
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        avg_out = jt.mean(x, dim=1, keepdims=True)
        max_out = jt.max(x, dim=1, keepdims=True)
        x_cat = jt.concat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn

class FPNResidualFusionCBAM(nn.Module):
    def __init__(self, in_channels_list, out_channels, dropout=0.1):
        """
        :param in_channels_list: 各层特征通道数 [c2, c3, c4, gcn]
        :param out_channels: 输出统一通道数
        """
        super().__init__()
        total_in = sum(in_channels_list)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv(in_ch, out_channels, 1, padding=0, bias=False),
                nn.BatchNorm(out_channels),
                nn.Relu()
            ) for in_ch in in_channels_list
        ])

        self.fusion = nn.Sequential(
            nn.Conv(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm(out_channels),
            nn.Relu()
        )


        self.channel_att = ChannelAttention(out_channels)
        self.spatial_att = SpatialAttention()
        self.dropout = nn.Dropout(dropout)

    def execute(self, c2, c3, c4, u):

        #FPN
        target_h, target_w = c2.shape[2], c2.shape[3]
        features = [c2, c3, c4, u]

        out = 0
        for feat, conv in zip(features, self.convs):
            feat_proj = conv(feat)
            feat_up = nn.interpolate(feat_proj, size=[target_h, target_w], mode='bilinear', align_corners=False)
            out = out + feat_up

        fused = self.fusion(out)



        # CBAM注意力
        fused = self.channel_att(fused)
        fused = self.spatial_att(fused)

        # Dropout
        fused = self.dropout(fused)

        return fused

class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv(in_channels, 1, kernel_size=1),  # 输出 shape: [B, 1, H, W]
            nn.Sigmoid()
        )

    def execute(self, img_feat, msk_img_feat):
        gate = self.gate_conv(img_feat)             # [B,1,H,W]
        fused_feat = gate * msk_img_feat + (1 - gate) * img_feat
        return fused_feat



class ResNet101BackboneFusion(nn.Module):
    def __init__(self):
        super().__init__()
        net = resnet101(pretrained=True)

        # # 输入通道重新适配
        # conv1 = nn.Conv(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # with jt.no_grad():
        #     conv1.weight[:, :2] = net.conv1.weight
        #     conv1.weight[:, 2:] = net.conv1.weight.mean(dim=1, keepdims=True)
        # net.conv1 = conv1   #[x,msk,mean(x+msk)]

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out = nn.AdaptiveAvgPool2d((1, 1))

        # ITE + Mamba for c1~c3
        self.ite1, self.ite2, self.ite3 = ITEBlock(256), ITEBlock(512), ITEBlock(1024)
        self.m1,   self.m2,   self.m3   = MambaBranch(256), MambaBranch(512), MambaBranch(1024)

        # mscb1~3 投影
        # self.seg_proj1 = nn.Conv(32, 256, 1)
        # self.seg_proj2 = nn.Conv(64, 512, 1)
        # self.seg_proj3 = nn.Conv(128, 1024, 1)

        # 融合门控
        self.gate1 = GatedFusion(256)
        self.gate2 = GatedFusion(512)
        self.gate3 = GatedFusion(1024)

        self.norm_seg1 = nn.InstanceNorm2d(256)
        self.norm_seg2 = nn.InstanceNorm2d(512)
        self.norm_seg3 = nn.InstanceNorm2d(1024)

        self.norm_c1 = nn.InstanceNorm2d(256)
        self.norm_c2 = nn.InstanceNorm2d(512)
        self.norm_c3 = nn.InstanceNorm2d(1024)

    def execute(self, x, mscb_feats):
        """
        :param x: 输入图像 [B, 4, H, W]（RGB + mask）
        :param mscb_feats: dict，包含 "d1", "d2", "d3" 的特征（来自 seg_model.decoder）
        :return: c2, c3, c4, fused_feat
        """

        x = self.stem(x)                 # [B, 64, H/4, W/4]

        # ---- Layer1（c1） ----
        c1 = self.layer1(x)              # [B, 256, H/4, W/4]
        c1 = self.norm_c1(c1)
        # seg1 = nn.interpolate(self.seg_proj1(mscb_feats["d1"]), size=c1.shape[2:], mode="bilinear")
        seg1 = self.norm_seg1(mscb_feats["d1"])
        c1 = self.gate1(c1, seg1)        # 融合
        c1 = self.m1(self.ite1(c1))

        # ---- Layer2（c2） ----
        c2 = self.layer2(c1)             # [B, 512, H/8, W/8]
        c2 = self.norm_c2(c2)
        # seg2 = nn.interpolate(self.seg_proj2(mscb_feats["d2"]), size=c2.shape[2:], mode="bilinear")
        seg2 = self.norm_seg2(mscb_feats["d2"])
        c2 = self.gate2(c2, seg2)
        c2 = self.m2(self.ite2(c2))

        # ---- Layer3（c3） ----
        c3 = self.layer3(c2)             # [B, 1024, H/16, W/16]
        c3 = self.norm_c3(c3)
        # seg3 = nn.interpolate(self.seg_proj3(mscb_feats["d3"]), size=c3.shape[2:], mode="bilinear")
        seg3 = self.norm_seg3(mscb_feats["d3"])
        c3 = self.gate3(c3, seg3)
        c3 = self.m3(self.ite3(c3))

        # ---- Layer4（c4） ----
        c4 = self.layer4(c3)             # [B, 2048, H/32, W/32]
        out = self.out(c4)              # [B, 2048, 1, 1]

        return c2, c3, c4, out





# ------------------- Main Fusion Model --------------------------
class Res101_Mamba_ITE_UNet_GraphPlus(nn.Module):
    def __init__(self, num_classes=Config.num_classes,in_channels=3,freeze_seg=True, seg_ckpt="busi_seg_ckpt.pkl"):
        super().__init__()
        Config.num_classes=num_classes
        # self.backbone = ResNet101Backbone(in_channels=in_channels)
        self.backbone = ResNet101BackboneFusion()

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


        # self.ite2, self.ite3 = ITEBlock(512), ITEBlock(1024)
        # self.ite4 = ITEBlock(2048)
        # self.m2, self.m3   = MambaBranch(512), MambaBranch(1024)
        # self.m4 = nn.Sequential(
        #     nn.Conv(2048, 1024, 1),
        #     nn.BatchNorm(1024),
        #     nn.ReLU(),
        #     nn.Conv(1024, 512, 1),
        #     nn.BatchNorm(512),
        #     nn.ReLU(),
        #     nn.Conv(512, 256, 1),
        #     # nn.BatchNorm(256)
        # )
        # self.seg_model          = nnUNet2D(in_channels=1, n_classes=2)

        self.seg_model = EMCADNet(encoder="resnet101")

        self.seg_model.load(str(seg_ckpt))
        if freeze_seg:
            for param in self.seg_model.parameters():
                param.stop_grad()

        self._d_feats = {}

        def save_hook(name):
            def fn(module, x, y):
                self._d_feats[name] = y

            return fn

        self.seg_model.decoder.mscb1.register_forward_hook(save_hook("d1"))
        self.seg_model.decoder.mscb2.register_forward_hook(save_hook("d2"))
        self.seg_model.decoder.mscb3.register_forward_hook(save_hook("d3"))
        self.seg_model.decoder.mscb4.register_forward_hook(save_hook("d4"))


        self.gcn           = SimpleGCN(2048, 1024)
        self.vgt           = VisualGeneralizationToken(512)

        # prototype
        self.proto         = GraphPrototype(n_proto=16, feat_dim=2048)
        # reduce
        # self.red2, self.red3 = nn.Conv(512,256,1), nn.Conv(1024,256,1)
        # self.fuse   = nn.Sequential(nn.Conv(256*3+512, 256, 3, padding=1, bias=False), nn.BatchNorm(256), nn.GELU())
        self.pool   = nn.AdaptiveAvgPool2d(1)
        # self.cls    = nn.Linear(256, num_classes*Config.score_levels)
        self.table  = jt.arange(1,Config.score_levels+1).float32()

        # self.proto_proj = nn.Linear(2048, 256)  # 把 128‑D prototype 提升到 256‑D

        self.cross_fusion = CrossAttentionFusion(
            q_dim=512, kv_dim=2048, hidden_dim=1024, num_heads=4)

        # self.cls_head = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, 512)
        # )

        # self.roi_proj = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, 512)
        # )

        # self.proj_fused = nn.Linear(512, 256)
        # self.proj_unet = nn.Conv(2048,512,1)

        self.rich_encoder_res = RichStructEncoder()
        self.roi_encoder = ROIEncoder()
        self.struct_attn = StructGuidedAttention()

        self.fusion_attn = MskCrossAttentionFusion()

        self.cls = nn.Linear(512, num_classes * Config.score_levels)
        self.proj_fused = nn.Sequential(
            nn.Linear(2048 + 512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512)
        )
        # self.cls = AAMHead(256, num_classes)

        self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=64+12+4)  # 用于 rich_struct_feat

        self.rich_proj = nn.Linear(64+12+4, 512)

        self.dir_attn_c4 = DirectionalAttention(in_channels=2048, height=14, width=14)

        self.gated_fusion = GatedFusion(in_channels=2048)  # 如果你的主干输出是512通道

        self.fusion_block = FPNResidualFusionCBAM(
            in_channels_list=[512, 1024, 2048, 2048],  # 示例：c2, c3, c4, u 的通道数
            out_channels=512
        )

    # -------- Helper --------
    def score_softmax(self, logits):
        B = logits.shape[0]
        return nn.softmax(logits.reshape(B, Config.num_classes, Config.score_levels), dim=2)

    # -------- Forward --------
    def execute(self, x):
        # ========== 图像预处理 ==========
        x_rgb = jt.concat([x, x, x], dim=1)


        # ========== 分割模型提取 mask & mscb ==========
        outs = self.seg_model(x)
        mask_logits = outs[0]  # [B,2,H,W]
        mask = mask_logits.argmax(dim=1, keepdims=True)[0]  # [B,1,H,W]

        mscb_feats = {
            "d1": self._d_feats["d1"],  # [B,256,H,W]
            "d2": self._d_feats["d2"],  # [B,512,H/2,W/2]
            "d3": self._d_feats["d3"],  # [B,1024,H/4,W/4]
            "d4": self._d_feats["d4"],  # [B,2048,H/8,W/8]
        }

        x_input = jt.concat([x, mask, x*mask], dim=1)  # [B, 4, H, W] — RGB + Mask
        # ========== 主干提取特征 c2~c4（已融合 ITE/Mamba/Seg） ==========
        c2, c3, c4, msk_img_feat = self.backbone(x_input, mscb_feats)
        #msk_img_feat = msk_img_feat.squeeze(-1).squeeze(-1)


        # ========== 原图特征 + DropMask 引导 ==========
        img_feat = self.img_encoder(x_rgb)#.squeeze(-1).squeeze(-1)  # [B, 2048]
        img_fused_feat = self.gated_fusion(img_feat, msk_img_feat).squeeze(-1).squeeze(-1)  # [B,2048]

        # ========== DropFeat + ROI + Rich Struct ==========
        d1 = nn.AdaptiveAvgPool2d((1, 1))(mscb_feats["d1"]).squeeze(-1).squeeze(-1)
        d2 = nn.AdaptiveAvgPool2d((1, 1))(mscb_feats["d2"]).squeeze(-1).squeeze(-1)
        d3 = nn.AdaptiveAvgPool2d((1, 1))(mscb_feats["d3"]).squeeze(-1).squeeze(-1)
        drop_feat = jt.concat([d1, d2, d3], dim=1)  # [B,224]

        rich_struct_feat = self.rich_encoder_res(x,mask)  # [B,76]
        roi_feat, pos_feat = self.roi_encoder(x, mask)  # [B,128], [B,4]
        rich_struct_feat = jt.concat([rich_struct_feat, pos_feat], dim=1)  # [B,80]

        # ========== 方向注意力 + GCN结构 ==========
        c4 = self.dir_attn_c4(c4)  # [B,2048,H/32,W/32]

        u = mscb_feats["d4"]  # [B,2048,H,W]
        # u = self.proj_unet(u)
        # u = nn.interpolate(u, size=c2.shape[2:], mode='bilinear', align_corners=False)
        u = self.gcn(u)  # [B,2048,H/8,W/8]

        # ========== FPN融合结构 ==========
        fused_feat = self.fusion_block(c2, c3, c4, u)  # [B, 512, H, W]
        fused_feat = self.vgt(fused_feat)  # [B, 512, H, W]
        feat = self.pool(fused_feat).reshape(x.shape[0], -1)  # [B, 512]

        # ========== Prototype引导 ==========
        graph_feat = u.reshape(x.shape[0], 2048, -1).transpose(0, 2, 1)  # [B,N,D]
        proto_vec = self.proto(graph_feat)  # [B,2048]

        feat = self.cross_fusion(feat, proto_vec)  # CrossAttention融合  [B,512]

        # ========== 多源融合：结构/Drop/ROI ==========
        feat_msk = self.fusion_attn(img_fused_feat, drop_feat, roi_feat) # [B,2048]
        struct_feat = self.struct_attn(rich_struct_feat, img_fused_feat)  # [B,2048] → [B,2048]
        all_feat = feat_msk + struct_feat  # [B,2048] + [B,2048]

        # feat_proj = self.cls_head(all_feat)

        fused_feat = self.proj_fused(jt.concat([all_feat, feat], dim=1))  # [B,512] → [B,256]

        # ========== 分类 ==========
        logits = self.cls(fused_feat)  # [B,num_classes*score_levels]
        score_mat = self.score_softmax(logits)
        scores = jt.matmul(score_mat, self.table)  # [B,num_classes]

        return {
            'fused_feat': fused_feat,
            'scores': scores,
            'score_mat': score_mat,
            'rich_struct_feat': rich_struct_feat,
            'gcn_feat': u,
            'mask': mask
        }


# ------------------ Losses --------------------
main_criterion  = nn.cross_entropy_loss
score_criterion = ScoreSoftmaxLoss_KL()
ce_loss = nn.cross_entropy_loss

def compute_loss(scores, score_mat, labels,num_classes=Config.num_classes, dom_logits=None, domain_labels=None, alpha=0.5, beta=0.1):
    loss = main_criterion(scores, labels) + alpha*score_criterion(score_mat, labels,num_classes=num_classes)
    if dom_logits is not None and domain_labels is not None:
        loss += beta*ce_loss(dom_logits, domain_labels)
    return loss


def sobel_filter(img):
    # img: [B, 1, H, W] tensor
    sobel_x = jt.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).reshape(1, 1, 3, 3).float()
    sobel_y = jt.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape(1, 1, 3, 3).float()
    gx = nn.conv2d(img, sobel_x, padding=1)
    gy = nn.conv2d(img, sobel_y, padding=1)
    grad = jt.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    return grad

def boundary_loss(pred_feat, mask):
    # pred_feat: [B, C, H, W]
    # mask: [B, 1, H, W]
    feat_grad = sobel_filter(pred_feat.mean(dim=1, keepdims=True))
    mask_grad = sobel_filter(mask)
    loss = nn.L1Loss()(feat_grad, mask_grad)
    return loss



def compute_fourier_descriptor(mask_np, degree=10):
    # mask_np: numpy array [H, W], binary mask
    contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return np.zeros((degree * 2,), dtype=np.float32)

    contour = max(contours, key=lambda x: len(x))  # longest one
    contour = contour.squeeze(1)
    complex_contour = contour[:, 0] + 1j * contour[:, 1]
    fft_desc = np.fft.fft(complex_contour)
    fft_desc = np.concatenate([np.real(fft_desc[1:degree+1]), np.imag(fft_desc[1:degree+1])])
    return fft_desc.astype(np.float32)

class FourierShapeLoss(nn.Module):
    def __init__(self, degree=10, feat_dim=512):
        super().__init__()
        self.degree = degree
        self.proj = nn.Linear(feat_dim, degree * 2)

    def execute(self, rich_feat, mask_list):
        # rich_feat: [B, D]
        # mask_list: list of binary masks (numpy)
        B = rich_feat.shape[0]
        gt_fd = [compute_fourier_descriptor(mask_list[i], self.degree) for i in range(B)]
        gt_fd = jt.array(np.stack(gt_fd))  # [B, 2*degree]
        pred_fd = self.proj(rich_feat)     # [B, 2*degree]
        return nn.MSELoss()(pred_fd, gt_fd)



def get_mask_principal_direction(mask_np):
    coords = np.argwhere(mask_np > 0)
    if coords.shape[0] < 2:
        return np.zeros(2, dtype=np.float32)
    coords = coords - coords.mean(0)
    cov = coords.T @ coords
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, -1]
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    return direction.astype(np.float32)

class DirectionLoss(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.head = nn.Linear(in_dim, 2)

    def execute(self, fused_feat, mask_list):
        # fused_feat: [B, D]
        # mask_list: list of [H, W] masks
        B = fused_feat.shape[0]
        gt_dirs = [get_mask_principal_direction(mask_list[i]) for i in range(B)]
        gt_dirs = jt.array(np.stack(gt_dirs))  # [B, 2]
        pred_dirs = self.head(fused_feat)      # [B, 2]
        pred_dirs = pred_dirs / (jt.norm(pred_dirs, dim=1, keepdim=True) + 1e-6)
        return nn.MSELoss()(pred_dirs, gt_dirs)

class ContourTokenCrossAttention(nn.Module):
    def __init__(self, dim=512, num_heads=4):
        super().__init__()
        self.contour_token = jt.randn(1, 1, dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def execute(self, x):
        # x: [B, N, D] (rich_feat sequence)
        B = x.shape[0]
        ctok = self.contour_token.expand(B, -1, -1)  # [B, 1, D]
        x_aug = jt.concat([ctok, x], dim=1)          # [B, N+1, D]
        attn_out, _ = self.attn(x_aug, x_aug, x_aug) # [B, N+1, D]
        return attn_out[:, 0], attn_out[:, 1:]       # 返回 contour token + refined feat

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = jt.randn(num_classes, feat_dim) * 0.1
    def execute(self, features, labels):
        centers_batch = self.centers[labels]
        return ((features - centers_batch) ** 2).sum(dim=1).mean()

# center_loss=CenterLoss(Config.num_classes,256)



def compute_total_loss(model,scores, score_mat, labels,
                       rich_feat, fused_feat,
                       gcn_feat=None,num_classes=Config.num_classes):

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

    # 汇总
    loss = loss_cls + 1 * loss_score + 0.2 * loss_struct_align + \
           0.2 * loss_struct_center + 0.5 * loss_uncertainty

    # print(loss_cls,loss_score,loss_struct_align,loss_struct_center,loss_uncertainty,loss_domain)

    # return loss
    return loss, loss_cls, loss_score, loss_struct_align, loss_struct_center, loss_uncertainty



__all__ = ["Res101_Mamba_ITE_UNet_GraphPlus", "compute_loss","compute_total_loss"]
