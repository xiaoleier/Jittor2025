# nnunet_jittor.py
"""
An **nnU‑Net‑style** training template in **Jittor** adapted for the
**BUS‑BRA breast‑ultrasound segmentation dataset** (2‑D grayscale images).

Main features
-------------
1. **2‑D U‑Net backbone (5 stages)** with skip connections + 2 deep‑supervision
   heads (configurable).
2. Combined **Dice + Cross‑Entropy** loss and weighted deep‑supervision.
3. **`BusBraDataset`** that reads PNG images & masks from the BUS‑BRA layout
   (\*/Images\/*.png, \*/Masks\/*.png).
4. Lightweight **training / validation loop**, multi‑GPU friendly (`jt.flags.use_cuda`).
5. Clearly marked hooks for data augmentation & hyper‑params.

Replace the dataset root & adjust hyper‑parameters in the `if __name__ == "__main__"` section.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path
from typing import List, Tuple, Optional

import jittor as jt
from jittor import nn
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import random

jt.flags.use_cuda = 1
# jt.flags.device_id = 0
# -----------------------------
# Conv blocks (2‑D)
# -----------------------------
class DoubleConv(nn.Module):
    """(Conv → InstanceNorm → LeakyReLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(scale=0.01),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(scale=0.01),
        )

    def execute(self, x):
        return self.block(x)


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def execute(self, x):
        return nn.pool(x, kernel_size=self.kernel_size, stride=self.stride, op='maximum')


def pool2d():
    return MaxPool2d(kernel_size=2, stride=2)


def upsample2d(in_ch: int, out_ch: int):
    return nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2, bias=False)


# -----------------------------
# nnU‑Net 2‑D backbone
# -----------------------------
class nnUNet2D(nn.Module):
    """Full‑resolution 2‑D nnU‑Net with deep supervision."""

    def __init__(self, in_channels: int = 1, n_classes: int = 2, base_filters: int = 32):
        super().__init__()
        fs = base_filters

        # Encoder
        self.enc1 = DoubleConv(in_channels, fs)
        self.pool1 = pool2d()
        self.enc2 = DoubleConv(fs, fs * 2)
        self.pool2 = pool2d()
        self.enc3 = DoubleConv(fs * 2, fs * 4)
        self.pool3 = pool2d()
        self.enc4 = DoubleConv(fs * 4, fs * 8)
        self.pool4 = pool2d()
        self.bottleneck = DoubleConv(fs * 8, fs * 16)

        # Decoder
        self.up4 = upsample2d(fs * 16, fs * 8)
        self.dec4 = DoubleConv(fs * 16, fs * 8)
        self.up3 = upsample2d(fs * 8, fs * 4)
        self.dec3 = DoubleConv(fs * 8, fs * 4)
        self.up2 = upsample2d(fs * 4, fs * 2)
        self.dec2 = DoubleConv(fs * 4, fs * 2)
        self.up1 = upsample2d(fs * 2, fs)
        self.dec1 = DoubleConv(fs * 2, fs)

        # Output heads
        self.out_main = nn.Conv2d(fs, n_classes, 1)
        self.out_ds2 = nn.Conv2d(fs * 2, n_classes, 1)
        self.out_ds3 = nn.Conv2d(fs * 4, n_classes, 1)

    def execute(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = self.dec4(jt.concat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(jt.concat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(jt.concat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(jt.concat([d1, e1], dim=1))

        out_main = self.out_main(d1)
        out_ds2 = nn.interpolate(self.out_ds2(d2), size=out_main.shape[2:], mode="bilinear", align_corners=False)
        out_ds3 = nn.interpolate(self.out_ds3(d3), size=out_main.shape[2:], mode="bilinear", align_corners=False)

        return out_main, out_ds2, out_ds3


# -----------------------------
# Losses
# -----------------------------
def one_hot(target: jt.Var, num_classes: int) -> jt.Var:
    return (target.unsqueeze(1) == jt.arange(num_classes).reshape(1, num_classes, 1, 1)).float()

class DiceCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def execute(self, pred: jt.Var, target: jt.Var):
        ce_loss = self.ce(pred, target)
        probs = nn.softmax(pred, dim=1)
        num_classes = probs.shape[1]
        true_1_hot = one_hot(target, num_classes)

        # true_1_hot = (target.unsqueeze(1) == jt.arange(num_classes).reshape(1, num_classes, 1, 1)).float()

        intersection = jt.sum(probs * true_1_hot, dims=(2, 3))
        cardinality = jt.sum(probs + true_1_hot, dims=(2, 3))
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice.mean()
        return ce_loss + dice_loss


def deep_supervision_loss(preds: Tuple[jt.Var, ...], target: jt.Var, weights=(1.0, 0.5, 0.25)) -> jt.Var:
    loss_fn = DiceCELoss()
    return sum(w * loss_fn(p, target) for p, w in zip(preds, weights))


# -----------------------------
# BUS‑BRA Dataset loader
# -----------------------------

class BreastCSVSet(jt.dataset.Dataset):
    """
    将 bus_data.csv 直接喂给模型的 Dataset。
    如果 csv 含 'split' 列，就按列筛选；否则用 train_test_split 切分。
    """
    def __init__(self, csv_path, mode='train', split_ratio=0.1,
                 resize:Optional[Tuple[int,int]]=(512,512), augment:bool=False,random_state=42, **kwargs):
        super().__init__(**kwargs)

        self.resize=resize
        self.augment=augment

        # self.transform = transform
        df = pd.read_csv(csv_path)

        # df = df[df['Pathology'] == 'malignant']

        # 判断是训练/验证/测试
        if 'split' in df.columns:
            df = df[df['split'] == mode]
        elif mode in ['train', 'val']:
            train_df, val_df = train_test_split(
                df, test_size=split_ratio, stratify=df['BIRADS'], random_state=random_state)
            df = train_df if mode == 'train' else val_df

        # ———推理模式：没有 BIRADS 就走这里———
        self.infer = 'BIRADS' not in df.columns

        # 建立 BIRADS → idx 的映射（训练/验证用）
        if not self.infer:
            classes = sorted(df['BIRADS'].unique())
            self.label2idx = {c: i for i, c in enumerate(classes)}

        # 转成 list 方便 __getitem__
        self.samples = df.to_dict('records')
        self.total_len = len(self.samples)


    def _load_img(self, path: Path):
        img = Image.open(path).convert("L")  # grayscale
        if self.resize:
            img = img.resize(self.resize, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # normalize 0‑1
        return jt.array(arr)[None, ...]  # [1, H, W]

    def _load_mask(self, path: Path):
        m = Image.open(path).convert("L")
        if self.resize:
            m = m.resize(self.resize, Image.NEAREST)
        arr = (np.array(m) > 0).astype(np.int32)  # binary mask
        return jt.array(arr)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        dir_path="./datasets/BUSBRA/Images/"
        self.mask_root="./datasets/BUSBRA/Masks/"
        # img_path = os.path.join(rec['dir'], rec['ID'])
        img_path = dir_path + rec['ID']+".png"
        mask_id = rec['ID'].replace("bus", "mask")
        mask_path = self.mask_root + mask_id + ".png"
        image = self._load_img(img_path)
        mask = self._load_mask(mask_path)

        # --- optional augmentations (flip, rotate) ---
        if self.augment:
            if np.random.rand() < 0.5:
                image = jt.flip(image, dim=[2])
                mask = jt.flip(mask, dim=[1])
            if np.random.rand() < 0.5:
                image = jt.flip(image, dim=[1])
                mask = jt.flip(mask, dim=[0])

        #
        # if self.infer:
        #     label = rec['ID']              # 推理：返回文件名
        # else:
        #     label = self.label2idx[rec['BIRADS']]  # 训练/验证：返回整数标签

        return image, mask

    def __len__(self):
        return self.total_len


# -----------------------------
# Dataset_BUSI_with_GT loader
# -----------------------------
class BusiDataset(jt.dataset.Dataset):
    def __init__(self, root: str, mode: str = 'train', split_ratio=0.1, resize=(512, 512), augment=False, random_state=42):
        super().__init__()
        self.resize = resize
        self.augment = augment

        all_imgs = []
        for subdir in ['benign', 'malignant', 'normal']:
        # for subdir in ['malignant']:
            img_dir = Path(root) / subdir
            for f in img_dir.glob("*.png"):
                if "_mask" not in f.stem:
                    mask_f = f.parent / f.with_stem(f.stem + "_mask").name
                    if mask_f.exists():
                        all_imgs.append((str(f), str(mask_f)))

        # 划分 train/val
        np.random.seed(random_state)
        np.random.shuffle(all_imgs)
        split_idx = int(len(all_imgs) * (1 - split_ratio))
        if mode == 'train':
            self.pairs = all_imgs[:split_idx]
        else:
            self.pairs = all_imgs[split_idx:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("L")
        if self.resize:
            image = image.resize(self.resize, Image.BILINEAR)
        image = jt.array(np.array(image, dtype=np.float32) / 255.0)[None, ...]  # [1, H, W]

        mask = Image.open(mask_path).convert("L")
        if self.resize:
            mask = mask.resize(self.resize, Image.NEAREST)
        mask = jt.array((np.array(mask) > 0).astype(np.int32))

        if self.augment:
            if np.random.rand() < 0.5:
                image = jt.flip(image, dim=[2])
                mask = jt.flip(mask, dim=[1])
            if np.random.rand() < 0.5:
                image = jt.flip(image, dim=[1])
                mask = jt.flip(mask, dim=[0])

        return image, mask






class BusuclmDataset(jt.dataset.Dataset):
    def __init__(self, root: str, mode: str = 'train', split_ratio=0.1, resize=(512, 512), augment=False, random_state=42):
        super().__init__()
        self.resize = resize
        self.augment = augment

        root = Path(root)
        img_dir = root / "images"
        # img_root = Path("./BUSUCLM/bus_uclm_separated")
        # img_dir = img_root / "malign"
        mask_dir = root / "masks"

        all_pairs = []
        for img_path in img_dir.glob("*.png"):
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                all_pairs.append((str(img_path), str(mask_path)))

        # 划分 train / val
        np.random.seed(random_state)
        np.random.shuffle(all_pairs)
        split_idx = int(len(all_pairs) * (1 - split_ratio))
        if mode == 'train':
            self.pairs = all_pairs[:split_idx]
        else:
            self.pairs = all_pairs[split_idx:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # 加载图像（灰度）
        image = Image.open(img_path).convert("L")
        if self.resize:
            image = image.resize(self.resize, Image.BILINEAR)
        image = jt.array(np.array(image, dtype=np.float32) / 255.0)[None, ...]  # [1, H, W]

        # 加载 mask 并将所有非黑色像素变为白色（即前景为255）
        mask = Image.open(mask_path).convert("RGB")
        mask = np.array(mask)
        mask = np.any(mask != [0, 0, 0], axis=-1).astype(np.uint8) * 255  # 所有非黑色设为255
        mask = Image.fromarray(mask).convert("L")
        if self.resize:
            mask = mask.resize(self.resize, Image.NEAREST)
        mask = jt.array((np.array(mask) > 0).astype(np.int32))  # 01 mask

        # 数据增强
        if self.augment:
            if np.random.rand() < 0.5:
                image = jt.flip(image, dim=[2])
                mask = jt.flip(mask, dim=[1])
            if np.random.rand() < 0.5:
                image = jt.flip(image, dim=[1])
                mask = jt.flip(mask, dim=[0])

        return image, mask





# -----------------------------
# CombinedDataset
# -----------------------------
class CombinedDataset(jt.dataset.Dataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets
        self.cum_sizes = np.cumsum([len(d) for d in datasets])

    def __len__(self):
        return self.cum_sizes[-1]

    def __getitem__(self, idx):
        ds_idx = np.searchsorted(self.cum_sizes, idx, side='right')
        prev = 0 if ds_idx == 0 else self.cum_sizes[ds_idx - 1]
        return self.datasets[ds_idx][idx - prev]






# -----------------------------
# Trainer
# -----------------------------

def train(model: nn.Module, loader: jt.dataset.Dataset, val_loader: jt.dataset.Dataset, epochs: int = 150, lr: float = 1e-3, out_dir: str = "ckpts"):
    opt = jt.optim.Adam(model.parameters(), lr)
    best_val = 1e9
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        colors = [31, 32, 33, 34, 35, 36]
        color = random.choice(colors)
        bar_format = f"\033[{color}m{{l_bar}}{{bar}}{{r_bar}}\033[0m"
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}",bar_format=bar_format)

        for imgs, masks in pbar:
            preds = model(imgs)
            loss = deep_supervision_loss(preds, masks)
            opt.zero_grad()
            loss.sync()
            opt.step(loss)
            total_loss += loss.item()
        jt.sync_all()
        print(f"Epoch {epoch:3d}/{epochs} | train‑loss = {total_loss / len(loader):.4f}")

        # ---- Validation ----
        # if val_loader and epoch % 5 == 0:
        if val_loader:
            model.eval()
            with jt.no_grad():
                val_loss = 0.0

                colors = [31, 32, 33, 34, 35, 36]
                color = random.choice(colors)
                bar_format = f"\033[{color}m{{l_bar}}{{bar}}{{r_bar}}\033[0m"
                pbar = tqdm(val_loader,  desc=f"Validating",bar_format=bar_format)

                for imgs, masks in pbar:
                    val_loss += deep_supervision_loss(model(imgs), masks).item()
            val_loss /= len(val_loader)
            print(f"                 val‑loss   = {val_loss:.4f}")
            # save best
            if val_loss < best_val:
                best_val = val_loss
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                jt.save(model.state_dict(), str(Path(out_dir) / "best_model.pkl"))
                # jt.save(model.state_dict(), str(Path(out_dir) / "nnunet_best_modelv3_malign.pkl"))

                # model.save(Path(out_dir) / "best_model.pkl")
                print("                 ✔ saved new best model")


# -----------------------------
# Entry
# -----------------------------

# -------------------------------------------------
# Inference helper (单张图片 → 分割掩码/可视化)
# -------------------------------------------------
def segment_image(
    image_path: str,
    ckpt_path: str,
    output_overlay: str,
    output_mask: str = "mask.png",
    resize: tuple[int, int] = (512, 512),
    threshold: float = 0.5,
):
    """
    Args
    ----
    image_path : 输入原图（灰度 PNG）
    ckpt_path  : 训练好的权重 .pkl
    output_mask: 保存二值掩码位置
    output_overlay: 保存原图+红色掩码叠加，可为 None
    resize     : 输入网络的尺寸，需与训练一致
    threshold  : 前景概率阈值（默认取 argmax，不用该参数）
    """
    # 1) 建网络并加载权重
    net = nnUNet2D(in_channels=1, n_classes=2)
    net.load(str(ckpt_path))          # jt.load → .load(state_dict)
    net.eval()

    # 2) 读入并预处理图片
    img_pil = Image.open(image_path).convert("L")
    if resize:
        img_pil = img_pil.resize(resize, Image.BILINEAR)
    arr = np.array(img_pil, dtype=np.float32) / 255.0
    x = jt.array(arr)[None, None, ...]   # [1,1,H,W]

    # 3) 推理
    with jt.no_grad():
        logits, _, _ = net(x)
        pred = logits.argmax(dim=1)[0]   # [H,W] 0/1
        mask_np = (pred.numpy() * 255).astype(np.uint8)

    mask_np = (pred.numpy() * 255).astype(np.uint8)
    mask_np = mask_np.squeeze()  # <- 关键：去掉多余维度，得到 (H, W)

    # 4) 保存二值掩码
    Image.fromarray(mask_np).save(output_mask)
    print(f"mask saved → {output_mask}")

    # 5) 可选：叠加红色透明掩码保存
    if output_overlay:
        overlay = Image.fromarray(np.stack([mask_np]*3, -1))   # 三通道
        overlay = overlay.convert("RGBA")
        overlay_np = np.array(overlay)
        overlay_np[..., 0] = 255      # R 通道
        overlay_np[..., 1:] = 0       # G,B 通道
        overlay_np[..., 3] = (mask_np > 0) * 120  # α 通道（透明度）
        overlay = Image.fromarray(overlay_np)

        # base = img_pil.convert("RGBA")
        base = img_pil.resize(mask_np.shape[::-1]).convert("RGBA")
        blended = Image.alpha_composite(base, overlay)
        blended.save(output_overlay)
        print(f"overlay saved → {output_overlay}")


# -----------------------------
# CLI 支持
# -----------------------------
# if __name__ == "__main__":
#     import argparse, sys
#
#     parser = argparse.ArgumentParser(description="nnU‑Net 2‑D inference")
#     parser.add_argument("--image", type=str, default="/home/yangxiaohui/ll/jittor_challenge/baseline/TrainSet/images/train/28.jpg",help="Path to PNG image")
#     parser.add_argument("--ckpt", type=str, default="./ckpts/best_model.pkl",help="Path to best_model.pkl")
#     parser.add_argument("--mask_out", type=str, default="mask.png")
#     parser.add_argument("--overlay_out", type=str, default="overlay.png")
#     args, _ = parser.parse_known_args()
#
#     if args.image and args.ckpt:
#         segment_image(
#             image_path=args.image,
#             ckpt_path=args.ckpt,
#             output_mask=args.mask_out,
#             output_overlay=args.overlay_out,
#         )
#         sys.exit(0)




# -----------------------------
# 推理函数
# -----------------------------
def segment_folder(net, input_dir, output_mask_dir, output_overlay_dir, resize=(512, 512)):
    input_dir = Path(input_dir)
    output_mask_dir = Path(output_mask_dir)
    output_overlay_dir = Path(output_overlay_dir)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    output_overlay_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(input_dir.glob("*.[jp][pn]g"))
    net.eval()
    with jt.no_grad():
        for img_path in tqdm(img_paths, desc="Segmenting"):
            img = Image.open(img_path).convert("L").resize(resize, Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            x = jt.array(arr)[None, None, ...]  # [1, 1, H, W]
            out, _, _ = net(x)
            pred = out.argmax(dim=1)[0]
            mask_np = (pred.numpy() * 255).astype(np.uint8)
            mask_np = mask_np.squeeze()

            # 保存 mask
            Image.fromarray(mask_np).save(output_mask_dir / img_path.name)

            # 保存 overlay
            rgba = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
            rgba[..., 0] = 255
            rgba[..., 3] = (mask_np > 0) * 120
            overlay = Image.fromarray(rgba, mode="RGBA")
            base = img.convert("RGBA")
            base = base.resize(mask_np.shape[::-1])
            # Image.alpha_composite(base, overlay).save(output_overlay_dir / img_path.name)

            blended = Image.alpha_composite(base, overlay)
            overlay_path = output_overlay_dir / f"{img_path.stem}_overlay.png"
            blended.save(overlay_path)


# -----------------------------
# Entry
# -----------------------------
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--busi_root", type=str, help="Path to Dataset_BUSI_with_GT")
#     # parser.add_argument("--infer_images", type=str, default="TestSet/images/TestSetA",help="Path to folder of inference images (e.g., TrainSet/images/train)")
#     # parser.add_argument("--infer_mask", type=str, default="TestSet/mask", help="Path to save predicted masks")
#     # parser.add_argument("--infer_overlay", type=str, default="TestSet/overlay", help="Path to save overlay images")
#     parser.add_argument("--infer_images", type=str, default="TrainSet/images/train",
#                         help="Path to folder of inference images (e.g., TrainSet/images/train)")
#     parser.add_argument("--infer_mask", type=str, default="TrainSet/mask_nuet", help="Path to save predicted masks")
#     parser.add_argument("--infer_overlay", type=str, default="TrainSet/overlay_nuet", help="Path to save overlay images")
#
#     # parser.add_argument("--infer_images", type=str,
#     #                     default="./BUSUCLM/bus_uclm_separated/malign",
#     #                     help="Path to folder of inference images (e.g., TrainSet/images/train)")
#     # parser.add_argument("--infer_images", type=str, default="./BUSUCLM/BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM/images",
#     #                     help="Path to folder of inference images (e.g., TrainSet/images/train)")
#     # parser.add_argument("--infer_mask", type=str, default="./BUSUCLM/mask_nuet", help="Path to save predicted masks")
#     # parser.add_argument("--infer_overlay", type=str, default="./BUSUCLM/overlay_nuet",
#     #                     help="Path to save overlay images")
#
#
#     parser.add_argument("--ckpt", type=str, default="ckpts/nnunet_best_modelv3_malign.pkl", help="Path to model checkpoint")
#     args = parser.parse_args()
#
#     jt.flags.use_cuda = 1 if jt.has_cuda else 0
#     net = nnUNet2D(in_channels=1, n_classes=2)
#     net.load(args.ckpt)
#
#     if args.infer_images:
#         segment_folder(net, args.infer_images, args.infer_mask, args.infer_overlay)
#         print("✅ Inference completed.")
#     else:
#         print("❌ No --infer_images path provided.")






if __name__ == "__main__":
    import argparse

    jt.flags.use_cuda = 1 if jt.has_cuda else 0

    parser = argparse.ArgumentParser(description="nnU‑Net 2‑D (Jittor) on BUS‑BRA")
    parser.add_argument("--csv_path", type=str, default="./datasets/BUSBRA/bus_data.csv",
                        help="Path to BUS‑BRA root containing train/val/test folders")
    parser.add_argument("--busi_root", type=str, default="./datasets/Dataset_BUSI_with_GT",
                        help="Path to Dataset_BUSI_with_GT")
    parser.add_argument("--busuclm", type=str, default="./datasets/BUS-UCLM", help="Path to Dataset_BUSI_with_GT")

    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)

    # parser.add_argument("--ckpt", type=str, default="ckpts/nnunet_best_modelv3_malign.pkl",
    #                     help="Path to model checkpoint")

    args = parser.parse_args()

    # dataset
    # train_ds = BusBraDataset(args.data_root, "train", augment=True)
    # val_ds = BusBraDataset(args.data_root, "val", augment=False)

    busbra_train = BreastCSVSet(args.csv_path, mode='train', augment=True,
                             batch_size=4, num_workers=8, shuffle=True)
    busbra_val = BreastCSVSet(args.csv_path, mode='val', augment=False,
                           batch_size=4, num_workers=8, shuffle=False)
    busi_train = BusiDataset(args.busi_root, mode='train', augment=True)
    busi_val = BusiDataset(args.busi_root, mode='val', augment=False)

    busuclm_train = BusuclmDataset(root=args.busuclm, mode='train', resize=(512, 512), augment=True)
    busuclm_val = BusuclmDataset(root=args.busuclm, mode='val', resize=(512, 512), augment=False)

    train_ds = CombinedDataset(busi_train, busbra_train)
    val_ds = CombinedDataset(busi_val, busbra_val)

    train_ds = CombinedDataset(train_ds, busuclm_train)
    val_ds = CombinedDataset(val_ds, busuclm_val)

    # train_ds = CombinedDataset(busbra_train, busuclm_train)
    # val_ds = CombinedDataset(busbra_val, busuclm_val)





    train_loader = train_ds.set_attrs(batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = val_ds.set_attrs(batch_size=args.batch, shuffle=False, num_workers=2)

    # model
    net = nnUNet2D(in_channels=1, n_classes=2)
    # if args.ckpt:
    #     net.load(args.ckpt)

    # training
    train(net, train_loader, val_loader, epochs=args.epochs, lr=args.lr)

    print("Training done ✅")
