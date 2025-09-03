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
import math
import os
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



__all__=["nnUNet2D","deep_supervision_loss"]