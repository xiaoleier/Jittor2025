import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset
from jittor import transform
from jittor.transform import Compose, Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, ToTensor, ImageNormalize

from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
from PIL import Image
import argparse
import random
import pandas as pd
from sklearn.model_selection import train_test_split


from ckpt2_models_utils import compute_loss,Res101_Mamba_ITE_UNet_GraphPlus,compute_total_loss




# jt.flags.device_id = 1  # 选择第一个GPU设备
jt.flags.use_cuda = 1

# ============== Dataset ==============
class BreastCSVSet(jt.dataset.Dataset):
    """
    将 bus_data.csv 直接喂给模型的 Dataset。
    如果 csv 含 'split' 列，就按列筛选；否则用 train_test_split 切分。
    """
    def __init__(self, csv_path, mode='train', split_ratio=0.1,
                 transform=None, augment=False,random_state=42, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform
        self.augment = augment
        df = pd.read_csv(csv_path)

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

    def __getitem__(self, idx):
        rec = self.samples[idx]
        dir_path="./datasets/BUSBRA/Images/"
        # img_path = os.path.join(rec['dir'], rec['ID'])
        img_path = dir_path + rec['ID']+".png"
        image = Image.open(img_path).convert('L')#.resize((512, 512))
        if self.transform:
            image = self.transform(image)  # transform expects [H, W]

        arr = np.array(image, dtype=np.float32) / 255.0
        image = jt.array(arr).unsqueeze(0)  # shape: [1, H, W]

        if self.augment:
            if np.random.rand() < 0.5:
                image = jt.flip(image, dim=[2])
            if np.random.rand() < 0.5:
                image = jt.flip(image, dim=[1])

        if self.infer:
            label = rec['ID']              # 推理：返回文件名
        else:
            label = self.label2idx[rec['BIRADS']]  # 训练/验证：返回整数标签

        return image, label



# ============== Model ==============

# ============== Training ==============
def training(model:nn.Module, optimizer:nn.Optimizer, train_loader:Dataset, now_epoch:int, num_epochs:int):
    model.train()
    losses = []
    colors = [31, 32, 33, 34, 35, 36]
    color = random.choice(colors)
    # bar_format = f"\033[{color}m{{l_bar}}{{bar}}{{r_bar}}\033[0m"
    # pbar = tqdm(train_loader, total=len(train_loader), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]" + " " * (80 - 10 - 10 - 10 - 10 - 3))

    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Training Epoch {now_epoch + 1}",
                bar_format=f"\033[{color}m{{l_bar}}{{bar}}{{r_bar}}\033[0m")


    step = 0
    for data in pbar:
        step += 1
        image, label = data
        # pred = model(image)
        # loss = nn.cross_entropy_loss(pred, label)
        # loss = compute_loss(pred, label, loss_weight)
        # loss.sync()

        # _,scores, score_mat = model(image)
        # loss = compute_loss(scores, score_mat, label)
        # loss.sync()

        fused_feat, scores, score_mat, rich_feat, mid_feat, mask = model(image)
        all_loss = compute_total_loss(model,scores, score_mat, label,
                                  rich_feat, fused_feat,
                                  domain_logits=None, domain_labels=None,
                                  mask=mask, gcn_feat=model._d_feats["d3"])

        loss = all_loss[0]

        loss.sync()



        # outs = model(image,aug=True)
        # loss = compute_loss(outs, label)
        # loss.sync()

        optimizer.step(loss)
        losses.append(loss.item())
        pbar.set_postfix(loss=loss.item())

    print(f'Epoch {now_epoch+1} / {num_epochs} [TRAIN] mean loss = {np.mean(losses):.2f}')

def evaluate(model:nn.Module, val_loader:Dataset):
    model.eval()
    preds, targets = [], []
    print("Evaluating...")

    colors = [31, 32, 33, 34, 35, 36]
    color = random.choice(colors)
    pbar = tqdm(val_loader, total=len(val_loader), desc="Evaluating",
                bar_format=f"\033[{color}m{{l_bar}}{{bar}}{{r_bar}}\033[0m")

    for data in pbar:
        image, label = data
        # pred = model(image)
        # _,pred, _ = model(image)
        outs = model(image)
        pred=outs[1]
        # pred, _ = model(image)
        # pred.sync()
        pred = pred.numpy().argmax(axis=1)
        preds.append(pred)
        targets.append(label.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    acc = np.mean(np.float32(preds == targets))
    return acc

def run(model:nn.Module, optimizer:nn.Optimizer, train_loader:Dataset, val_loader:Dataset, num_epochs:int, modelroot:str,ckpt:str):
    best_acc = 0
    for epoch in range(num_epochs):
        training(model, optimizer, train_loader, epoch, num_epochs)
        acc = evaluate(model, val_loader)
        if acc > best_acc:
            best_acc = acc
            model.save(os.path.join(modelroot, ckpt))
        print(f'Epoch {epoch+1} / {num_epochs} [VAL] best_acc = {best_acc:.2f}, acc = {acc:.2f}')


# ============== Test ==================

def test(model:nn.Module, test_loader:Dataset, result_path:str):
    model.eval()
    preds = []
    names = []
    print("Testing...")
    colors = [31, 32, 33, 34, 35, 36]
    color = random.choice(colors)
    pbar = tqdm(test_loader, total=len(test_loader),
                bar_format=f"\033[{color}m{{l_bar}}{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}]\033[0m" + " " * (
                        80 - 10 - 10 - 10 - 10 - 3))

    for data in pbar:
        image, image_names = data
        # pred = model(image)
        _,pred,_ = model(image)
        # pred, _ = model(image)
        pred.sync()
        pred = pred.numpy().argmax(axis=1)
        preds.append(pred)
        names.extend(image_names)
    preds = np.concatenate(preds)
    with open(result_path, 'w') as f:
        for name, pred in zip(names, preds):
            f.write(name + ' ' + str(pred) + '\n')

# ============== Main ==============
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='./datasets/BUSBRA/bus_data.csv')
    parser.add_argument('--testonly', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument("--ckpt", type=str, default="pertrain_checkpoint2.pkl", help="save checkpoint path")
    parser.add_argument('--modelroot', type=str, default='./model_save')
    parser.add_argument('--loadfrom', type=str, default='./model_save/best.pkl')
    parser.add_argument('--result_path', type=str, default='./result.txt')
    args = parser.parse_args()

    # ============ 数据预处理 ============
    transform_train = Compose([
        Resize((512, 512)),
        RandomCrop(448),
        RandomHorizontalFlip(),
        # ToTensor(),
        # ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = Compose([
        Resize((512, 512)),
        CenterCrop(448),
        # ToTensor(),
        # ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ============ 构建数据集 / 计算类别数 ============
    if not args.testonly:

        train_set = BreastCSVSet(args.csv_path, mode='train', transform=transform_train, augment=True,
                                 batch_size=4, num_workers=16, shuffle=True)
        val_set = BreastCSVSet(args.csv_path, mode='val', transform=transform_val, augment=False,
                               batch_size=4, num_workers=16, shuffle=False)

        num_classes = len(train_set.label2idx)
    else:
        test_set =BreastCSVSet(args.csv_path, mode='test', transform=transform_val, augment=False,
                               batch_size=4, num_workers=6, shuffle=False)
        # 推理时类别数可随意，但得给模型一个值
        num_classes = 4   # 若已知 0–6 共 7 类，可硬编码；否则亦可 load 时替换

    # ============ 初始化模型 ============
    # model = ResNet101(num_classes=num_classes, pretrain=True,in_channels=4)

    # model = DualEncoderCrossAttNet(num_classes=num_classes,pretrained=True)
    # model = ResNet101MaskAtt(num_classes=num_classes,pretrained=True)
    # model = ResNet101WithMaskAttention(num_classes=num_classes)

    model = Res101_Mamba_ITE_UNet_GraphPlus(num_classes=num_classes,in_channels=4,freeze_seg=True, seg_ckpt="ckpts/best_model.pkl")


    if not args.testonly:
        optimizer = nn.Adam(model.parameters(), lr=1e-5)
        run(model, optimizer, train_set, val_set, args.epochs, args.modelroot,args.ckpt)
    else:
        model.load(args.loadfrom)
        test(model, test_set, args.result_path)
