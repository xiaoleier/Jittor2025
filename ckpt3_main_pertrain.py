import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["JT_SYNC"] = "1"
# os.environ["trace_py_var"] = "3"


import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset
from jittor import transform
from jittor.transform import Compose, Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, ToTensor, ImageNormalize

from tqdm import tqdm

import numpy as np
from PIL import Image
import argparse
import random
import pandas as pd
from sklearn.model_selection import train_test_split



from ckpt3_models_utils import compute_loss,Res101_Mamba_ITE_UNet_GraphPlus,compute_total_loss



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
        dir_path="../BUSBRA/Images/"
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




class MERGEDataset(Dataset):
    def __init__(self, root_dir, image_size=224, mode='train', transform=None, augment=False,split_ratio=0.8, seed=42, oversample=True):
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size
        self.mode = mode  # 'train' or 'val'
        self.classes = ['benign', 'malign', 'normal']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.transform = transform
        self.augment = augment

        # 收集所有数据
        self.data = []
        self.label_count = {}

        # 收集每个类别样本路径
        cls_samples = {cls: [] for cls in self.classes}
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cls_dir, fname)
                    label = self.class_to_idx[cls]
                    cls_samples[cls].append((img_path, label))


        # 固定随机种子，划分 9:1
        random.seed(seed)
        for cls in self.classes:
            samples = cls_samples[cls]
            random.shuffle(samples)
            split_idx = int(len(samples) * split_ratio)
            if self.mode == 'train':
                cls_train = samples[:split_idx]
                if oversample:
                    max_class_len = max(
                        len(cls_samples[c][:int(len(cls_samples[c]) * split_ratio)]) for c in self.classes)
                    repeat_times = max_class_len // len(cls_train)
                    remainder = max_class_len % len(cls_train)
                    cls_train = cls_train * repeat_times + cls_train[:remainder]
                self.data.extend(cls_train)
            else:
                cls_val = samples[split_idx:]
                self.data.extend(cls_val)

        random.shuffle(self.data)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        image = Image.open(img_path).convert("L")  # 改成 "RGB" 如果是彩色图像
        if self.transform:
            image = self.transform(image)  # transform expects [H, W]

        arr = np.array(image, dtype=np.float32) / 255.0
        image = jt.array(arr).unsqueeze(0)  # shape: [1, H, W]

        if self.augment:
            if np.random.rand() < 0.5:
                image = jt.flip(image, dim=[2])
            if np.random.rand() < 0.5:
                image = jt.flip(image, dim=[1])
        # image = self.transform(image)
        return image, label


# ============== Model ==============

# ============== Training ==============
def training(model:nn.Module, optimizer:nn.Optimizer, train_loader:Dataset, now_epoch:int, num_epochs:int):
    model.train()
    losses = []

    # 初始化各项损失列表
    loss_cls_list = []
    loss_score_list = []
    loss_struct_align_list = []
    loss_struct_center_list = []
    loss_uncertainty_list = []

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

        outs = model(image)

        loss, loss_cls, loss_score, loss_struct_align, loss_struct_center, loss_uncertainty = compute_total_loss(
            model, outs['scores'], outs['score_mat'], label,
            outs['rich_struct_feat'], outs['fused_feat'], gcn_feat=outs['gcn_feat'],
            num_classes=3)

        loss.sync()

        optimizer.step(loss)
        losses.append(loss.item())

        # 记录各项损失
        loss_cls_list.append(loss_cls)
        loss_score_list.append(loss_score)
        loss_struct_align_list.append(loss_struct_align)
        loss_struct_center_list.append(loss_struct_center)
        loss_uncertainty_list.append(loss_uncertainty)

        pbar.set_postfix(loss=loss.item())

    # 计算各项损失的均值
    mean_loss_cls = np.mean(loss_cls_list)
    mean_loss_score = np.mean(loss_score_list)
    mean_loss_struct_align = np.mean(loss_struct_align_list)
    mean_loss_struct_center = np.mean(loss_struct_center_list)
    mean_loss_uncertainty = np.mean(loss_uncertainty_list)

    print(f'Epoch {now_epoch} / {num_epochs} [TRAIN] mean loss = {np.mean(losses):.2f}')
    print(f'Epoch {now_epoch} / {num_epochs} [TRAIN] mean loss_cls = {mean_loss_cls:.2f}')
    print(f'Epoch {now_epoch} / {num_epochs} [TRAIN] mean loss_score = {mean_loss_score:.2f}')
    print(f'Epoch {now_epoch} / {num_epochs} [TRAIN] mean loss_struct_align = {mean_loss_struct_align:.2f}')
    print(f'Epoch {now_epoch} / {num_epochs} [TRAIN] mean loss_struct_center = {mean_loss_struct_center:.2f}')
    print(f'Epoch {now_epoch} / {num_epochs} [TRAIN] mean loss_uncertainty = {mean_loss_uncertainty:.2f}')



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
        pred = outs['scores']
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
        print(f'Epoch {epoch} / {num_epochs} [VAL] best_acc = {best_acc:.2f}, acc = {acc:.2f}')


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
        outs = model(image)
        # pred, _ = model(image)
        pred = outs[1]
        pred = pred.numpy().argmax(axis=1)
        preds.append(pred)
        names.extend(image_names)
    preds = np.concatenate(preds)
    with open(result_path, 'w') as f:
        for name, pred in zip(names, preds):
            f.write(name + ' ' + str(pred) + '\n')




def load_weights_flexible(model, ckpt_path):
    if not os.path.isfile(ckpt_path):
        print(f"[WARN] weight file not found: {ckpt_path}")
        return
    state = jt.load(ckpt_path)
    new_state = {}
    for k, v in state.items():
        if k not in model.state_dict():      # 新模型里没有
            continue
        if list(v.shape) != list(model.state_dict()[k].shape):
            print(f"[SKIP] {k}  old:{list(v.shape)}  new:{list(model.state_dict()[k].shape)}")
            continue
        new_state[k] = v
    model.load_parameters(new_state)
    print(f"[INFO] loaded {len(new_state)}/{len(state)} layers from {ckpt_path}")







# ============== Main ==============
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--csv_path', type=str, default='../BUSBRA/bus_data.csv')
    # parser.add_argument('--dataroot', type=str, default='./BUSUCLM/bus_uclm_separated')
    parser.add_argument('--dataroot', type=str, default='./datasets/MERGE_DATA')
    parser.add_argument('--testonly', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=230)
    parser.add_argument('--modelroot', type=str, default='./model_save')
    parser.add_argument("--ckpt", type=str, default="pertrain_checkpoint3.pkl", help="save checkpoint path")
    parser.add_argument('--loadfrom', type=str, default='./model_save/best_bus_addmask_emcad_FUS_0718.pkl')
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

        # train_set = BreastCSVSet(args.csv_path, mode='train', transform=transform_train, augment=True,
        #                          batch_size=4, num_workers=16, shuffle=True)
        # val_set = BreastCSVSet(args.csv_path, mode='val', transform=transform_val, augment=False,
        #                        batch_size=4, num_workers=16, shuffle=False)

        train_dataset = MERGEDataset(root_dir=args.dataroot, mode='train',transform=transform_train, augment=True,oversample=True)
        train_set = train_dataset.set_attrs(batch_size=4, shuffle=True)

        # 验证集
        val_dataset = MERGEDataset(root_dir=args.dataroot, mode='val', transform=transform_val, augment=False,)
        val_set = val_dataset.set_attrs(batch_size=3, shuffle=False)


        # num_classes = len(train_set.label2idx)
    else:
        test_set =MERGEDataset(root_dir=args.dataroot, mode='val', transform=transform_val, augment=False,)
        # 推理时类别数可随意，但得给模型一个值
        num_classes = 3   # 若已知 0–6 共 7 类，可硬编码；否则亦可 load 时替换

    # ============ 初始化模型 ============
    # model = ResNet101(num_classes=num_classes, pretrain=True,in_channels=4)

    # model = DualEncoderCrossAttNet(num_classes=num_classes,pretrained=True)
    # model = ResNet101MaskAtt(num_classes=num_classes,pretrained=True)
    # model = ResNet101WithMaskAttention(num_classes=num_classes)

    model = Res101_Mamba_ITE_UNet_GraphPlus(num_classes=3,in_channels=3,freeze_seg=True, seg_ckpt="ckpts/emcadnet_best_modelv4.pkl")


    if not args.testonly:

        # load_weights_flexible(model, args.loadfrom)

        optimizer = nn.Adam(model.parameters(), lr=1e-5)
        run(model, optimizer, train_set, val_set, args.epochs, args.modelroot,args.ckpt)
    else:
        model.load(args.loadfrom)
        test(model, test_set, args.result_path)
