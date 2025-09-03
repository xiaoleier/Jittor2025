import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset
from jittor import transform
from jittor.transform import Compose, Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, ToTensor, ImageNormalize
from jittor.models import Resnet50, Resnet101, Resnet152, Resnet18, Resnet34
from tqdm import tqdm

import numpy as np
from PIL import Image
import argparse
import random
from pathlib import Path

# from model_utils import ResNeXt50MultiScaleCBAM
# from score_softmax_net import compute_loss,Net
# from convnext_score_softmax_net import compute_loss,ConvNeXtTinyScoreNet
# from convnext_acmix_multiscale_score import compute_loss,ConvNeXtTinyACmixScoreNet
# from convnext_mamba_ite_score import compute_loss,ConvNeXtTinyITE_MambaScoreNet
# from resnet101_mamba_ite_score import compute_loss,ResNet101_ITE_Mamba_Score
# from unet_score_softmax import compute_loss,UNetScoreSoftmax
# from resnet101_fusion_unet_score import compute_loss,ResNet101_Fusion_UNet_Score
# from res101_fusion_unet_gcn_vgt_score import compute_loss,Res101_Mamba_ITE_UNet_GraphPlus
# from res101_mamba_ite_unet_graphplus import compute_loss,Res101_Mamba_ITE_UNet_GraphPlus
# from res101 import ResNet101,ResNet101MaskAtt,compute_class_weight,compute_loss,ResNet101WithMaskAttention

from models_utils import compute_loss,Res101_Mamba_ITE_UNet_GraphPlus,compute_total_loss


jt.flags.use_cuda = 1
# jt.flags.device_id = 3

# ============== Dataset ==============
# class ImageFolder(Dataset):
#     def __init__(self, root, annotation_path=None, transform=None, **kwargs):
#         super().__init__(**kwargs)
#         self.root = root
#         self.transform = transform
#         if annotation_path is not None:
#             with open(annotation_path, 'r') as f:
#                 data_dir = [line.strip().split(' ') for line in f]
#             data_dir = [(x[0], int(x[1])) for x in data_dir]
#         else:
#             data_dir = sorted(os.listdir(root))
#             data_dir = [(x, None) for x in data_dir]
#         self.data_dir = data_dir
#         self.total_len = len(self.data_dir)
#
#     def __getitem__(self, idx):
#         image_path, label = os.path.join(self.root, self.data_dir[idx][0]), self.data_dir[idx][1]
#         image = Image.open(image_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         image_name = self.data_dir[idx][0]
#         label = image_name if label is None else label
#         return jt.array(image), label




# -----------------------------
# TrainSetDataset  (images/train + mask + annotation.txt)
# -----------------------------
class ImageFolder(jt.dataset.Dataset):
    def __init__(self, root, annotation_path=None, transform=None, augment=False,**kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.transform = transform
        self.augment = augment
        if annotation_path is not None:
            with open(annotation_path, 'r') as f:
                data_dir = [line.strip().split(' ') for line in f]
            data_dir = [(x[0], int(x[1])) for x in data_dir]
        else:
            data_dir = sorted(os.listdir(root))
            data_dir = [(x, None) for x in data_dir]
        self.data_dir = data_dir
        self.total_len = len(self.data_dir)

    def __getitem__(self, idx):
        image_path, label = os.path.join(self.root, self.data_dir[idx][0]), self.data_dir[idx][1]
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        arr = np.array(image, dtype=np.float32) / 255.0
        image = jt.array(arr).unsqueeze(0)  # shape: [1, H, W]

        if self.augment:
            if np.random.rand() < 0.5:
                image = jt.flip(image, dim=[2])
            if np.random.rand() < 0.5:
                image = jt.flip(image, dim=[1])

        image_name = self.data_dir[idx][0]
        label = image_name if label is None else label
        return image, label

# ============== Model ==============

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



# class Net2(nn.Module):
#     def __init__(self, num_classes, pretrain, loadfrom=None):
#         super().__init__()
#         model = Resnet101(num_classes=num_classes,pretrained=pretrain)
#         if loadfrom:
#             self.base_net = load_weights_flexible(model, loadfrom)
#         else:
#             self.base_net = model
#
#
#
#     def execute(self, x):
#         x = self.base_net(x)
#         return x

class CustomHead(nn.Module):
    """
    一个两层 MLP + Dropout 的示例。
    接收 2048-d 向量 → 512 → num_classes
    """
    def __init__(self, in_feat=2048, num_classes=6, p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(512, num_classes)
        )

    def execute(self, x):
        return self.net(x)


# class Net2(nn.Module):
#     def __init__(self, num_classes=6, pretrain=True, loadfrom=None):
#         super().__init__()
#         self.base_net = ResNet101(num_classes=6, pretrain=True,in_channels=4)  # 先随便 1000
#         if loadfrom:
#             load_weights_flexible(self.base_net, loadfrom)
#
#         # 换成新任务的输出层
#         in_feat = self.base_net.base_net.fc.weight.shape[1]
#         # self.base_net.fc = nn.Linear(in_feat, num_classes)
#         self.base_net.base_net.fc = nn.Identity()
#         # self.base_net.fc.init_parameters()
#
#         self.neck = nn.Identity()
#
#         self.head = CustomHead(in_feat, num_classes)
#
#     def extract_feat(self, x):
#         """只抽特征，不含 head，用于验证或可视化"""
#         x = self.base_net(x)  # B×2048
#         x = self.neck(x)
#         return x
#
#     def execute(self, x):
#         feat = self.extract_feat(x)
#         logits = self.head(feat)
#         return logits

class ScoreHead(nn.Module):
    """
    一个两层 MLP + Dropout 的示例。
    接收 2048-d 向量 → 512 → num_classes
    """
    def __init__(self, in_feat=2048, num_classes=6,score_level=5, p=0.5):
        super().__init__()
        self.cls    = nn.Linear(in_feat, num_classes*score_level)
        self.table = jt.arange(1, score_level + 1).float32()
        self.score_level = score_level

    def score_softmax(self, logits):
        B = logits.shape[0]
        return nn.softmax(logits.reshape(B,6, self.score_level), dim=2)

    def execute(self, x):
        logits = self.cls(x)
        score_mat = self.score_softmax(logits)
        scores = jt.matmul(score_mat, self.table)
        return scores, score_mat

class Net2(nn.Module):
    def __init__(self, num_classes=6, pretrain=True, loadfrom=None,seg_ckpt=None):
        super().__init__()
        self.base_net = Res101_Mamba_ITE_UNet_GraphPlus(num_classes=num_classes,in_channels=4,freeze_seg=True, seg_ckpt=seg_ckpt)  # 先随便 1000
        if loadfrom:
            load_weights_flexible(self.base_net, loadfrom)

        # 换成新任务的输出层
        # in_feat = self.base_net.cross_fusion.

        # in_feat = self.base_net.fc.weight.shape[1]
        # self.base_net.fc = nn.Linear(in_feat, num_classes)
        # self.base_net.cls=nn.Identity()
        # self.base_net.score_softmax()=nn.Identity()
        # self.base_net.fc = nn.Identity()
        # self.base_net.fc.init_parameters()

        self.neck = nn.Identity()

        self.head = ScoreHead(256, num_classes,score_level=10)

    def extract_feat(self, x):
        """只抽特征，不含 head，用于验证或可视化"""
        # x,_,_ = self.base_net(x)  # B×2048
        fused_feat, scores, score_mat, rich_feat, mid_feat, mask = self.base_net(x)


        x = self.neck(fused_feat)
        return x, scores, score_mat, rich_feat, mid_feat, mask

    def execute(self, x):
        feat, scores, score_mat, rich_feat, mid_feat, mask = self.extract_feat(x)
        scores, score_mat = self.head(feat)
        return feat, scores, score_mat,rich_feat, mid_feat, mask




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
    loss_domain_list = []


    colors = [31, 32, 33, 34, 35, 36]
    color = random.choice(colors)
    # bar_format = f"\033[{color}m{{l_bar}}{{bar}}{{r_bar}}\033[0m"
    # pbar = tqdm(train_loader, total=len(train_loader), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]" + " " * (80 - 10 - 10 - 10 - 10 - 3))

    pbar = tqdm(train_loader, total=len(train_loader),desc=f"Training Epoch {now_epoch+1}",
                bar_format=f"\033[{color}m{{l_bar}}{{bar}}{{r_bar}}\033[0m")

    step = 0
    for data in pbar:
        step += 1
        image, label = data
        # pred = model(image)
        # loss = nn.cross_entropy_loss(pred, label)
        # loss.sync()

        # scores, score_mat = model(image)
        # loss = compute_loss(scores, score_mat, label,num_classes=6)
        # loss.sync()

        # scores, score_mat = model(image)
        # loss = compute_loss(scores, score_mat, label,num_classes=6)
        # loss.sync()
        # outs = model(image,aug=True)
        # loss = compute_loss(outs, label)
        # loss.sync()

        fused_feat, scores, score_mat, rich_feat, mid_feat, mask = model(image)
        # loss = compute_total_loss(model.base_net, scores, score_mat, label,
        #                           rich_feat, fused_feat,
        #                           domain_logits=None, domain_labels=None,
        #                           mask=mask, gcn_feat=model.base_net._d_feats["d3"],
        #                           num_classes=6)

        loss, loss_cls, loss_score, loss_struct_align, loss_struct_center, loss_uncertainty, loss_domain = compute_total_loss(
            model.base_net, scores, score_mat, label,
            rich_feat, fused_feat,
            domain_logits=None, domain_labels=None,
            mask=mask, gcn_feat=model.base_net._d_feats["d4"],
            num_classes=6)

        loss.sync()



        optimizer.step(loss)
        losses.append(loss.item())

        # 记录各项损失
        loss_cls_list.append(loss_cls)
        loss_score_list.append(loss_score)
        loss_struct_align_list.append(loss_struct_align)
        loss_struct_center_list.append(loss_struct_center)
        loss_uncertainty_list.append(loss_uncertainty)
        loss_domain_list.append(loss_domain)

        pbar.set_postfix(loss=loss.item())

        # 计算各项损失的均值
    mean_loss_cls = np.mean(loss_cls_list)
    mean_loss_score = np.mean(loss_score_list)
    mean_loss_struct_align = np.mean(loss_struct_align_list)
    mean_loss_struct_center = np.mean(loss_struct_center_list)
    mean_loss_uncertainty = np.mean(loss_uncertainty_list)
    mean_loss_domain = np.mean(loss_domain_list)

    print(f'Epoch {now_epoch+1} / {num_epochs} [TRAIN] mean loss = {np.mean(losses):.2f}')
    print(f'Epoch {now_epoch+1} / {num_epochs} [TRAIN] mean loss_cls = {mean_loss_cls:.2f}')
    print(f'Epoch {now_epoch+1} / {num_epochs} [TRAIN] mean loss_score = {mean_loss_score:.2f}')
    print(f'Epoch {now_epoch+1} / {num_epochs} [TRAIN] mean loss_struct_align = {mean_loss_struct_align:.2f}')
    print(f'Epoch {now_epoch+1} / {num_epochs} [TRAIN] mean loss_struct_center = {mean_loss_struct_center:.2f}')
    print(f'Epoch {now_epoch+1} / {num_epochs} [TRAIN] mean loss_uncertainty = {mean_loss_uncertainty:.2f}')
    print(f'Epoch {now_epoch+1} / {num_epochs} [TRAIN] mean loss_domain = {mean_loss_domain:.2f}')


    # print(f'Epoch {now_epoch} / {num_epochs} [TRAIN] mean loss = {np.mean(losses):.2f}')

def evaluate(model:nn.Module, val_loader:Dataset):
    model.eval()
    preds, targets = [], []
    print("Evaluating...")

    colors = [31, 32, 33, 34, 35, 36]
    color = random.choice(colors)
    pbar = tqdm(val_loader, total=len(val_loader),desc="Evaluating",
                bar_format=f"\033[{color}m{{l_bar}}{{bar}}{{r_bar}}\033[0m")

    for data in pbar:
        image, label = data
        # pred = model(image)
        # pred,_ = model(image)
        outs = model(image)
        pred = outs[1]
        pred.sync()
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
        if acc >= best_acc:
            best_acc = acc
            model.save(os.path.join(modelroot, ckpt))
            print("save best model")
        print(f'Epoch {epoch+1} / {num_epochs} [VAL] best_acc = {best_acc:.2f}, acc = {acc:.2f}')


# ============== Test ==================

def test(model:nn.Module, test_loader:Dataset, result_path:str):
    model.eval()
    preds = []
    names = []
    print("Testing...")
    colors = [31, 32, 33, 34, 35, 36]
    color = random.choice(colors)
    pbar = tqdm(test_loader, total=len(test_loader), desc="Testing",
                bar_format=f"\033[{color}m{{l_bar}}{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}]\033[0m" + " " * (
                        80 - 10 - 10 - 10 - 10 - 3))

    for data in pbar:
        image, image_names = data
        # pred = model(image)
        # pred,_ = model(image)
        # pred, _ = model(image)
        outs = model(image)
        pred = outs[1]
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
    # parser.add_argument('--dataroot', type=str, default='./TrainSet')
    # parser.add_argument('--testonly', action='store_true', default=False)

    parser.add_argument('--dataroot', type=str, default='./TestSetA')
    parser.add_argument('--testonly', action='store_true', default=True)

    parser.add_argument('--epochs', type=int, default=210)
    parser.add_argument('--modelroot', type=str, default='./model_save')
    parser.add_argument('--loadpertain', type=str, default='./model_save/pertrain_checkpoint1.pkl')
    parser.add_argument("--ckpt", type=str, default="checkpoint1.pkl", help="save checkpoint path")
    parser.add_argument("--seg_ckpt", type=str, default="ckpts/emcadnet_best_modelv4.pkl")

    parser.add_argument('--loadfrom', type=str, default='./model_save/checkpoint1.pkl')
    parser.add_argument('--result_path', type=str, default='./result.txt')
    args = parser.parse_args()


    model = Net2(num_classes=6, pretrain=True,loadfrom=args.loadpertain,seg_ckpt=args.seg_ckpt)




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



    if not args.testonly:
        optimizer = nn.Adam(model.parameters(), lr=1e-5)
        load_weights_flexible(model, args.loadfrom)

        train_set = ImageFolder(
            root=os.path.join(args.dataroot, 'images/train'),
            annotation_path=os.path.join(args.dataroot, 'labels/trainval.txt'),
            transform=transform_train,
            augment=True,
            batch_size=4,
            num_workers=8,
            shuffle=True
        )
        val_set = ImageFolder(
            root=os.path.join(args.dataroot, 'images/train'),
            annotation_path=os.path.join(args.dataroot, 'labels/val.txt'),
            transform=transform_val,
            augment=False,
            batch_size=4,
            num_workers=8,
            shuffle=False
        )

        run(model, optimizer, train_set, val_set, args.epochs, args.modelroot,args.ckpt)
    else:
        test_loader = ImageFolder(
            root=args.dataroot,
            transform=transform_val,
            augment=False,
            batch_size=1,
            num_workers=1,
            shuffle=False
        )

        # model.load(args.loadfrom)
        load_weights_flexible(model, args.loadfrom)
        test(model, test_loader, args.result_path)
