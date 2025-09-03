**第五届计图大赛赛道一 B 榜提交说明文档**：

---

### 📄 `contest1_学点儿东西_b榜第15名


# 第五届计图比赛 - 赛道一 B 榜提交说明文档

## 1. 团队信息
- 团队名称：学点儿东西
- B榜排名：第15名
- 联系人姓名：梁磊
- 微信号：leiGeGer_Sir
- 联系电话：17735469179

## 2.提交说明
代码同A榜提交代码，最优结果 A榜checkpoint1.pkl

## 3.开源链接

## 2. 项目概述

本项目专注于乳腺癌超声图像的多类别病灶识别，提出了一种多分支融合模型 `Res101_Mamba_ITE_UNet_GraphPlus`，
结合了局部/全局特征建模、结构增强与分割先验，显著提升了跨结构的视觉泛化能力和判别能力。

核心模块包括：

- **ResNet101 主干网络**：提取图像深层特征；
- **Mamba 分支**：捕捉长程依赖，补足 CNN 表达局限；
- **ITE 分支**：建模局部结构间相互关系；
- **UNet 分割辅助分支**：提供组织边界先验；
- **GCN 分支**：引入图结构辅助判别；
- **方向注意力（Directional Attention）**：建模方向一致性；
- **结构量化与引导机制**：增强结构信息注入；
- **ScoreSoftmax**：增强评分层稳定性；
- **GRL + Prototype 机制**：引入跨域泛化能力。

模型基于 Jittor 框架开发，训练与推理过程可完全复现，最终在 A 榜取得优异成绩。



## 3. 代码结构说明

代码位于 `/workspace/code`，结构如下：

```
├── ckps/                 # 保存已经训练好的权重文件，将权重文件复制到相应文件夹后，可以直接进行推理执行
├── ckpts/                # 图像分割模型权重目录
├── datasets/             # 数据集目录（保存额外的数据集BUSBRA、BUSUCLM、BUSI、MERGE_DATA）
├── Dockerfile            # Docker封装文件
├── EMCADNet.py           # 分割模块EMCADNet结构定义
├── emcad_decoder.py      # EMCADNet解码器模块
├── inference.sh          # 推理执行脚本
├── main.py               # 主入口（集成训练 / 测试逻辑）
├── main_pertrain.py      # 预训练入口（适用于某些ckpt）
├── model_save/           # 训练中间结果输出目录以及checkpoint存放位置
├── models_utils.py       # 模型工具函数模块
├── nnUnet_jittor.py      # nnUNet结构定义（不可修改）
├── nnUnet_main.py        # nnUNet相关训练入口
├── requirements.txt      # Python依赖说明
├── result.txt            # 推理结果输出示例
├── TestSetA/             # 测试集（文件夹）
├── TrainSet/             # 训练集（文件夹）
├── *.sh / *.py           # 各版本训练/推理脚本


````


## 4. 环境配置（基于 Docker 镜像）

- **基础镜像**：`jittor/jittor:cuda12.2-cudnn8`
- **Python 版本**：3.9.21
- **依赖说明**：
  - jittor==1.3.9.14
  - numpy, scikit-learn, opencv-python, matplotlib, tqdm, pandas, pyyaml 等已在 Dockerfile 中安装

### Docker 镜像构建命令

```bash
docker build -t contest1_team_37 .
```

```
### 镜像验证命令

```bash
docker run --gpus all -it contest1_team_37 bash
```


## 4. 环境配置（基于 anaconda 虚拟环境）


- **Python 版本**：3.9.21
- **依赖说明**：
  - jittor==1.3.9.14
  - numpy, scikit-learn, opencv-python, matplotlib, tqdm, pandas, pyyaml 等已在 详见 requirements.txt

### anaconda 虚拟环境 构建命令

```bash
conda create -n jittor_contest python==3.9.21
conda activate jittor_contest
```


### 安装相关库文件

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```





---

## 5. 训练与推理流程

### 🏋️‍♀️ 最后阶段训练命令：

```bash
bash train.sh
```

* 模型输出保存于 `/workspace/model_save/`
* sh无法执行时，可以使用 `train.sh` 内参数，手动执行对应的py文件
* 使用训练脚本：`main.py`


### 🏋️‍♀️ 完整训练命令：

```bash
bash emcad_main.sh
bash nnUnet_main.sh
bash pertrain.sh
bash train.sh
```

* `emcad_main.sh` 与 `nnUnet_main.sh` 执行emcad与nnUnet的图像分割训练代码
* `pertrain.sh` 执行在额外数据集上的预训练操作
* `train.sh`使用训练脚本：`main.py` 执行训练任务


### 🔍 推理命令（评估 A榜）：

```bash
bash inference.sh
```

* 推理数据路径：默认 `/workspace/TestSetA`
* 预测结果保存在 `/workspace/result.txt`
* 推理脚本：`main.py`

---

## 6. Checkpoint 说明

| 文件名                         | 模型结构版本              | 备注         |
| --------------------------- | ------------------- | ---------- |
| `checkpoint1.pkl` | 多分支融合，含分割/方向注意/结构引导 | 最终用于 A 榜提交 |
| `pertrain_checkpoint1.pkl`   | 在额外数据集上的模型预训练   | 模型预训练      |
| `emcadnet_best_modelv4.pkl`      | EMCADNet图片分割网络训练结果       | 分割mask用于训练     |
| `best_model.pkl`      | nnUet图片分割网络训练结果       | 分割mask用于训练     |

所有模型默认保存在 `/workspace/[ckpts/][model_save/]`。

---

## 7. 注意事项与复现说明

* 运行推理前请确认 GPU 可用 (`nvidia-smi`)；
* 若模型路径或数据路径需修改，请更新  shell 脚本；
* 默认使用 `jt.flags.device_id = 0` 运行于 GPU 模式；
* ckpt2_main.py、ckpt2_main_pertrain.py、ckpt2_models_utils.py 是checkpoint2.pkl的代码逻辑,
  执行策略同 main.py 、main_pertrain.py、model_utils.py
* ckpt3_main.py、ckpt3_main_pertrain.py、ckpt3_models_utils.py 是checkpoint3.pkl的代码逻辑,
  执行策略同 main.py 、main_pertrain.py、model_utils.py

---

## 8. 附录：Docker 镜像导出/导入命令

```bash
# 镜像导出
docker save contest1_team_37 > contest1_team_37.tar

# 镜像导入
docker load < contest1_team_37.tar
```

