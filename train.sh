#!/bin/bash


echo "🚀 开始训练流程..."

# 定义路径（可根据实际路径修改）
DATAROOT="./TrainSet"
TESTONLY=False
MODELROOT="./model_save"
SEG_CKPT="ckpts/emcadnet_best_modelv4.pkl"
CKPT="checkpoint1.pkl"
LOADPERTAIN="./model_save/pertrain_checkpoint1.pkl"
EPOCHS=210




# 执行推理
python main.py \
    --dataroot ${DATAROOT} \
    --testonly ${TESTONLY} \
    --modelroot ${MODELROOT} \
    --seg_ckpt ${SEG_CKPT} \
    --ckpt ${CKPT} \
    --loadpertain ${LOADPERTAIN} \
    --epochs ${EPOCHS}




echo "✅ 训练完成，输出已保存至 ${MODELROOT}/${CKPT}"
