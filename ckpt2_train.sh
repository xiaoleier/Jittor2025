#!/bin/bash


echo "🚀 开始训练流程..."

# 定义路径（可根据实际路径修改）
DATAROOT="./TrainSet"
TESTONLY=False
MODELROOT="./model_save"
CKPT="checkpoint2.pkl"
LOADPERTAIN="./model_save/pertrain_checkpoint2.pkl"
EPOCHS=100




# 执行推理
python ckpt2_main.py \
    --dataroot ${DATAROOT} \
    --testonly ${TESTONLY} \
    --modelroot ${MODELROOT} \
    --ckpt ${CKPT} \
    --loadpertain ${LOADPERTAIN} \
    --epochs ${EPOCHS}




echo "✅ 训练完成，输出已保存至 ${MODELROOT}/${CKPT}"
