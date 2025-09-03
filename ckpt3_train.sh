#!/bin/bash


echo "🚀 开始训练流程..."

# 定义路径（可根据实际路径修改）
DATAROOT="./TrainSet"
TESTONLY=False
MODELROOT="./model_save"
CKPT="checkpoint3.pkl"
LOADPERTAIN="./model_save/pertrain_checkpoint3.pkl"
EPOCHS=250




# 执行推理
python ckpt3_main.py \
    --dataroot ${DATAROOT} \
    --testonly ${TESTONLY} \
    --modelroot ${MODELROOT} \
    --ckpt ${CKPT} \
    --loadpertain ${LOADPERTAIN} \
    --epochs ${EPOCHS}




echo "✅ 训练完成，输出已保存至 ${MODELROOT}/${CKPT}"
