#!/bin/bash


echo "🚀 开始推理流程..."

# 定义路径（可根据实际路径修改）
DATAROOT="./TestSetA"
TESTONLY=True
MODELROOT="./model_save"
LOADFROM="./model_save/checkpoint2.pkl"
RESULTPATH="./result.txt"




# 执行推理
python ckpt2_main.py \
    --dataroot ${DATAROOT} \
    --testonly ${TESTONLY} \
    --modelroot ${MODELROOT} \
    --loadfrom ${LOADFROM} \
    --resultpath ${RESULTPATH} \


echo "✅ 推理完成，输出已保存至 ${RESULTPATH}"
