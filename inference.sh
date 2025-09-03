#!/bin/bash


echo "🚀 开始推理流程..."

# 定义路径（可根据实际路径修改）
DATAROOT="./TestSetA"
TESTONLY=True
MODELROOT="./model_save"
SEG_CKPT="ckpts/emcadnet_best_modelv4.pkl"
LOADFROM="./model_save/checkpoint1.pkl"
RESULTPATH="./result.txt"




# 执行推理
python main.py \
    --dataroot ${DATAROOT} \
    --testonly ${TESTONLY} \
    --modelroot ${MODELROOT} \
    --seg_ckpt ${SEG_CKPT} \
    --loadfrom ${LOADFROM} \
    --resultpath ${RESULTPATH} \


echo "✅ 推理完成，输出已保存至 ${RESULTPATH}"
