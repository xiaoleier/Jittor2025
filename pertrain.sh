#!/bin/bash


echo "🚀 开始预训练流程..."

# 定义路径（可根据实际路径修改）
CSVPATH="./datasets/BUSBRA/bus_data.csv"
TESTONLY=False
MODELROOT="./model_save"
SEG_CKPT="ckpts/emcadnet_best_modelv4.pkl"
CKPT="pertrain_checkpoint1.pkl"
EPOCHS=100




# 执行推理
python main_pertrain.py \
    --csv_path ${CSVPATH} \
    --testonly ${TESTONLY} \
    --modelroot ${MODELROOT} \
    --seg_ckpt ${SEG_CKPT} \
    --ckpt ${CKPT} \
    --epochs ${EPOCHS}




echo "✅ 训练完成，输出已保存至 ${MODELROOT}/${CKPT}"
