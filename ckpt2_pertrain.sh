#!/bin/bash


echo "🚀 开始预训练流程..."

# 定义路径（可根据实际路径修改）
CSVPATH="./datasets/BUSBRA/bus_data.csv"
TESTONLY=False
MODELROOT="./model_save"
CKPT="pertrain_checkpoint2.pkl"
EPOCHS=50



# 执行推理
python ckpt2_main_pertrain.py \
    --csv_path ${CSVPATH} \
    --testonly ${TESTONLY} \
    --modelroot ${MODELROOT} \
    --ckpt ${CKPT} \
    --epochs ${EPOCHS}




echo "✅ 训练完成，输出已保存至 ${MODELROOT}/${CKPT}"
