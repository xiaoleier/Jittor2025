#!/bin/bash


echo "🚀 nnUNet 训练开始..."



# 执行推理
python nnUnet_main.py


echo "✅ 训练完成，输出已保存至 ckpts/best_model.pkl"
