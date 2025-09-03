#!/bin/bash


echo "🚀 EMCADNet 训练开始..."



# 执行推理
python EMCADNet_main.py


echo "✅ 训练完成，输出已保存至 ckpts/emcadnet_best_modelv4.pkl"
