#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash msrun_single_bert_emotect_finetune.sh"
echo "==========================================="

EXEC_PATH=$(pwd)
if [ ! -d "${EXEC_PATH}/data" ]; then
    if [ ! -f "${EXEC_PATH}/emotion_detection.tar.gz" ]; then
        wget wget https://baidu-nlp.bj.bcebos.com/emotion_detection-dataset-1.0.0.tar.gz -O emotion_detection.tar.gz
    fi
    tar xvf emotion_detection.tar.gz
fi
export DATA_PATH=${EXEC_PATH}/data/


rm -rf msrun_log_msrun_single_bert_emotect_finetune
mkdir msrun_log_msrun_single_bert_emotect_finetune
echo "start training"
export LD_PRELOAD=$LD_PRELOAD:/home/tridu33/.conda/envs/openmind-pt/lib/python3.9/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0:/home/tridu33/.conda/envs/openmind-pt/lib/python3.9/site-packages/torch.libs/libgomp-6e1a1d1b.so.1.0.0
# unset LD_PRELOAD

# 设置设备数量export ASCEND_RT_VISIBLE_DEVICES=0或者指定1表示CPU，不使用分布式
msrun --worker_num=1 --local_worker_num=1 --master_port=8118 --log_dir=msrun_log_msrun_single_bert_emotect_finetune --join=True --cluster_time_out=30 npu_bert_emotect_finetune.py

# export ASCEND_RT_VISIBLE_DEVICES=7
# msrun --worker_num=7 --local_worker_num=7 --master_port=8120 --log_dir=msrun_log_msrun_single_bert_emotect_finetune --join=True --cluster_time_out=300 npu_bert_emotect_finetune.py
