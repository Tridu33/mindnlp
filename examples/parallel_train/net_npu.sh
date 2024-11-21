#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash msrun_single.sh"
echo "==========================================="

# export MY_MODELS_PATH=/home/usersshared/LLMmodels # ~/.bashrc

if [ ! -d "${MY_MODELS_PATH}/MNIST_Data" ]; then
    if [ ! -f "${MY_MODELS_PATH}/compressed/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${MY_MODELS_PATH}/MNIST_Data/train/

rm -rf msrun_log_MIINIST
mkdir msrun_log_MIINIST
echo "start training"

export LD_PRELOAD=$LD_PRELOAD:/home/tridu33/.conda/envs/openmind-pt/lib/python3.9/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0:/home/tridu33/.conda/envs/openmind-pt/lib/python3.9/site-packages/torch.libs/libgomp-6e1a1d1b.so.1.0.0
unset LD_PRELOAD

msrun --worker_num=4 --local_worker_num=4 --master_port=8119 --log_dir=msrun_log_MIINIST --join=True --cluster_time_out=300 net_npu.py
