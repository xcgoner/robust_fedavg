#!/bin/bash
#PBS -l select=1:ncpus=272 -lplace=excl

source activate tensorflow_intel

export OMP_NUM_THREADS=88;
export KMP_AFFINITY=granularity=fine,compact,1,0;

cd ~/federated/robust_fedavg
time numactl -p 1 python train_imagenet_keras_benchmark.py --dir /export/shared/uiuc/imagenet/ILSVRC2012/datasets --batchsize 32 2>&1 | tee ~/federated/results/train_imagenet_keras_benchmark.log