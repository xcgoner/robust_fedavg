#!/bin/bash
#PBS -l select=1:ncpus=272 -lplace=excl

source activate mxnet

export OMP_NUM_THREADS=88;
export KMP_AFFINITY=granularity=fine,compact,1,0;

cd ~/federated/robust_fedavg
time numactl -p 1 python train_imagenet_benchmark.py 2>&1 | tee ~/federated/results/train_imagenet_benchmark.log