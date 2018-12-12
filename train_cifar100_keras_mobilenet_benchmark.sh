#!/bin/bash
#PBS -l select=1:ncpus=272 -lplace=excl

source activate tensorflow_hvd

export OMP_NUM_THREADS=136;
export KMP_AFFINITY=granularity=fine,compact,1,0;

cd ~/federated/robust_fedavg
time numactl -p 1 python train_cifar100_keras_mobilenet_benchmark.py --nsplit 40 --batchsize 256 --epochs 800 --dir /homes/cx2/datasets/cifar100 2>&1 | tee ~/federated/results/train_cifar100_keras_mobilenet_benchmark.log