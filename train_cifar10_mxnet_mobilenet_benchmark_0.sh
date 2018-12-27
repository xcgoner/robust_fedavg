#!/bin/bash
#PBS -l select=1:ncpus=272 -lplace=excl

source /opt/intel/compilers_and_libraries_2017/linux/mpi/bin64/mpivars.sh
source activate mxnet_latest


### OPA FABRIC ###
export I_MPI_FABRICS=ofi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=136

### OPA FABRIC ###
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

export KMP_AFFINITY=granularity=fine,compact,1,0;

logfile=/homes/cx2/federated/results/train_cifar10_mxnet_mobilenet_benchmark_0.txt

> $logfile

python /homes/cx2/federated/robust_fedavg/train_cifar10_mxnet_mobilenet_benchmark_0.py --classes 10 --nsplit 100 --batchsize 50 --lr 0.1 --regularization 0 --epochs 800 --dir /homes/cx2/datasets/cifar10 -o $logfile 2>&1 | tee /homes/cx2/federated/results/train_cifar10_mxnet_mobilenet_benchmark_0.log
