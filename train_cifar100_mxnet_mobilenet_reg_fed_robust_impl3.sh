#!/bin/bash
#PBS -l select=5:ncpus=272 -lplace=excl

source /opt/intel/compilers_and_libraries_2017/linux/mpi/bin64/mpivars.sh
source activate mxnet_latest


### OPA FABRIC ###
export I_MPI_FABRICS=ofi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=34

### OPA FABRIC ###
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

export KMP_AFFINITY=granularity=fine,compact,1,0;

cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile

# cat $PBS_O_WORKDIR/hostfile

# logfile=/homes/cx2/federated/results/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.txt
logfile=/homes/cx2/federated/results/train_cifar100_mxnet_cifarresnet_reg_fed_robust_impl3.txt

> $logfile

# mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.py --classes 10 --nsplit 100 --batchsize 50 --lr 0.8 --lr-decay 0.5 --lr-decay-epoch 400 --regularization 0.01 --reg-inc 0.01 --reg-inc-epoch 200 --epochs 500 --iterations 5 --dir /homes/cx2/datasets/cifar10 -o $logfile 2>&1 | tee /homes/cx2/federated/results/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.log
mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl3.py --classes 10 --model cifar_resnet20_v1 --nsplit 100 --batchsize 50 --lr 0.1 --lr-decay 0.5 --lr-decay-epoch 200 --reg-factor 0.1 --reg-epsilon 400 --epochs 800 --iterations 5 --dir /homes/cx2/datasets/cifar10 -o $logfile 2>&1 | tee /homes/cx2/federated/results/train_cifar100_mxnet_cifarresnet_reg_fed_robust_impl3.log