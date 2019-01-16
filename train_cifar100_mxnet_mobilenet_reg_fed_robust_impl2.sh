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
# logfile=/homes/cx2/federated/results/train_cifar100_mxnet_cifarresnet_reg_fed_robust_impl2.txt
logfile=/homes/cx2/federated/results/train_cifar100_mxnet_default_reg_fed_robust_impl2.txt

> $logfile

# mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.py --classes 10 --nsplit 100 --batchsize 50 --lr 0.8 --lr-decay 0.5 --lr-decay-epoch 400 --regularization 0.01 --reg-inc 0.01 --reg-inc-epoch 200 --epochs 500 --iterations 5 --dir /homes/cx2/datasets/cifar10 -o $logfile 2>&1 | tee /homes/cx2/federated/results/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.log
# mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.py --classes 10 --model cifar_resnet20_v1 --nsplit 100 --batchsize 50 --lr 0.1 --lr-decay 1 --lr-decay-epoch 150 --regularization 0.9 --reg-inc 0.0001 --reg-inc-epoch 150 --epochs 800 --iterations 1 --trim 1 --start-epoch 140 --dir /homes/cx2/datasets/cifar10 -o $logfile 2>&1 | tee /homes/cx2/federated/results/train_cifar100_mxnet_cifarresnet_reg_fed_robust_impl2.log
# mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.py --classes 10 --model default --nsplit 100 --batchsize 50 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 800 --regularization 0.5 --reg-inc 0.0001 --reg-inc-epoch 800 --epochs 800 --iterations 4 --dir /homes/cx2/datasets/cifar10 -o $logfile 2>&1 | tee /homes/cx2/federated/results/train_cifar100_mxnet_default_reg_fed_robust_impl2.log
# mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.py --classes 10 --model cifar_resnet20_v1 --nsplit 100 --batchsize 50 --lr 0.4 --lr-decay 0.1 --lr-decay-epoch 800 --regularization 0.8 --reg-inc 0.05 --reg-inc-epoch 200,250,300,350 --epochs 800 --iterations 2 --start-epoch 190 --dir /homes/cx2/datasets/cifar10 -o $logfile 2>&1 | tee /homes/cx2/federated/results/train_cifar100_mxnet_cifarresnet_reg_fed_robust_impl2.log
# mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.py --classes 10 --model cifar_resnet20_v1 --nsplit 100 --batchsize 50 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 800 --regularization 0.8 --reg-inc 0.1 --reg-inc-epoch 200,300 --epochs 400 --iterations 2 --start-epoch 190 --dir /homes/cx2/datasets/cifar10_normalized -o $logfile 2>&1 | tee /homes/cx2/federated/results/train_cifar100_mxnet_cifarresnet_reg_fed_robust_impl2.log
# mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.py --classes 10 --model cifar_resnet20_v1 --nsplit 100 --batchsize 50 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 800 --regularization 0.8 --reg-inc 0.1 --reg-inc-epoch 800 --epochs 400 --iterations 2 --dir /homes/cx2/datasets/cifar10_normalized -o $logfile 2>&1 | tee /homes/cx2/federated/results/train_cifar100_mxnet_cifarresnet_reg_fed_robust_impl2.log
# mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.py --classes 10 --model default --nsplit 100 --batchsize 50 --lr 0.1 --lr-decay 0.8 --lr-decay-epoch 350 --regularization 0.2 --reg-inc 0.1 --reg-inc-epoch 6000 --epochs 1200 --iterations 2 --start-epoch 340 --dir /homes/cx2/datasets/cifar10_normalized -o $logfile 2>&1 | tee /homes/cx2/federated/results/train_cifar100_mxnet_default_reg_fed_robust_impl2.log
# mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.py --classes 10 --model default --nsplit 100 --batchsize 50 --lr 0.15 --lr-decay 0.1 --lr-decay-epoch 350 --regularization 0.2 --reg-inc 0.1 --reg-inc-epoch 6000 --epochs 800 --iterations 1 --start-epoch 340 --dir /homes/cx2/datasets/cifar10_normalized -o $logfile 2>&1 | tee /homes/cx2/federated/results/train_cifar100_mxnet_default_reg_fed_robust_impl2.log
# mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.py --classes 10 --model default --nsplit 100 --batchsize 50 --lr 0.1 --lr-decay 0.4 --lr-decay-epoch 400 --regularization 0.2 --reg-inc 0.1 --reg-inc-epoch 800 --epochs 800 --iterations 1 --dir /homes/cx2/datasets/cifar10_normalized_unbalanced -o $logfile 2>&1 | tee /homes/cx2/federated/results/train_cifar100_mxnet_default_reg_fed_robust_impl2.log
mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.py --classes 10 --model default --nsplit 100 --batchsize 50 --lr 0.15 --lr-decay 0.4 --lr-decay-epoch 400 --regularization 0.12 --reg-inc 0.1 --reg-inc-epoch 0 --epochs 800 --iterations 1 --dir /homes/cx2/datasets/cifar10_normalized_unbalanced -o $logfile 2>&1 | tee /homes/cx2/federated/results/train_cifar100_mxnet_default_reg_fed_robust_impl2.log
