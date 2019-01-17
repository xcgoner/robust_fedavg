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


# cat $PBS_O_WORKDIR/hostfile

inputdir=/homes/cx2/datasets/cifar10_normalized_unbalanced

valdir=/homes/cx2/datasets/cifar10_normalized/dataset_split_10



logfile=/homes/cx2/federated/results/exp_unbalanced_lr_05_reg_000.txt

> $logfile

cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile
mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.py --classes 10 --model default --nsplit 100 --batchsize 50 --lr 0.05 --lr-decay 0.4 --lr-decay-epoch 400 --regularization 0 --reg-inc 0.1 --reg-inc-epoch 800 --epochs 800 --iterations 1 --seed 733 --dir $inputdir --valdir $valdir -o $logfile 2>&1 | tee /homes/cx2/federated/results/exp_script_4.log
cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile
mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.py --classes 10 --model default --nsplit 100 --batchsize 50 --lr 0.05 --lr-decay 0.4 --lr-decay-epoch 400 --regularization 0 --reg-inc 0.1 --reg-inc-epoch 800 --epochs 800 --iterations 1 --seed 337 --dir $inputdir --valdir $valdir -o $logfile 2>&1 | tee /homes/cx2/federated/results/exp_script_4.log
cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile
mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.py --classes 10 --model default --nsplit 100 --batchsize 50 --lr 0.05 --lr-decay 0.4 --lr-decay-epoch 400 --regularization 0 --reg-inc 0.1 --reg-inc-epoch 800 --epochs 800 --iterations 1 --seed 755 --dir $inputdir --valdir $valdir -o $logfile 2>&1 | tee /homes/cx2/federated/results/exp_script_4.log
cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile
mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.py --classes 10 --model default --nsplit 100 --batchsize 50 --lr 0.05 --lr-decay 0.4 --lr-decay-epoch 400 --regularization 0 --reg-inc 0.1 --reg-inc-epoch 800 --epochs 800 --iterations 1 --seed 557 --dir $inputdir --valdir $valdir -o $logfile 2>&1 | tee /homes/cx2/federated/results/exp_script_4.log
cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile
mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed_robust_impl2.py --classes 10 --model default --nsplit 100 --batchsize 50 --lr 0.05 --lr-decay 0.4 --lr-decay-epoch 400 --regularization 0 --reg-inc 0.1 --reg-inc-epoch 800 --epochs 800 --iterations 1 --seed 446 --dir $inputdir --valdir $valdir -o $logfile 2>&1 | tee /homes/cx2/federated/results/exp_script_4.log

