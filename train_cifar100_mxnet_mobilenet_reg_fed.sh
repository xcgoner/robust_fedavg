#!/bin/bash
#PBS -l select=8:ncpus=272 -lplace=excl

source /opt/intel/compilers_and_libraries_2017/linux/mpi/bin64/mpivars.sh
source activate mxnet_mpi


### OPA FABRIC ###
export I_MPI_FABRICS=ofi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=64

### OPA FABRIC ###
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

export KMP_AFFINITY=granularity=fine,compact,1,0;

cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile

# cat $PBS_O_WORKDIR/hostfile

logfile=/homes/cx2/federated/results/train_cifar100_mxnet_mobilenet_reg_fed.txt

> $logfile

# mpirun -np 8 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed.py --nsplit 40 --batchsize 128 --lr 0.04 --regularization 0.1 --epochs 800 --dir /homes/cx2/datasets/cifar100 -o $logfile 2>&1 | tee /homes/cx2/federated/results/train_cifar100_mxnet_mobilenet_reg_fed.log
mpirun -np 16 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/train_cifar100_mxnet_mobilenet_reg_fed.py --granularity coarse --nsplit 40 --batchsize 16 --lr 0.01 --regularization 5 --epochs 800 --dir /homes/cx2/datasets/cifar100_coarse -o $logfile 2>&1 | tee /homes/cx2/federated/results/train_cifar100_mxnet_mobilenet_reg_fed.log