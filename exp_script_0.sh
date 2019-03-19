#!/bin/bash
#PBS -l select=1:ncpus=112 -lplace=excl

source /opt/intel/compilers_and_libraries_2017/linux/mpi/bin64/mpivars.sh
source activate mxnet_latest


### OPA FABRIC ###
export I_MPI_FABRICS=ofi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=56

### OPA FABRIC ###
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

export KMP_AFFINITY=granularity=fine,compact,1,0;


# cat $PBS_O_WORKDIR/hostfile

inputdir=/homes/cx2/datasets/cifar10_normalized/sgd

valdir=/homes/cx2/datasets/cifar10_normalized/sgd

watchfile=/homes/cx2/federated/results/exp_script_0.log


logfile=/homes/cx2/federated/results/sgd_1.txt

> $logfile

python /homes/cx2/federated/fedrob/sgd.py --classes 10 --model default --nsplit 100 --nworkers 10 --batchsize 50 --lr 0.1 --epochs 800 --seed 733 --interval 1 --dir $inputdir --valdir $valdir --log $logfile 2>&1 | tee $watchfile