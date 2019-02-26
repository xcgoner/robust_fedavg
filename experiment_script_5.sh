#!/bin/bash
#PBS -l select=10:ncpus=272 -lplace=excl

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

basedir=/homes/cx2/datasets
logdir=/homes/cx2/federated/results

# training data
inputdir=$basedir/cifar10_normalized_async

# validation data
valdir=$basedir/cifar10_normalized/dataset_split_10

watchfile=$logdir/experiment_script_5.log

logfile=$logdir/experiment_script_5.txt

# prepare the dataset
# python convert_cifar10_to_np_normalized_unbalanced.py --nsplit 100 --normalize 1 --step 8 --output $basedir/cifar10_unbalanced/
# python convert_cifar10_to_np_normalized.py --nsplit 100 --normalize 1 --output $basedir/cifar10_balanced/

> $logfile

# hostfile
cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile

mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/fed_async/train_cifar10_mxnet_fedavg_impl.py --classes 10 --model default --nsplit 100 --batchsize 50 --lr 0.1 --mmode "reset" --lr-decay 1.0 --lr-decay-epoch 2000 --alpha 0.0 --epochs 2000 --iterations 1 --seed 337 --dir $inputdir --valdir $valdir -o $logfile 2>&1 | tee $watchfile
