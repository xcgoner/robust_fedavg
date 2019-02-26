#!/bin/bash
#PBS -l select=5:ncpus=272 -lplace=excl

source /opt/intel/compilers_and_libraries_2017/linux/mpi/bin64/mpivars.sh
source activate mxnet_latest


### OPA FABRIC ###
export I_MPI_FABRICS=ofi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=17


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

watchfile=$logdir/exp_script_13.log

model="default"
# lr=0.1
# rho=0.01
# alpha=0.9

for lr in 0.05
do
    for iterations in 1 2
    do
        logfile=$logdir/fed_async_cifar10_fedavg_baseline_${model}_${lr}_${iterations}.txt

        > $logfile
        for seed in 337 773 557 755 
        do
            # hostfile
            cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile

            mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/fed_async/train_cifar10_mxnet_fedavg_impl.py --classes 10 --model ${model} --nsplit 100 --batchsize 50 --lr ${lr} --mmode "reset" --lr-decay 1.0 --lr-decay-epoch 2000 --alpha 0.0 --epochs 2000 --iterations ${iterations} --seed ${seed} --dir $inputdir --valdir $valdir -o $logfile 2>&1 | tee $watchfile
        done
    done
done

