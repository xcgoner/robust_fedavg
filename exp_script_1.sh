#!/bin/bash
#PBS -l select=11:ncpus=112 -lplace=excl

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

basedir=/homes/cx2/datasets
logdir=/homes/cx2/federated/results

# training data
inputdir=$basedir/cifar10_normalized_async

# validation data
valdir=$basedir/cifar10_normalized_async/dataset_split_10

watchfile=$logdir/exp_script_1.log

model="default"
lr=0.1
rho=0.01
alpha=0.9
for threshold in 0.94 0.96 0.98
do
    for interval in 2 4 6 8 12
    do
        logfile=$logdir/fed_async_cifar10_${model}_${lr}_${rho}_${alpha}_${threshold}_${interval}.txt

        > $logfile
        for seed in 337 773 557 755 575 737 373 757 224 442
        do
            # hostfile
            cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile

            # start training
            mpirun -np 11 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/fed_async/train_cifar10_mxnet_fedasync_impl.py --classes 10 --model ${model} --nsplit 10 --batchsize 50 --lr ${lr} --rho ${rho} --alpha ${alpha} --alpha-decay 0.5 --alpha-decay-epoch 200,300 --epochs 400 --server-send-threshold ${threshold} --queue-shuffle-interval ${interval} --iterations 1 --seed ${seed} --dir $inputdir --valdir $valdir -o $logfile 2>&1 | tee $watchfile
        done
    done
done
