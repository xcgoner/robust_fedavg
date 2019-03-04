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

basedir=/homes/cx2/datasets
logdir=/homes/cx2/federated/results

# training data
inputdir=$basedir/cifar10_normalized_async

# validation data
valdir=$basedir/cifar10_normalized_async/dataset_split_100

watchfile=$logdir/exp_script_2.log

model="default"
lr=0.1
rho=0.005
alpha=0.6

for maxdelay in 8 12 16 20
do
    for iterations in 1
        do
        logfile=$logdir/fed_async_cifar10_singlethread_testdelay_${model}_${lr}_${rho}_${alpha}_${maxdelay}_${iterations}.txt

        > $logfile
        for seed in 337 773 557 755 
        do
            python /homes/cx2/federated/fed_async/train_cifar10_mxnet_fedasync_singlethread_impl.py --classes 10 --model ${model} --nsplit 100 --batchsize 50 --lr ${lr} --rho ${rho} --alpha ${alpha} --alpha-decay 0.5 --alpha-decay-epoch 800 --epochs 2000 --max-delay ${maxdelay} --iterations ${iterations} --seed ${seed} --dir $inputdir --valdir $valdir -o $logfile 2>&1 | tee $watchfile
        done
    done
done
