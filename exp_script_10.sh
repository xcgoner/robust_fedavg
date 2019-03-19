#!/bin/bash
#PBS -l select=5:ncpus=112 -lplace=excl

source /opt/intel/compilers_and_libraries_2017/linux/mpi/bin64/mpivars.sh
source activate mxnet_latest


### OPA FABRIC ###
export I_MPI_FABRICS=ofi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=28

### OPA FABRIC ###
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

export KMP_AFFINITY=granularity=fine,compact,1,0;


# cat $PBS_O_WORKDIR/hostfile

inputdir=/homes/cx2/datasets/cifar10_normalized

valdir=/homes/cx2/datasets/cifar10_normalized/dataset_split_10

watchfile=/homes/cx2/federated/results/exp_script_10.log


for lr in 0.05 0.1 0.15
do
    for alphadecay in 0.9
    do
        for nbyz in 4
        do
            logfile=/homes/cx2/federated/results/fedrob_byz_balanced_${lr}_${alphadecay}_${nbyz}_${nbyz}.txt
            > $logfile
            for seed in 337 773 557 755 
            do 
                cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile10
                mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile10 python /homes/cx2/federated/fedrob/fedrob.py --classes 10 --model default --nsplit 100 --batchsize 50 --nbyz ${nbyz} --trim ${nbyz} --lr ${lr} --alpha 1 --alpha-decay ${alphadecay} --alpha-decay-epoch 400 --epochs 800 --iterations 1 --seed ${seed} --dir $inputdir --valdir $valdir --log $logfile 2>&1 | tee $watchfile
            done
        done
    done
done

for lr in 0.05 0.1 0.15
do
    for alphadecay in 0.9
    do
        for nbyz in 0 2
        do
            for trim in 2 3 4
            do
                logfile=/homes/cx2/federated/results/fedrob_byz_balanced_${lr}_${alphadecay}_${nbyz}_${trim}.txt
                > $logfile
                for seed in 337 773 557 755 
                do 
                    cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile10
                    mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile10 python /homes/cx2/federated/fedrob/fedrob.py --classes 10 --model default --nsplit 100 --batchsize 50 --nbyz ${nbyz} --trim ${trim} --lr ${lr} --alpha 1 --alpha-decay ${alphadecay} --alpha-decay-epoch 400 --epochs 800 --iterations 1 --seed ${seed} --dir $inputdir --valdir $valdir --log $logfile 2>&1 | tee $watchfile
                done
            done
        done
    done
done

