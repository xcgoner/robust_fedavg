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

inputdir=/homes/cx2/datasets/wikitext2_normalized

valdir=/homes/cx2/datasets/wikitext2_normalized/dataset_split_10

watchfile=/homes/cx2/federated/results/exp_script_24.log

lr=40
nbyz=1
trim=1

# for lr in 0.05 0.1 0.15
for alphadecay in 1 0.8 0.6
do
    logfile=/homes/cx2/federated/results/fedrob_bptt_nobyz_balanced_${lr}_${nbyz}_${trim}_${alphadecay}.txt
    > $logfile
    for seed in 337 773 557 755
    do 
        cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile24
        mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile24 python /homes/cx2/federated/fedrob/fedrob_bptt.py --model default --nsplit 100 --nbyz ${nbyz} --trim ${trim} --lr 40 --lr-decay 0.5 --lr-decay-epoch 200 --alpha 1 --alpha-decay ${alphadecay} --alpha-decay-epoch 600 --epochs 1000 --iterations 1 --seed ${seed} --dir $inputdir --valdir $valdir --log $logfile 2>&1 | tee $watchfile
    done
done

