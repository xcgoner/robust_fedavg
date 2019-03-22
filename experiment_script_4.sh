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

watchfile=/homes/cx2/federated/results/experiment_script_4.log


logfile=/homes/cx2/federated/results/fedrob_4.txt

> $logfile

cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile
mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/fedrob/fedrob_bptt.py --model default --nsplit 100 --nbyz 0 --trim 0 --lr 20 --alpha 1 --alpha-decay 0.9 --alpha-decay-epoch 400 --epochs 800 --iterations 2 --seed 733 --dir $inputdir --valdir $valdir --log $logfile 2>&1 | tee $watchfile