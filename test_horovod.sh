#!/bin/bash
#PBS -l select=1:ncpus=272 -lplace=excl

source /opt/intel/compilers_and_libraries_2018.0.128/linux/mpi/intel64/bin/mpivars.sh
source activate tensorflow_hvd

### OPA FABRIC ###
export I_MPI_FABRICS=ofi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=66

### OPA FABRIC ###
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile

cat $PBS_O_WORKDIR/hostfile

# cd ~/federated/robust_fedavg

mpirun -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/test_horovod.py 2>&1 | tee /homes/cx2/federated/results/test_horovod.log