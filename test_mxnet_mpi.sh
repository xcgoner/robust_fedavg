#!/bin/bash
#PBS -l select=1:ncpus=272 -lplace=excl

source /opt/intel/compilers_and_libraries_2017/linux/mpi/bin64/mpivars.sh
source activate mxnet_mpi


### OPA FABRIC ###
export I_MPI_FABRICS=ofi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=32

### OPA FABRIC ###
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile

cat $PBS_O_WORKDIR/hostfile

# cd ~/federated/robust_fedavg

mpirun -np 5 -machinefile $PBS_O_WORKDIR/hostfile python /homes/cx2/federated/robust_fedavg/test_mxnet_mpi.py 2>&1 | tee /homes/cx2/federated/results/test_mxnet_mpi.log