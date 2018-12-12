#!/bin/bash
#PBS -l select=1:ncpus=272 -lplace=excl

source /opt/intel/compilers_and_libraries_2018.0.128/linux/mpi/intel64/bin/mpivars.sh
source activate tensorflow_hvd


# export HFI_NO_CPUAFFINITY=1
export OMP_NUM_THREADS=32
export KMP_AFFINITY=granularity=fine,compact,1,0

# export I_MPI_FABRICS=shm:tmi
# export I_MPI_TMI_PROVIDER=psm2
# export I_MPI_FALLBACK=0
# export I_MPI_HYDRA_BOOTSTRAP=rsh
# export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

### IMPI
export I_MPI_FABRICS=tmi
export I_MPI_DEBUG=6
export I_MPI_HYDRA_BRANCH_COUNT=-1
export I_MPI_FALLBACK=0
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

ulimit -l unlimited

cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile

cd ~/federated/robust_fedavg

mpiexec.hydra -l -n 1 -ppn 1 -hostfile $PBS_O_WORKDIR/hostfile python test_mpi.py 2>&1 | tee ~/federated/results/test_mpi.log