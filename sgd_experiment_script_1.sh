#!/bin/bash
#PBS -l select=1:ncpus=112 -lplace=excl

source activate mxnet_mkl


export KMP_AFFINITY=granularity=fine,compact,1,0;

cd /homes/cx2/src/robust_fedavg

lr=20
logfile=results/fed_async_wikitext_sgd.log

python train_wikitext_mxnet_sgd_impl.py --log ${logfile}
