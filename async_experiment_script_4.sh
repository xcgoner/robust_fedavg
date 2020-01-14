#!/bin/bash
#PBS -l select=1:ncpus=112 -lplace=excl

source activate mxnet_mkl


export KMP_AFFINITY=granularity=fine,compact,1,0;

cd /homes/cx2/src/robust_fedavg

lr=20
rho=0.0001
maxdelay=16
alpha=0.4
atype="hinge"
a=10
b=4
logfile=results/fed_async_wikitext_fedasync_${lr}_${rho}_${maxdelay}_${atype}_${alpha}.log

python train_wikitext_mxnet_fedasync_singlethread_impl.py --rho ${rho} \
--alpha ${alpha} --alpha-decay 1 --alpha-adaptive ${a} --alpha-adaptive2 ${b} \
--alpha-type ${atype} --max-delay ${maxdelay} --log ${logfile} --interval 100
