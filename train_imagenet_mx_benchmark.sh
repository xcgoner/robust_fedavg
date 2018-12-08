#!/bin/bash
#PBS -l select=1:ncpus=272 -lplace=excl

source activate mxnet

export OMP_NUM_THREADS=44;
export KMP_AFFINITY=granularity=fine,compact,1,0;

cd ~/federated/robust_fedavg
time numactl -p 1 python train_imagenet_benchmark.py --rec-train /export/shared/uiuc/imagenet/ILSVRC2012/datasets_rec/rec/train.rec --rec-train-idx /export/shared/uiuc/imagenet/ILSVRC2012/datasets_rec/rec/train.idx --rec-val /export/shared/uiuc/imagenet/ILSVRC2012/datasets_rec/rec/val.rec --rec-val-idx /export/shared/uiuc/imagenet/ILSVRC2012/datasets_rec/rec/val.idx --model resnet50_v1 --mode hybrid --lr 0.4 --lr-mode cosine --num-epochs 120 --batch-size 128 --num-gpus 0 -j 10 --warmup-epochs 5 --dtype float32 --log-interval 200 --last-gamma --no-wd --label-smoothing 2>&1 | tee ~/federated/results/train_imagenet_benchmark.log