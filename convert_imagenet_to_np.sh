#!/bin/bash
#PBS -l select=1:ncpus=272 -lplace=excl

source activate tensorflow_mkl

export OMP_NUM_THREADS=136;
export KMP_AFFINITY=granularity=fine,compact,1,0;

cd ~/federated/robust_fedavg
# time numactl -p 1 python convert_imagenet_to_np.py --nsplit 80 --size 160 --input /export/shared/uiuc/imagenet/ILSVRC2012/datasets --output /export/shared/uiuc/imagenet/ILSVRC2012/fed_datasets 2>&1 | tee ~/federated/results/convert_imagenet_to_np.log
# time numactl -p 1 python convert_imagenet_to_np.py --size 160 --mode val --input /export/shared/uiuc/imagenet/ILSVRC2012/datasets --output /export/shared/uiuc/imagenet/ILSVRC2012/fed_datasets 2>&1 | tee ~/federated/results/convert_imagenet_to_np.log
# time numactl -p 1 python convert_imagenet_to_np.py --size 160 --mode val --input /export/shared/uiuc/imagenet/ILSVRC2012/datasets --output /export/shared/uiuc/imagenet/ILSVRC2012/fed_datasets 2>&1 | tee ~/federated/results/convert_imagenet_to_np.log
# time numactl -p 1 python convert_imagenet_to_np.py --nsplit 80 --size 128 --input /export/shared/uiuc/imagenet/ILSVRC2012/datasets --output /export/shared/uiuc/imagenet/ILSVRC2012/fed_datasets 2>&1 | tee ~/federated/results/convert_imagenet_to_np.log
time numactl -p 1 python convert_imagenet_to_np.py --size 128 --mode val --input /export/shared/uiuc/imagenet/ILSVRC2012/datasets --output /export/shared/uiuc/imagenet/ILSVRC2012/fed_datasets 2>&1 | tee ~/federated/results/convert_imagenet_to_np_val128.log
