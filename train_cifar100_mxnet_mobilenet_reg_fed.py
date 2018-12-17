import argparse, time, logging, os, math, random
os.environ["MXNET_USE_OPERATOR_TUNING"] = "0"


import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRScheduler

from os import listdir
import os.path
import argparse

import pickle

from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

print('rank: %d' % (mpi_rank), flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, help="dir of the data", required=True)
parser.add_argument("-b", "--batchsize", type=int, help="batchsize", default=8)
parser.add_argument("-e", "--epochs", type=int, help="epochs", default=100)
parser.add_argument("-v", "--interval", type=int, help="log interval", default=10)
parser.add_argument("-n", "--nsplit", type=int, help="number of split", default=40)
parser.add_argument("-l", "--lr", type=float, help="learning rate", default=0.001)
parser.add_argument("-r", "--regularization", type=float, help="weight of regularization", default=0.00001)
parser.add_argument("-o", "--log", type=str, help="dir of the log file", default='train_cifar100.log')
parser.add_argument("-g", "--granularity", type=str, help="fine or coarse", default='fine')
 
args = parser.parse_args()

print(args, flush=True)

filehandler = logging.FileHandler(args.log)
streamhandler = logging.StreamHandler()

if mpi_rank == 0:
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

data_dir = os.path.join(args.dir, 'dataset_split_{}'.format(args.nsplit))
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

training_files = []
for filename in sorted(listdir(train_dir)):
    absolute_filename = os.path.join(train_dir, filename)
    training_files.append(absolute_filename)

context = mx.cpu()

if args.granularity == 'fine':
    classes = 100
else:
    classes = 20

def get_train_batch(train_filename):
    with open(train_filename, "rb") as f:
        B, L = pickle.load(f)

    return nd.transpose(nd.array(B.astype('float32') / 255.0), (0, 3, 1, 2)), nd.array(L)

def get_test_batch(data_dir):
    test_filename = os.path.join(data_dir, "val_data.pkl")
    with open(test_filename, "rb") as f:
        B, L = pickle.load(f)
    
    return nd.transpose(nd.array(B.astype('float32') / 255.0), (0, 3, 1, 2)), nd.array(L)

# partition dataset
if mpi_rank == 0:
    training_file_index = [i for i in range(len(training_files))]
    random.shuffle(training_file_index)
    training_file_index_list = [index_list.astype('int').tolist() for index_list in np.array_split(training_file_index, mpi_size)]
    # print(training_file_index_list, flush=True)
else:
    training_file_index_list = None
training_file_index_list = mpi_comm.scatter(training_file_index_list, root=0)
sub_training_files = [training_files[i] for i in training_file_index_list]

train_data_list = []
for training_file in sub_training_files:
    [train_X, train_Y] = get_train_batch(training_file)
    train_dataset = mx.gluon.data.dataset.ArrayDataset(train_X, train_Y)
    train_data = gluon.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, last_batch='rollover', num_workers=1)
    train_data_list.append(train_data)

if mpi_rank == 0:
    [val_X, val_Y] = get_test_batch(val_dir)
    val_dataset = mx.gluon.data.dataset.ArrayDataset(val_X, val_Y)
    val_data = gluon.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, last_batch='keep', num_workers=1)

model_kwargs = {'ctx': context, 'pretrained': False, 'classes': classes}
net = get_model('mobilenetv2_0.25', **model_kwargs)

net.initialize(mx.init.MSRAPrelu(), ctx=context)

# no weight decay
for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
    v.wd_mult = 0.0
    
optimizer = 'sgd'
optimizer_params = {'momentum': 0, 'learning_rate': args.lr, 'wd': 0}

trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

loss_func = gluon.loss.SoftmaxCrossEntropyLoss()

train_metric = mx.metric.Accuracy()

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)

# warmup

print('warm up', flush=True)
train_data = random.choice(train_data_list)
for i, (data, label) in enumerate(train_data):
    with ag.record():
        outputs = net(data)
        loss = loss_func(outputs, label)
    loss.backward()
    trainer.step(args.batchsize)

if mpi_rank == 0:
    params_prev = [param.data().copy() for param in net.collect_params().values()]
else:
    params_prev = None

nd.waitall()

# broadcast
params_prev = mpi_comm.bcast(params_prev, root=0)
for param, param_prev in zip(net.collect_params().values(), params_prev):
    param.set_data(param_prev)

for epoch in range(args.epochs):
        # train_metric.reset()

        train_data = random.choice(train_data_list)

        # train
        for i, (data, label) in enumerate(train_data):
            if i > 50:
                break
            with ag.record():
                outputs = net(data)
                loss = loss_func(outputs, label)
            loss.backward()
            # add regularization
            for param, param_prev in zip(net.collect_params().values(), params_prev):
                if param.grad_req != 'null':
                    grad = param.grad()
                    # nd.elemwise_add(x, y, out=y)
                    grad[:] = grad + args.regularization * ( param.data() - param_prev )
            trainer.step(args.batchsize)
            # train_metric.update(label, outputs)
        
        # aggregation
        nd.waitall()
        params_prev_np = [param.data().copy().asnumpy() for param in net.collect_params().values()]
        params_prev_np_list = mpi_comm.gather(params_prev_np, root=0)
        if mpi_rank == 0:
            n_params = len(params_prev_np)
            params_prev_np = [ ( np.mean( np.stack( [params[j] for params in params_prev_np_list], axis=0), axis=0 ) ) for j in range(n_params) ]
        else:
            params_prev_np = None
        params_prev_np = mpi_comm.bcast(params_prev_np, root=0)
        params_prev = [ nd.array(param_np) for param_np in params_prev_np ]
        for param, param_prev in zip(net.collect_params().values(), params_prev):
            param.set_data(param_prev)

        # test
        nd.waitall()
        if mpi_rank == 0:
            acc_top1.reset()
            acc_top5.reset()
            for i, (data, label) in enumerate(val_data):
                outputs = net(data)
                acc_top1.update(label, outputs)
                acc_top5.update(label, outputs)

            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()

            logger.info('[Epoch %d] validation: acc-top1=%f acc-top5=%f'%(epoch, top1, top5))
            # logger.info('[Epoch %d] validation: acc-top1=%f acc-top5=%f'%(epoch, top1, top5))






