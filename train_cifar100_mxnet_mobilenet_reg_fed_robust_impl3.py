import argparse, time, logging, os, math, random
os.environ["MXNET_USE_OPERATOR_TUNING"] = "0"


import numpy as np
from scipy import stats
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
# parser.add_argument("-r", "--regularization", type=float, help="factor of regularization", default=0.00001)
parser.add_argument("--reg-factor", type=float, help="increment factor of regularization", default=0.9999)
parser.add_argument("--reg-epsilon", type=float, help="epoch to 0.6 regularization", default=300)
parser.add_argument("-o", "--log", type=str, help="dir of the log file", default='train_cifar100.log')
parser.add_argument("-c", "--classes", type=int, help="number of classes", default=20)
parser.add_argument("-i", "--iterations", type=int, help="number of local epochs", default=50)
parser.add_argument("-a", "--aggregation", type=str, help="aggregation method", default='mean')
parser.add_argument("-q", "--nbyz", type=int, help="number of Byzantine workers", default=0)
parser.add_argument("-t", "--trim", type=int, help="number of trimmed workers on one side", default=0)
parser.add_argument("--lr-decay", type=float, help="lr decay rate", default=0.1)
parser.add_argument("--lr-decay-epoch", type=str, help="lr decay epoch", default='400')
parser.add_argument("--iid", type=int, help="IID setting", default=0)
parser.add_argument("--model", type=str, help="model", default='mobilenetv2_1.0')
 
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

classes = args.classes

def get_train_batch(train_filename):
    with open(train_filename, "rb") as f:
        B, L = pickle.load(f)

    return nd.transpose(nd.array(B.astype('float32') / 255.0), (0, 3, 1, 2)), nd.array(L)

def get_train_batch_byz(train_filename):
    with open(train_filename, "rb") as f:
        B, L = pickle.load(f)

    return nd.transpose(nd.array(B.astype('float32') / 255.0), (0, 3, 1, 2)), nd.array(classes - 1 - L)

def get_test_batch(data_dir):
    test_filename = os.path.join(data_dir, "val_data.pkl")
    with open(test_filename, "rb") as f:
        B, L = pickle.load(f)
    
    return nd.transpose(nd.array(B.astype('float32') / 255.0), (0, 3, 1, 2)), nd.array(L)

train_data_list = []
for training_file in training_files:
    [train_X, train_Y] = get_train_batch(training_file)
    train_dataset = mx.gluon.data.dataset.ArrayDataset(train_X, train_Y)
    train_data = gluon.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, last_batch='rollover', num_workers=1)
    train_data_list.append(train_data)

if mpi_rank == 0:
    [val_X, val_Y] = get_test_batch(val_dir)
    val_dataset = mx.gluon.data.dataset.ArrayDataset(val_X, val_Y)
    val_data = gluon.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, last_batch='keep', num_workers=1)

model_name = args.model

model_kwargs = {'ctx': context, 'pretrained': False, 'classes': classes}
net = get_model(model_name, **model_kwargs)

if model_name.startswith('cifar'):
    net.initialize(mx.init.Xavier(), ctx=context)
else:
    net.initialize(mx.init.MSRAPrelu(), ctx=context)

# # no weight decay
# for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
#     v.wd_mult = 0.0
    
optimizer = 'sgd'
optimizer_params = {'momentum': 0, 'learning_rate': args.lr, 'wd': 0.0001}

lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]

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

if mpi_rank == 0:
    worker_list = list(range(mpi_size))

training_file_index_list = [i for i in range(len(training_files))]

epsilon = args.reg_epsilon

for epoch in range(args.epochs):
        # train_metric.reset()

        tic = time.time()

        if args.iid == 0:
            if mpi_rank == 0:
                random.shuffle(training_file_index_list)
                training_file_index_sublist = training_file_index_list[0:mpi_size]
            else:
                training_file_index_sublist = None
            training_file_index = mpi_comm.scatter(training_file_index_sublist, root=0)
            train_data = train_data_list[training_file_index]


        if epoch in lr_decay_epoch:
            trainer.set_learning_rate(trainer.learning_rate * args.lr_decay)

        alpha = args.reg_factor * ( 1 - epsilon / (math.sqrt(epoch) + epsilon) )

        trainer.set_learning_rate(trainer.learning_rate * (1-alpha))

        if alpha > 0:
            for param, param_prev in zip(net.collect_params().values(), params_prev):
                if param.grad_req != 'null':
                    param_prev[:] = param_prev * alpha

        # select byz workers
        if args.nbyz > 0:
            if mpi_rank == 0:
                random.shuffle(worker_list)
                byz_worker_list = worker_list[0:args.nbyz]
            else:
                byz_worker_list = None
            byz_worker_list = mpi_comm.bcast(byz_worker_list, root=0)
        else:
            byz_worker_list = []

        if mpi_rank in byz_worker_list:
            # byz worker
            [byz_train_X, byz_train_Y] = get_train_batch_byz(random.choice(training_files))
            byz_train_dataset = mx.gluon.data.dataset.ArrayDataset(byz_train_X, byz_train_Y)
            byz_train_data = gluon.data.DataLoader(byz_train_dataset, batch_size=args.batchsize, shuffle=True, last_batch='rollover', num_workers=1)
            net.initialize(mx.init.MSRAPrelu(), ctx=context, force_reinit=True)
            for local_epoch in range(args.iterations):
                for i, (data, label) in enumerate(byz_train_data):
                    with ag.record():
                        outputs = net(data)
                        loss = loss_func(outputs, label)
                    loss.backward()
                    trainer.step(args.batchsize)
        else:
            # train
            # local epoch
            for local_epoch in range(args.iterations):
                if args.iid == 1:
                    train_data = random.choice(train_data_list)
                for i, (data, label) in enumerate(train_data):
                    with ag.record():
                        outputs = net(data)
                        loss = loss_func(outputs, label)
                    loss.backward()
                    # add regularization
                    if alpha > 0:
                        for param, param_prev in zip(net.collect_params().values(), params_prev):
                            if param.grad_req != 'null':
                                weight = param.data()
                                weight[:] = weight * (1-alpha) + param_prev
                    trainer.step(args.batchsize)
                    # train_metric.update(label, outputs)
        
        # aggregation
        nd.waitall()
        params_prev_np = [param.data().copy().asnumpy() for param in net.collect_params().values()]
        params_prev_np_list = mpi_comm.gather(params_prev_np, root=0)
        if mpi_rank == 0:
            n_params = len(params_prev_np)
            if args.aggregation == "trim" or args.trim > 0:
                params_prev_np = [ ( stats.trim_mean( np.stack( [params[j] for params in params_prev_np_list], axis=0), args.trim/mpi_size, axis=0 ) ) for j in range(n_params) ]
            else:
                params_prev_np = [ ( np.mean( np.stack( [params[j] for params in params_prev_np_list], axis=0), axis=0 ) ) for j in range(n_params) ]
        else:
            params_prev_np = None
        params_prev_np = mpi_comm.bcast(params_prev_np, root=0)
        params_prev = [ nd.array(param_np) for param_np in params_prev_np ]
        for param, param_prev in zip(net.collect_params().values(), params_prev):
            param.set_data(param_prev)

        # test
        nd.waitall()

        toc = time.time()

        if mpi_rank == 0 and ( epoch % args.interval == 0 or epoch == args.epochs-1 ) :
            acc_top1.reset()
            acc_top5.reset()
            for i, (data, label) in enumerate(val_data):
                outputs = net(data)
                acc_top1.update(label, outputs)
                acc_top5.update(label, outputs)

            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()

            logger.info('[Epoch %d] validation: acc-top1=%f acc-top5=%f, time=%f, lr=%f, alpha=%f'%(epoch, top1, top5, toc-tic, trainer.learning_rate, alpha))
            # logger.info('[Epoch %d] validation: acc-top1=%f acc-top5=%f'%(epoch, top1, top5))
        






