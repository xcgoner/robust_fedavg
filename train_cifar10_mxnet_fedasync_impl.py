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
parser.add_argument("--valdir", type=str, help="dir of the val data", required=True)
parser.add_argument("-b", "--batchsize", type=int, help="batchsize", default=50)
parser.add_argument("-e", "--epochs", type=int, help="number of epochs", default=100)
parser.add_argument("-v", "--interval", type=int, help="log interval (epochs)", default=10)
parser.add_argument("-n", "--nsplit", type=int, help="number of split", default=40)
parser.add_argument("-l", "--lr", type=float, help="learning rate", default=0.001)
parser.add_argument("--momentum", type=float, help="momentum", default=0)
# instead of define the weight of the regularization, we define lr * regularization weight, this option is equivalent to \gamma \times \rho in the paper
parser.add_argument("--rho", type=float, help="regularization \times lr", default=0.00001)
parser.add_argument("--alpha", type=float, help="mixing hyperparameter", default=0.9)
parser.add_argument("-o", "--log", type=str, help="dir of the log file", default='train_cifar100.log')
parser.add_argument("-c", "--classes", type=int, help="number of classes", default=20)
parser.add_argument("-i", "--iterations", type=int, help="number of local epochs", default=50)
parser.add_argument("-a", "--aggregation", type=str, help="aggregation method, mean or trim", default='mean')
parser.add_argument("--lr-decay", type=float, help="lr decay rate", default=0.1)
parser.add_argument("--lr-decay-epoch", type=str, help="lr decay epoch", default='400')
# --iid 0 means non-IID
parser.add_argument("--iid", type=int, help="IID setting", default=0)
parser.add_argument("--model", type=str, help="model", default='mobilenetv2_1.0')
parser.add_argument("--seed", type=int, help="random seed", default=733)
parser.add_argument("--server-send-interval", type=int, help="interval of server sending model back to workers", default=3)
 
args = parser.parse_args()
# args.nsplit = mpi_size

# print(args, flush=True)

filehandler = logging.FileHandler(args.log)
streamhandler = logging.StreamHandler()

# log is only printed by worker 0
if mpi_rank == 0:
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

# set random seed
mx.random.seed(args.seed + mpi_rank)
random.seed(args.seed + mpi_rank)
np.random.seed(args.seed + mpi_rank)

# set path
data_dir = os.path.join(args.dir, 'dataset_split_{}'.format(args.nsplit))
train_dir = os.path.join(data_dir, 'train')
# path to validation data
val_dir = os.path.join(args.valdir, 'val')

training_files = []
for filename in sorted(listdir(train_dir)):
    absolute_filename = os.path.join(train_dir, filename)
    training_files.append(absolute_filename)

context = mx.cpu()

classes = args.classes

# load training data
def get_train_batch(train_filename):
    with open(train_filename, "rb") as f:
        B, L = pickle.load(f)
    return nd.transpose(nd.array(B), (0, 3, 1, 2)), nd.array(L)

# load validation data
def get_val_train_batch(data_dir):
    test_filename = os.path.join(data_dir, 'train_data.pkl')
    with open(test_filename, "rb") as f:
        B, L = pickle.load(f)
    return nd.transpose(nd.array(B), (0, 3, 1, 2)), nd.array(L)

def get_val_val_batch(data_dir):
    test_filename = os.path.join(data_dir, 'val_data.pkl')
    with open(test_filename, "rb") as f:
        B, L = pickle.load(f)
    return nd.transpose(nd.array(B), (0, 3, 1, 2)), nd.array(L)

model_name = args.model

if model_name == 'default':
    net = gluon.nn.Sequential()
    with net.name_scope():
        #  First convolutional layer
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Dropout(rate=0.25))
        #  Second convolutional layer
        # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # Third convolutional layer
        net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1,1), activation='relu'))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1,1), activation='relu'))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Dropout(rate=0.25))
        # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # Flatten and apply fullly connected layers
        net.add(gluon.nn.Flatten())
        # net.add(gluon.nn.Dense(512, activation="relu"))
        # net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dropout(rate=0.25))
        net.add(gluon.nn.Dense(classes))
else:
    model_kwargs = {'ctx': context, 'pretrained': False, 'classes': classes}
    net = get_model(model_name, **model_kwargs)

# initialization
if model_name.startswith('cifar') or model_name == 'default':
    net.initialize(mx.init.Xavier(), ctx=context)
else:
    net.initialize(mx.init.MSRAPrelu(), ctx=context)

# # no weight decay
# for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
#     v.wd_mult = 0.0

# SGD optimizer
optimizer = 'sgd'
lr = args.lr
optimizer_params = {'momentum': args.momentum, 'learning_rate': lr, 'wd': 0.0001}
# optimizer_params = {'momentum': 0.0, 'learning_rate': lr, 'wd': 0.0}

lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]

loss_func = gluon.loss.SoftmaxCrossEntropyLoss()

train_metric = mx.metric.Accuracy()

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
train_cross_entropy = mx.metric.CrossEntropy()

# warmup
print('warm up', flush=True)
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
trainer.set_learning_rate(0.01)
[train_X, train_Y] = get_train_batch(training_files[0])
train_dataset = mx.gluon.data.dataset.ArrayDataset(train_X, train_Y)
train_data = gluon.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, last_batch='rollover', num_workers=1)
for local_epoch in range(5):
    for i, (data, label) in enumerate(train_data):
        with ag.record():
            outputs = net(data)
            loss = loss_func(outputs, label)
        loss.backward()
        trainer.step(args.batchsize)

# # force initialization
# train_data = random.choice(train_data_list)
# for i, (data, label) in enumerate(train_data):
#     outputs = net(data)

if mpi_rank == 0:
    params_prev = [param.data().copy() for param in net.collect_params().values()]
else:
    params_prev = None

nd.waitall()

# broadcast the initialized model
params_prev = mpi_comm.bcast(params_prev, root=0)
for param, param_prev in zip(net.collect_params().values(), params_prev):
    param.set_data(param_prev)

nd.waitall()


rho = args.rho
alpha = args.alpha

if mpi_rank == 0:
    # server
    server_send_list = []
    server_recv_list = [i for i in range(1, mpi_size)]

    [val_train_X, val_train_Y] = get_val_train_batch(val_dir)
    val_train_dataset = mx.gluon.data.dataset.ArrayDataset(val_train_X, val_train_Y)
    val_train_data = gluon.data.DataLoader(val_train_dataset, batch_size=1000, shuffle=False, last_batch='keep', num_workers=1)

    [val_val_X, val_val_Y] = get_val_val_batch(val_dir)
    val_val_dataset = mx.gluon.data.dataset.ArrayDataset(val_val_X, val_val_Y)
    val_val_data = gluon.data.DataLoader(val_val_dataset, batch_size=1000, shuffle=False, last_batch='keep', num_workers=1)

    for epoch in range(args.epochs):
        # receive model from 
        nd.waitall()
        if not server_send_list and not server_recv_list:
            logger.info('warning: no send and recv')

        if server_recv_list:
            random.shuffle(server_recv_list)
            recv_worker = server_recv_list.pop(0)
            print('Rank %02d, recv %02d' % (mpi_rank, recv_worker), flush=True)
            recv_params = mpi_comm.recv(source=recv_worker, tag=0)
            for param, param_nd in zip(net.collect_params().values(), recv_params):
                weight = param.data()
                weight[:] = weight * (1-alpha) + param_nd * alpha
            server_send_list.append(recv_worker)

        # server send model
        if server_send_list:
            random.shuffle(server_send_list)
            if epoch > 0 and epoch % args.server_send_interval == 0:
                send_worker = server_send_list.pop(0)
                send_params = [param.data().copy() for param in net.collect_params().values()]
                print('Rank %02d, send %02d' % (mpi_rank, send_worker), flush=True)
                mpi_comm.send(send_params, dest=send_worker, tag=0)
                server_recv_list.append(send_worker)
        
        # validation
        if  ( epoch % args.interval == 0 or epoch == args.epochs-1 ) :
            acc_top1.reset()
            acc_top5.reset()
            train_cross_entropy.reset()
            # get accuracy on testing data
            for i, (data, label) in enumerate(val_val_data):
                outputs = net(data)
                acc_top1.update(label, outputs)
                acc_top5.update(label, outputs)

            # get cross entropy loss on traininig data
            for i, (data, label) in enumerate(val_train_data):
                outputs = net(data)
                train_cross_entropy.update(label, nd.softmax(outputs))

            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            _, crossentropy = train_cross_entropy.get()

            logger.info('[Epoch %d] validation: acc-top1=%f acc-top5=%f, loss=%f, lr=%f, rho=%f' % (epoch, top1, top5, crossentropy, trainer.learning_rate, rho))
            
            nd.waitall()
else:
    # worker
    [train_X, train_Y] = get_train_batch(training_files[mpi_rank-1])
    train_dataset = mx.gluon.data.dataset.ArrayDataset(train_X, train_Y)
    train_data = gluon.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, last_batch='rollover', num_workers=1)
    for epoch in range(1, args.epochs):

        # # learning rate decay
        # if epoch in lr_decay_epoch:
        #     lr = lr * args.lr_decay
        #     rho = rho * args.lr_decay

        # reset optimizer
        trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
        trainer.set_learning_rate(lr)

        if rho > 0:
            # param_prev is the initial model, pulled at the beginning of the epoch
            # param_prev is rescaled by rho
            for param, param_prev in zip(net.collect_params().values(), params_prev):
                if param.grad_req != 'null':
                    param_prev[:] = param_prev * rho

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
                if rho > 0:
                    for param, param_prev in zip(net.collect_params().values(), params_prev):
                        if param.grad_req != 'null':
                            weight = param.data()
                            # param_prev is already rescaled by rho before
                            weight[:] = weight * (1-rho) + param_prev
                trainer.step(args.batchsize)
        
        nd.waitall()
        # push model
        params_np = [param.data().copy() for param in net.collect_params().values()]
        print('Rank %02d, send %02d' % (mpi_rank, 0), flush=True)
        mpi_comm.send(params_np, dest=0, tag=0)
        # pull model
        print('Rank %02d, recv %02d' % (mpi_rank, 0), flush=True)
        params_prev = mpi_comm.recv(source=0, tag=0)
        for param, param_prev in zip(net.collect_params().values(), params_prev):
            param.set_data(param_prev)

        # validation
        nd.waitall()


            






