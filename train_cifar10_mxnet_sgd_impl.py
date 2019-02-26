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

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, help="dir of the data", required=True)
parser.add_argument("--valdir", type=str, help="dir of the val data", required=True)
parser.add_argument("-b", "--batchsize", type=int, help="batchsize", default=50)
parser.add_argument("-e", "--epochs", type=int, help="number of epochs", default=100)
parser.add_argument("-v", "--interval", type=int, help="log interval (epochs)", default=10)
parser.add_argument("-n", "--nsplit", type=int, help="number of split", default=40)
parser.add_argument("-l", "--lr", type=float, help="learning rate", default=0.001)
parser.add_argument("--momentum", type=float, help="momentum", default=0)
parser.add_argument("-o", "--log", type=str, help="dir of the log file", default='train_cifar100.log')
parser.add_argument("-c", "--classes", type=int, help="number of classes", default=20)
parser.add_argument("-i", "--iterations", type=int, help="number of local epochs", default=50)
parser.add_argument("--model", type=str, help="model", default='mobilenetv2_1.0')
parser.add_argument("--seed", type=int, help="random seed", default=733)
 
args = parser.parse_args()

# print(args, flush=True)

filehandler = logging.FileHandler(args.log)
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

# set random seed
mx.random.seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

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

# lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]

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

nd.waitall()

[val_train_X, val_train_Y] = get_val_train_batch(val_dir)
val_train_dataset = mx.gluon.data.dataset.ArrayDataset(val_train_X, val_train_Y)
val_train_data = gluon.data.DataLoader(val_train_dataset, batch_size=1000, shuffle=False, last_batch='keep', num_workers=1)
train_data = gluon.data.DataLoader(val_train_dataset, batch_size=1000, shuffle=True, last_batch='rollover', num_workers=1)

[val_val_X, val_val_Y] = get_val_val_batch(val_dir)
val_val_dataset = mx.gluon.data.dataset.ArrayDataset(val_val_X, val_val_Y)
val_val_data = gluon.data.DataLoader(val_val_dataset, batch_size=1000, shuffle=False, last_batch='keep', num_workers=1)

trainer.set_learning_rate(lr)

tic = time.time()
for epoch in range(args.epochs):

    # train    
    for i, (data, label) in enumerate(train_data):
        with ag.record():
            outputs = net(data)
            loss = loss_func(outputs, label)
        loss.backward()
        trainer.step(args.batchsize)
    
    nd.waitall()
    
    # validation
    if  epoch % args.interval == 0 or epoch == args.epochs-1:
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

        logger.info('[Epoch %d] validation: acc-top1=%f acc-top5=%f, loss=%f, lr=%f, time=%f' % (epoch, top1, top5, crossentropy, trainer.learning_rate, time.time()-tic))
        tic = time.time()
        
        nd.waitall()



            






