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

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batchsize", type=int, help="batchsize", default=8)
parser.add_argument("-e", "--epochs", type=int, help="epochs", default=100)
parser.add_argument("-l", "--lr", type=float, help="learning rate", default=0.001)
parser.add_argument("-o", "--log", type=str, help="dir of the log file", default='train_cifar100.log')
args = parser.parse_args()

print(args, flush=True)

filehandler = logging.FileHandler(args.log)
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

def transformer(data, label):
    # data = mx.image.imresize(data, 224, 224)
    data = mx.nd.transpose(data, (2,0,1))
    data = data.astype(np.float32)
    return data, label

train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR100(train=True, transform=transformer),
    batch_size=args.batchsize, shuffle=True, last_batch='discard', num_workers=1)

test_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR100(train=False, transform=transformer),
    batch_size=1024, shuffle=False, last_batch='discard', num_workers=1)

classes = 20
context = mx.cpu()

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

for epoch in range(args.epochs):
    # train_metric.reset()

    # train
    for i, (data, label) in enumerate(train_data):
        with ag.record():
            outputs = net(data)
            loss = loss_func(outputs, label)
        loss.backward()
        trainer.step(args.batchsize)
        # train_metric.update(label, outputs)

    # test
    acc_top1.reset()
    acc_top5.reset()
    for i, (data, label) in enumerate(test_data):
        outputs = net(data)
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()

    logger.info('[Epoch %d] validation: acc-top1=%f acc-top5=%f'%(epoch, top1, top5))
    # logger.info('[Epoch %d] validation: acc-top1=%f acc-top5=%f'%(epoch, top1, top5))






