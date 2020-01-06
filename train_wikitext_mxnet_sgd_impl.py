import argparse, time, logging, os, math, random
os.environ["MXNET_USE_OPERATOR_TUNING"] = "0"


import numpy as np
from scipy import stats
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.utils import download

import gluonnlp as nlp

from os import listdir
import os.path
import argparse

import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batchsize", type=int, help="batchsize", default=20)
parser.add_argument("-e", "--epochs", type=int, help="number of epochs", default=20)
parser.add_argument("-v", "--interval", type=int, help="log interval (epochs)", default=1)
parser.add_argument("-n", "--nsplit", type=int, help="number of split", default=40)
parser.add_argument("-l", "--lr", type=float, help="learning rate", default=20)
parser.add_argument("--momentum", type=float, help="momentum", default=0)
parser.add_argument("-o", "--log", type=str, help="dir of the log file", default='train_wikitext.log')
parser.add_argument("-i", "--iterations", type=int, help="number of local epochs", default=50)
parser.add_argument("--model", type=str, help="model", default='standard_lstm_lm_200')
parser.add_argument("--seed", type=int, help="random seed", default=733)
 
args = parser.parse_args()

# print(args, flush=True)

filehandler = logging.FileHandler(args.log)
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

bptt = 35
grad_clip = 0.25
batch_size = args.batchsize

# set random seed
mx.random.seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

dataset_name = 'wikitext-2'

# Load the dataset
train_dataset, val_dataset, test_dataset = [
    nlp.data.WikiText2(
        segment=segment, bos=None, eos='<eos>', skip_empty=False)
    for segment in ['train', 'val', 'test']
]

# Extract the vocabulary and numericalize with "Counter"
vocab = nlp.Vocab(
    nlp.data.Counter(train_dataset), padding_token=None, bos_token=None)

# Batchify for BPTT
bptt_batchify = nlp.data.batchify.CorpusBPTTBatchify(
    vocab, bptt, batch_size, last_batch='discard')
train_data, val_data, test_data = [
    bptt_batchify(x) for x in [train_dataset, val_dataset, test_dataset]
]

context = [mx.cpu()]

model_name = args.model

net, vocab = nlp.model.get_model(model_name, vocab=vocab, dataset_name=None)

# initialization
net.initialize(mx.init.Xavier(), ctx=context)

# # no weight decay
# for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
#     v.wd_mult = 0.0

# SGD optimizer
optimizer = 'sgd'
lr = args.lr
optimizer_params = {'momentum': args.momentum, 'learning_rate': lr, 'wd': 0}
# optimizer_params = {'momentum': 0.0, 'learning_rate': lr, 'wd': 0.0}

# lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]

loss_func = gluon.loss.SoftmaxCrossEntropyLoss()

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(i) for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


# Note that ctx is short for context
def evaluate(model, data_source, batch_size, ctx):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(
        batch_size=batch_size, func=mx.nd.zeros, ctx=ctx)
    for i, (data, target) in enumerate(data_source):
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        output, hidden = model(data, hidden)
        hidden = detach(hidden)
        L = loss(output.reshape(-3, -1), target.reshape(-1))
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal

# warmup
# print('warm up', flush=True)
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
# trainer.set_learning_rate(0.01)
# [train_X, train_Y] = get_train_batch(training_files[0])
# train_dataset = mx.gluon.data.dataset.ArrayDataset(train_X, train_Y)
# train_data = gluon.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, last_batch='rollover', num_workers=1)
# for local_epoch in range(5):
#     for i, (data, label) in enumerate(train_data):
#         with ag.record():
#             outputs = net(data)
#             loss = loss_func(outputs, label)
#         loss.backward()
#         trainer.step(args.batchsize)

# nd.waitall()

trainer.set_learning_rate(lr)

parameters = net.collect_params().values()

tic = time.time()
for epoch in range(args.epochs):

    # train    
    hiddens = [model.begin_state(batch_size//len(context), func=mx.nd.zeros, ctx=ctx)
                for ctx in context]
    for i, (data, target) in enumerate(train_data):
        data_list = gluon.utils.split_and_load(data, context,
                                                batch_axis=1, even_split=True)
        target_list = gluon.utils.split_and_load(target, context,
                                                    batch_axis=1, even_split=True)
        hiddens = detach(hiddens)
        L = 0
        Ls = []

        with autograd.record():
            for j, (X, y, h) in enumerate(zip(data_list, target_list, hiddens)):
                output, h = net(X, h)
                batch_L = loss_func(output.reshape(-3, -1), y.reshape(-1,))
                L = L + batch_L.as_in_context(context[0]) / (len(context) * X.size)
                Ls.append(batch_L / (len(context) * X.size))
                hiddens[j] = h
        L.backward()
        grads = [p.grad(x.context) for p in parameters for x in data_list]
        gluon.utils.clip_global_norm(grads, grad_clip)

        trainer.step(1)
    
    nd.waitall()
    
    # validation
    if  epoch % args.interval == 0 or epoch == args.epochs-1:
        val_L = evaluate(model, test_data, batch_size, context[0])

        logger.info('[Epoch %d] test: loss=%f, ppl=%f, lr=%f, time=%f' % (epoch, val_L, math.exp(val_L), trainer.learning_rate, time.time()-tic))
        tic = time.time()
        
        nd.waitall()



            






