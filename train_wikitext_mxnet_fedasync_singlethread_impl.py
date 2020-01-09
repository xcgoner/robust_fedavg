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
parser.add_argument("-v", "--interval", type=int, help="log interval (epochs)", default=10)
parser.add_argument("-n", "--nsplit", type=int, help="number of split", default=100)
parser.add_argument("-l", "--lr", type=float, help="learning rate", default=20)
parser.add_argument("--momentum", type=float, help="momentum", default=0)
# instead of define the weight of the regularization, we define lr * regularization weight, this option is equivalent to \gamma \times \rho in the paper
parser.add_argument("--rho", type=float, help="regularization \times lr", default=0.00001)
parser.add_argument("--alpha", type=float, help="mixing hyperparameter", default=0.9)
parser.add_argument("-o", "--log", type=str, help="dir of the log file", default='train_cifar100.log')
parser.add_argument("-c", "--classes", type=int, help="number of classes", default=20)
parser.add_argument("-i", "--iterations", type=int, help="number of local epochs", default=4000)
parser.add_argument("-a", "--aggregation", type=str, help="aggregation method, mean or trim", default='mean')
parser.add_argument("--alpha-decay", type=float, help="alpha decay rate", default=0.5)
parser.add_argument("--alpha-decay-epoch", type=str, help="alpha decay epoch", default='3000')
parser.add_argument("--alpha-adaptive", type=float, help="adaptive mixing hyperparameter", default=0)
parser.add_argument("--alpha-adaptive2", type=float, help="adaptive mixing hyperparameter2", default=1)
parser.add_argument("--alpha-type", type=str, help="type of adaptive alpha", default='none')
parser.add_argument("--model", type=str, help="model", default='standard_lstm_lm_200')
parser.add_argument("--seed", type=int, help="random seed", default=733)
parser.add_argument("--max-delay", type=int, help="maximum of global delay", default=10)
 
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
        L = loss_func(output.reshape(-3, -1), target.reshape(-1))
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal

n_devices = args.nsplit
train_data_list = [ [] for i in range(n_devices) ]
for i, (data, target) in enumerate(train_data):
    data_list = gluon.utils.split_and_load(data, context,
                                                batch_axis=1, even_split=True)
    target_list = gluon.utils.split_and_load(target, context,
                                                batch_axis=1, even_split=True)
    train_data_list[i % n_devices].append([data_list, target_list])
print('data cached')

parameters = net.collect_params().values()

# warmup
print('warm up', flush=True)
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
trainer.set_learning_rate(0.1)
# train    
hiddens = [net.begin_state(batch_size//len(context), func=mx.nd.zeros, ctx=ctx)
            for ctx in context]
random.shuffle(train_data_list)
for w in range(n_devices):
    train_worker_data = train_data_list[w]
    random.shuffle(train_worker_data)
    for i, data in enumerate(train_worker_data):
        data_list = data[0]
        target_list = data[1]
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

params_prev = [param.data().copy() for param in net.collect_params().values()]
params_prev_list = [params_prev]
server_ts = 0
ts_list = [server_ts]

nd.waitall()


rho = args.rho
alpha = args.alpha

sum_delay = 0
tic = time.time()
best_val = float("Inf")
n_grads = 0
for epoch in range(args.iterations):

    trainer.set_learning_rate(lr)

    # alpha decay
    if epoch in alpha_decay_epoch:
        alpha = alpha * args.alpha_decay

    # obtain previous model
    model_idx = random.randint(0, len(params_prev_list)-1)
    params_prev = params_prev_list[model_idx]
    for param, param_prev in zip(net.collect_params().values(), params_prev):
        if param.grad_req != 'null':
            weight = param.data()
            weight[:] = param_prev

    params_prev = [param.data().copy() for param in net.collect_params().values()]
    if rho > 0:
        # param_prev is the initial model, pulled at the beginning of the epoch
        # param_prev is rescaled by rho
        for param, param_prev in zip(net.collect_params().values(), params_prev):
            if param.grad_req != 'null':
                param_prev[:] = param_prev * rho
    nd.waitall()

    worker_ts = ts_list[model_idx]

    # train on worker
    # randomly select a local dataset
    train_worker_data = random.choice(train_data_list)
    random.shuffle(train_worker_data)
    # reset optimizer
    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
    trainer.set_learning_rate(lr)     

    # local epoch
    hiddens = [net.begin_state(batch_size//len(context), func=mx.nd.zeros, ctx=ctx)
                    for ctx in context]
    for i, data in enumerate(train_worker_data):
        data_list = data[0]
        target_list = data[1]
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
        n_grads += 1
    
    nd.waitall()

    # update on server 
    worker_delay = epoch - worker_ts
    sum_delay += worker_delay
    if args.alpha_type == 'power':
        if args.alpha_adaptive > 0:
            alpha_factor = 1 / math.pow(worker_delay + 1.0, args.alpha_adaptive)
        else:
            alpha_factor = 1
    elif args.alpha_type == 'exp':
        alpha_factor = math.exp(-worker_delay * args.alpha_adaptive)
    elif args.alpha_type == 'sigmoid':
        # a soft-gated function in the range of (1/a, 1]
        # maximum value
        a = args.alpha_adaptive2
        # slope
        c = args.alpha_adaptive
        b = math.exp(- c * worker_delay)
        alpha_factor = (1 - b) / a + b
    elif args.alpha_type == 'hinge':
        if worker_delay <= args.alpha_adaptive2:
            alpha_factor = 1
        else:
            alpha_factor = 1.0 / ( (worker_delay - args.alpha_adaptive2) * args.alpha_adaptive + 1)
            # alpha_factor = math.exp(- (worker_delay-args.alpha_adaptive2) * args.alpha_adaptive)
    else:
        alpha_factor = 1
    alpha_scaled = alpha * alpha_factor
    for param, param_server in zip(net.collect_params().values(), params_prev_list[-1]):
        weight = param.data()
        weight[:] = param_server * (1-alpha_scaled) + weight * alpha_scaled
    # push
    params_prev_list.append([param.data().copy() for param in net.collect_params().values()])
    ts_list.append(epoch+1)
    # pop
    if len(params_prev_list) > args.max_delay:
        del params_prev_list[0]
        del ts_list[0]
    nd.waitall()
    
    # validation
    if  epoch % args.interval == 0 or epoch == args.iterations-1:
        val_L = evaluate(net, test_data, batch_size, context[0])

        logger.info('[Epoch %d] test: loss=%f, ppl=%f, grads=%f, lr=%f, time=%f' % (epoch, val_L, math.exp(val_L), n_grads, trainer.learning_rate, time.time()-tic))
        tic = time.time()
        
        nd.waitall()

        if val_L < best_val:
            best_val = val_L
        else:
            lr *= 0.25



            






