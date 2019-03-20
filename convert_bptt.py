import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os
import sys
import math
import time
import random
import datetime
import pickle

from os import listdir
import os.path
import argparse

import glob
import math

import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon.utils import download

import gluonnlp as nlp

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nsplit", type=int, help="number of partitions of training set", default=100)
    parser.add_argument("--nsplitv", type=int, help="number of partitions of validation set", default=10)
    parser.add_argument("-o", "--output", type=str, help="dir of output", required=True)
    args = parser.parse_args()

    output_dir = os.path.join(args.output, 'dataset_split_{}'.format(args.nsplit))
    output_train_dir = os.path.join(output_dir, 'train')
    output_val_dir = os.path.join(output_dir, 'val')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_train_dir):
        os.makedirs(output_train_dir)
    if not os.path.exists(output_val_dir):
        os.makedirs(output_val_dir)

    context = [mx.cpu()]
    batch_size = 20
    bptt = 35

    dataset_name = 'wikitext-2'
    train_dataset, val_dataset, test_dataset = [
        nlp.data.WikiText2(
            segment=segment, bos=None, eos='<eos>', skip_empty=False)
        for segment in ['train', 'val', 'test']
    ]

    vocab = nlp.Vocab(
        nlp.data.Counter(train_dataset), padding_token=None, bos_token=None)

    bptt_batchify = nlp.data.batchify.CorpusBPTTBatchify(
        vocab, bptt, batch_size, last_batch='discard')
    train_data, val_data, test_data = [
        bptt_batchify(x) for x in [train_dataset, val_dataset, test_dataset]
    ]

    x_train_list = []
    y_train_list = []
    for i, (data, target) in enumerate(train_data):
        x_train_list.append(data)
        y_train_list.append(target)
    for i, (data, target) in enumerate(val_data):
        x_train_list.append(data)
        y_train_list.append(target)
    x_test_list = []
    y_test_list = []
    for i, (data, target) in enumerate(test_data):
        x_test_list.append(data)
        y_test_list.append(target)

    x_train = nd.concat(*x_train_list, dim=0)
    y_train = nd.concat(*y_train_list, dim=0)
    x_test = nd.concat(*x_test_list, dim=0)
    y_test = nd.concat(*y_test_list, dim=0)

    nd.waitall()

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    batch_size = int(math.floor(x_train.shape[0] / bptt / args.nsplit)) * bptt

    nice_n_train = math.ceil(x_train.shape[0] / batch_size) * batch_size

    print('n_train: {}, n_val: {}, partition_size: {}, n_train_part: {}'.format(x_train.shape[0], x_test.shape[0], batch_size, nice_n_train // batch_size), flush=True)

    # perm = list(range( x_train.shape[0] ))
    # random.shuffle(perm)
    
    # x_train = np.array([x_train[index, :] for index in perm])
    # y_train = np.array([y_train[index, :] for index in perm])

    for i in range(int(nice_n_train // batch_size)):
        output_train_filename = os.path.join(output_train_dir, "train_data_%03d.pkl" % i)
        i_start = i * batch_size
        i_end = i_start + batch_size
        if i_end > x_train.shape[0]:
            i_end = x_train.shape[0]
        b = x_train[i_start:i_end, :]
        l = y_train[i_start:i_end, :]
        print(b.shape)
        print(l.shape)
        print(output_train_filename, flush=True)
        with open(output_train_filename, "wb") as f:
            pickle.dump([b, l], f)

    if args.nsplitv > 1:
        batch_size_v = x_test.shape[0] // bptt // args.nsplitv * bptt
        for i in range(args.nsplitv):
            output_test_filename = os.path.join(output_val_dir, "val_data_%03d.pkl" % i)
            i_start = i * batch_size_v
            i_end = i_start + batch_size_v
            if i_end > x_test.shape[0]:
                i_end = x_test.shape[0]
            b = x_test[i_start:i_end, :]
            l = y_test[i_start:i_end, :]
            print(b.shape)
            print(l.shape)
            print(output_test_filename, flush=True)
            with open(output_test_filename, "wb") as f:
                pickle.dump([b, l], f)
    else:
        output_test_filename = os.path.join(output_val_dir, "val_data.pkl")
        print(x_test.shape)
        print(y_test.shape)
        print(output_test_filename, flush=True)
        with open(output_test_filename, "wb") as f:
            pickle.dump([x_test, y_test], f)