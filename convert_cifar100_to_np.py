import numpy as np
import os
import sys
import math
import time
import random
import datetime
import pickle

from keras.utils.generic_utils import Progbar
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

import keras
from keras import backend as K
from keras.datasets import cifar100
from keras.utils import np_utils, plot_model


from os import listdir
import os.path
import argparse

# config =tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0",allow_growth=True))
# config = tf.ConfigProto()
# config.gpu_options.allocator_type ='BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.90
# tf.Session(config=config)

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nsplit", type=int, help="number of partitions", default=50)
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

    # (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')

    batch_size = int(math.floor(x_train.shape[0] / args.nsplit))

    nice_n_train = np.ceil(x_train.shape[0] / batch_size) * batch_size

    print('n_train: {}, n_val: {}, partition_size: {}, n_train_part: {}'.format(x_train.shape[0], x_test.shape[0], batch_size, nice_n_train // batch_size), flush=True)

    perm = list(range( x_train.shape[0] ))
    random.shuffle(perm)
    
    x_train = np.array([x_train[index, :, :, :] for index in perm])
    y_train = np.array([y_train[index, :] for index in perm])

    for i in range(int(nice_n_train // batch_size)):
        output_train_filename = os.path.join(output_train_dir, "train_data_%03d.pkl" % i)
        i_start = i * batch_size
        i_end = i_start + batch_size
        if i_end > x_train.shape[0]:
            i_end = x_train.shape[0]
        b = x_train[i_start:i_end, :, :, :]
        l = y_train[i_start:i_end, :]
        print(b.shape)
        print(l.shape)
        print(output_train_filename, flush=True)
        with open(output_train_filename, "wb") as f:
            pickle.dump([b, l], f)

    output_test_filename = os.path.join(output_val_dir, "val_data.pkl")
    print(x_test.shape)
    print(y_test.shape)
    print(output_test_filename, flush=True)
    with open(output_test_filename, "wb") as f:
        pickle.dump([x_test, y_test], f)