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
from keras.datasets import cifar10
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
    parser.add_argument("-o", "--output", type=str, help="dir of output", required=True)
    parser.add_argument("--normalize", type=int, help="normalized or not", default=0)
    args = parser.parse_args()

    output_dir = args.output
    output_train_dir = os.path.join(output_dir, 'train')
    output_val_dir = os.path.join(output_dir, 'val')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_train_dir):
        os.makedirs(output_train_dir)
    if not os.path.exists(output_val_dir):
        os.makedirs(output_val_dir)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # crop
    x_train = x_train[:, 4:28, 4:28, :]
    x_test = x_test[:, 4:28, 4:28, :]

    # Normalize data.
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # If subtract pixel mean is enabled
    if args.normalize == 1:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
        x_train /= 128.
        x_test /= 128.

    output_train_filename = os.path.join(output_train_dir, "train_data.pkl" )
    print(x_train.shape)
    print(y_train.shape)
    print(output_train_filename, flush=True)
    with open(output_train_filename, "wb") as f:
        pickle.dump([x_train, y_train], f)

    output_test_filename = os.path.join(output_val_dir, "val_data.pkl")
    print(x_test.shape)
    print(y_test.shape)
    print(output_test_filename, flush=True)
    with open(output_test_filename, "wb") as f:
        pickle.dump([x_test, y_test], f)