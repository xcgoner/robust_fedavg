import numpy as np
import os
import sys
import math
import time
import random
import datetime
import pickle

# from progressbar import *
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

import keras
from keras import backend as K
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Model,Sequential
from keras.datasets import mnist,cifar10
from keras.utils import np_utils, plot_model
from keras.applications.mobilenet import MobileNet
from keras.datasets import mnist,cifar10
from keras.utils import np_utils
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from os import listdir
import os.path
import argparse

# config =tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0",allow_growth=True))
# config = tf.ConfigProto()
# config.gpu_options.allocator_type ='BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.90
# tf.Session(config=config)

def get_train_batch():
    index = 0

    global current_index

    B = np.zeros(shape=(batch_size, target_size, target_size, 3), dtype='uint8')
    L = np.zeros(shape=(batch_size))

    # print(batch_size)
    # print(len(training_images))

    while index < batch_size and current_index < len(training_images):
        try:
            # print(training_images[current_index])
            img = load_img(training_images[current_index], target_size=(target_size, target_size))
            B[index] = img_to_array(img, dtype='uint8')
            del(img)

            L[index] = training_labels[current_index]

            index = index + 1
            current_index = current_index + 1
        except:
            print("index {} :Ignore image {}".format(index, training_images[current_index]))
            current_index = current_index + 1
    
    if index < batch_size:
        print("batch size smaller than expected")
        B = B[0:index, :, :, :]
        L = L[0:index]

    return B, L

def get_test_batch():
    index = 0

    global current_index_test

    B = np.zeros(shape=(batch_size, target_size, target_size, 3), dtype='uint8')
    L = np.zeros(shape=(batch_size))

    while index < batch_size and current_index_test < len(testing_images):
        try:
            img = load_img(testing_images[current_index_test], target_size=(target_size, target_size))
            B[index] = img_to_array(img, dtype='uint8')
            del(img)

            L[index] = testing_labels[current_index_test]

            index = index + 1
            current_index_test = current_index_test + 1
        except:
            # print("Ignore image {}".format(testing_images[current_index_test]))
            current_index_test = current_index_test + 1

    if index < batch_size:
        B = B[0:index, :, :, :]
        L = L[0:index]

    return B, L

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--input", type=str, help="dir of the data", required=True)
    parser.add_argument("-n", "--nsplit", type=int, help="number of partitions", default=400)
    parser.add_argument("-s", "--size", type=int, help="target size", default=224)
    parser.add_argument("-o", "--output", type=str, help="dir of output", required=True)
    args = parser.parse_args()

    train_dir = os.path.join(args.input, 'train')
    val_dir = os.path.join(args.input, 'val')

    print(train_dir)

    target_size = int(args.size)

    output_dir = os.path.join(args.output, 'dataset_{}'.format(target_size))
    output_train_dir = os.path.join(output_dir, 'train')
    output_val_dir = os.path.join(output_dir, 'val')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_train_dir):
        os.makedirs(output_train_dir)
    if not os.path.exists(output_val_dir):
        os.makedirs(output_val_dir)

    category_list = sorted(listdir(train_dir))
    category_dict =  dict((category_list[i], i) for i in range(len(category_list)))

    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []

    for category in category_list:
        category_dir = os.path.join(train_dir, category)
        for filename in sorted(listdir(category_dir)):
            absolute_filename = os.path.join(category_dir, filename)
            training_images.append(absolute_filename)
            training_labels.append(category_dict[category])

    for category in category_list:
        category_dir = os.path.join(val_dir, category)
        for filename in sorted(listdir(category_dir)):
            absolute_filename = os.path.join(category_dir, filename)
            testing_images.append(absolute_filename)
            testing_labels.append(category_dict[category])

    batch_size = int(math.floor(len(training_images) / args.nsplit))
    
    num_classes = 1000

    nice_n_train = np.ceil(len(training_images) / batch_size) * batch_size
    nice_n_val = np.ceil(len(testing_images) / batch_size) * batch_size

    print('n_train: {}, n_val: {}, partition_size: {}, n_train_part: {}, n_val_part: {}'.format(len(training_images), len(testing_images), batch_size, nice_n_train // batch_size, nice_n_val // batch_size), flush=True)

    perm = list(range(len(training_images)))

    batch_no = 0
    current_index = 0
    while batch_no < int(nice_n_train / batch_size):
        [b, l] = get_train_batch()
        output_train_filename = os.path.join(output_train_dir, "train_data_%03d.pkl" % batch_no)
        print(b.shape)
        print(l.shape)
        print(output_train_filename, flush=True)
        with open(output_train_filename, "wb") as f:
            pickle.dump([b, l], f)
        batch_no = batch_no + 1