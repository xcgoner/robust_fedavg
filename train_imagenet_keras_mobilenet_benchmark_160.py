import numpy as np
import os
import sys
import math
import time
import random
import datetime

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
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.datasets import mnist,cifar10
from keras.utils import np_utils
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.generic_utils import Progbar

from os import listdir
import os.path
import argparse

import pickle

# config =tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0",allow_growth=True))
# config = tf.ConfigProto()
# config.gpu_options.allocator_type ='BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.90
# tf.Session(config=config)

def get_train_batch(train_filename):
    with open(train_filename, "rb") as f:
        B, L = pickle.load(f)

    return B.astype('float16') / 255.0, keras.utils.to_categorical(L, num_classes)

def get_test_batch(data_dir):
    test_filename = os.path.join(data_dir, "val_data.pkl")
    with open(test_filename, "rb") as f:
        B, L = pickle.load(f)

    return B.astype('float16') / 255.0, keras.utils.to_categorical(L, num_classes)

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, help="dir of the data", required=True)
    parser.add_argument("-b", "--batchsize", type=int, help="batchsize", default=128)
    parser.add_argument("-e", "--epochs", type=int, help="epochs", default=100)
    parser.add_argument("-v", "--interval", type=int, help="log interval", default=10)
    parser.add_argument("-s", "--size", type=int, help="target size", default=224)
    args = parser.parse_args()

    target_size = args.size

    data_dir = os.path.join(args.dir, 'dataset_{}'.format(target_size))
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    training_files = []
    for filename in sorted(listdir(train_dir)):
        absolute_filename = os.path.join(train_dir, filename)
        training_files.append(absolute_filename)

    # create model
    input_tensor = Input(shape=(target_size, target_size, 3))
    model = MobileNetV2(alpha=0.75, weights=None, include_top=True, input_tensor=input_tensor)
    model.compile(loss      = 'categorical_crossentropy',
                  optimizer = RMSprop(lr=0.045, decay = 0.00004),
                  metrics   = ['accuracy'])
    print(os.path.basename(__file__))
    model_checkpoint = ModelCheckpoint(os.path.basename(__file__) +'.hdf5', monitor='loss', save_best_only=True)
    callbacks_list = [model_checkpoint]

    batch_size = args.batchsize
    epochs = args.epochs
    num_classes = 1000

    print('training start')

    for i in range(0, epochs):
        print('epoch {}/{}'.format(i+1, epochs))

        random.shuffle(training_files)

        # subsampling
        sub_training_files = training_files[0:10]

        pbar = Progbar(len(sub_training_files))

        # train
        for j in range(len(sub_training_files)):
            filename = training_files[j]
            print(filename)
            [X, Y] = get_train_batch(filename)
            model.fit(X, Y, batch_size=batch_size, epochs=1, verbose=2, callbacks=callbacks_list)
            pbar.update(j+1)

        # validation
        [X, Y] = get_test_batch(val_dir)
        [loss, accuracy] = model.evaluate(X, Y, batch_size=batch_size, verbose=1)
        print("Cross entropy: %0.2f, accuracy: %0.2f" % (loss, accuracy))
        model.save(os.path.basename(__file__) +'.hdf5')