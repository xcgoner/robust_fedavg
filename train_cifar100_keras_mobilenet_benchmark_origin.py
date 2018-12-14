import numpy as np
import os
import sys
import math
import time
import random
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

import keras
from keras import backend as K
from keras.layers import Input
from keras.models import Model,Sequential
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.datasets import cifar100
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.generic_utils import Progbar

import argparse


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batchsize", type=int, help="batchsize", default=8)
    parser.add_argument("-e", "--epochs", type=int, help="epochs", default=100)
    parser.add_argument("-v", "--interval", type=int, help="log interval", default=10)
    args = parser.parse_args()

    # create model
    # num_classes = 100
    num_classes = 20
    target_size = 32
    input_tensor = Input(shape=(target_size, target_size, 3))
    model = MobileNetV2(alpha=0.25, weights=None, include_top=True, input_tensor=input_tensor, classes=num_classes)
    # model.compile(loss      = 'categorical_crossentropy',
    #               optimizer = RMSprop(lr=0.045, decay = 0.00004),
    #               metrics   = ['accuracy'])
    model.compile(loss      = 'categorical_crossentropy',
                  optimizer = SGD(lr=0.01, momentum=0),
                  metrics   = ['accuracy'])
    print(os.path.basename(__file__))
    model_checkpoint = ModelCheckpoint(os.path.basename(__file__) +'.hdf5', monitor='loss', save_best_only=True)
    callbacks_list = [model_checkpoint]

    batch_size = args.batchsize
    epochs = args.epochs

    print('training start')

    # (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    for i in range(0, epochs):

        # train
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=2, callbacks=callbacks_list)

        # validation
        [loss, accuracy] = model.evaluate(x_test, y_test, batch_size=1024, verbose=1)
        print("Cross entropy: %0.2f, accuracy: %0.2f" % (loss, accuracy), flush=True)
        # model.save(os.path.basename(__file__) +'.hdf5')