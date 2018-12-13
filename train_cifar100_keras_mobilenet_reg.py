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
from keras import regularizers
from keras.models import Model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.generic_utils import Progbar

from os import listdir
import os.path
import argparse

import pickle

def get_train_batch(train_filename):
    with open(train_filename, "rb") as f:
        B, L = pickle.load(f)

    return B.astype('float16') / 255.0, keras.utils.to_categorical(L, num_classes)

def get_test_batch(data_dir):
    test_filename = os.path.join(data_dir, "val_data.pkl")
    with open(test_filename, "rb") as f:
        B, L = pickle.load(f)

    return B.astype('float16') / 255.0, keras.utils.to_categorical(L, num_classes)

def new_regularizer(alpha, weight_matrix_0):
    return ( lambda weight_matrix: alpha * K.sum(K.squared_difference(weight_matrix, tf.convert_to_tensor( np.array(weight_matrix_0) ))) )

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, help="dir of the data", required=True)
    parser.add_argument("-b", "--batchsize", type=int, help="batchsize", default=8)
    parser.add_argument("-e", "--epochs", type=int, help="epochs", default=100)
    parser.add_argument("-v", "--interval", type=int, help="log interval", default=10)
    parser.add_argument("-n", "--nsplit", type=int, help="number of split", default=40)
    args = parser.parse_args()

    data_dir = os.path.join(args.dir, 'dataset_split_{}'.format(args.nsplit))
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    training_files = []
    for filename in sorted(listdir(train_dir)):
        absolute_filename = os.path.join(train_dir, filename)
        training_files.append(absolute_filename)

    # create model
    num_classes = 100
    target_size = 32
    input_tensor = Input(shape=(target_size, target_size, 3))
    model = MobileNetV2(alpha=0.75, weights=None, include_top=True, input_tensor=input_tensor, classes=num_classes)
    # model.compile(loss      = 'categorical_crossentropy',
    #               optimizer = RMSprop(lr=0.045, decay = 0.00004),
    #               metrics   = ['accuracy'])
    model.compile(loss      = 'categorical_crossentropy',
                  optimizer = SGD(lr=0.04, momentum=0),
                  metrics   = ['accuracy'])
    print(os.path.basename(__file__))
    model_checkpoint = ModelCheckpoint(os.path.basename(__file__) +'.hdf5', monitor='loss', save_best_only=True)
    callbacks_list = [model_checkpoint]

    batch_size = args.batchsize
    epochs = args.epochs

    print('training start', flush=True)

    [val_X, val_Y] = get_test_batch(val_dir)

    epoch_pbar = Progbar(epochs)

    for i in range(0, epochs):
        print('epoch {}/{}'.format(i+1, epochs), flush=True)

        random.shuffle(training_files)

        # subsampling
        sub_training_files = training_files[0:30]

        pbar = Progbar(len(sub_training_files))

        if i > 0:
            for layer in model.layers:
                layer.kernel_regularizer = new_regularizer(0.001, layer.get_weights()) 
            model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.04, momentum=0), metrics=['accuracy'])

        # train
        for j in range(len(sub_training_files)):
            filename = training_files[j]
            print(filename, flush=True)
            [X, Y] = get_train_batch(filename)
            model.fit(X, Y, batch_size=batch_size, epochs=1, verbose=2, callbacks=callbacks_list)
            pbar.update(j+1)

        # validation
        
        [loss, accuracy] = model.evaluate(val_X, val_Y, batch_size=1024, verbose=2)
        print("Cross entropy: %0.2f, accuracy: %0.2f" % (loss, accuracy), flush=True)
        model.save(os.path.basename(__file__) +'.hdf5')

        epoch_pbar.update(i+1)