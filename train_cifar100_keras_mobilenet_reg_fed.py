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

from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

print('rank: %d' % (mpi_rank), flush=True)

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
    parser.add_argument("-l", "--lr", type=float, help="learning rate", default=0.001)
    parser.add_argument("-r", "--regularization", type=float, help="weight of regularization", default=0.00001)
    args = parser.parse_args()

    print(args, flush=True)

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
    model = MobileNetV2(alpha=0.25, weights=None, include_top=True, input_tensor=input_tensor, classes=num_classes)
    # model.compile(loss      = 'categorical_crossentropy',
    #               optimizer = RMSprop(lr=0.045, decay = 0.00004),
    #               metrics   = ['accuracy'])
    model.compile(loss      = 'categorical_crossentropy',
                  optimizer = SGD(lr=args.lr/10, momentum=0),
                  metrics   = ['accuracy'])
    print(os.path.basename(__file__), flush=True)
    # model_checkpoint = ModelCheckpoint(os.path.basename(__file__) +'.hdf5', monitor='loss', save_best_only=True)
    # callbacks_list = [model_checkpoint]

    batch_size = args.batchsize
    epochs = args.epochs

    # partition dataset
    if mpi_rank == 0:
        training_file_index = [i for i in range(len(training_files))]
        random.shuffle(training_file_index)
        chunk_size = int(math.floor(len(training_files) / mpi_size))
        training_file_index_list = [training_file_index[i * chunk_size:(i + 1) * chunk_size] for i in range((len(training_file_index) + chunk_size - 1) // chunk_size )] 
    else:
        training_file_index_list = None
    training_file_index_list = mpi_comm.bcast(training_file_index_list, root=0)

    # print('rank: {}, index_list: {}'.format((mpi_rank, training_file_index_list)), flush=True)
    # print(training_file_index_list, flush=True)

    sub_training_files = [training_files[i] for i in training_file_index_list[mpi_rank]]

    # print('rank: {}, file_list: {}'.format((mpi_rank, sub_training_files)), flush=True)
    # print(sub_training_files, flush=True)

    print('warm up', flush=True)

    if mpi_rank == 0:
        [X, Y] = get_train_batch(sub_training_files[0])
        model.fit(X, Y, batch_size=batch_size, epochs=1, verbose=2)
    
    for layer in model.layers:
        if mpi_rank == 0:
            layer_weights = layer.get_weights()
        else:
            layer_weights = None
        layer_weights = mpi_comm.bcast(layer_weights, root=0)
        layer.set_weights(layer_weights)

    print('training start', flush=True)

    if mpi_rank == 0:
        [val_X, val_Y] = get_test_batch(val_dir)

    epoch_pbar = Progbar(epochs)

    for i in range(0, epochs):
        if mpi_rank == 0:
            print('epoch {}/{}'.format(i+1, epochs), flush=True)

        random.shuffle(sub_training_files)

        pbar = Progbar(len(sub_training_files))

        # for layer in model.layers:
        #     layer.kernel_regularizer = new_regularizer(args.regularization, layer.get_weights()) 
        # model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=args.lr, momentum=0), metrics=['accuracy'])

        # train
        # for filename in sub_training_files:
        #     [X, Y] = get_train_batch(filename)
        #     # model.fit(X, Y, batch_size=batch_size, epochs=1, verbose=2)
        #     model.fit(X, Y, batch_size=batch_size, epochs=1, verbose=0)
        [X, Y] = get_train_batch(random.choice(sub_training_files))
        model.fit(X, Y, batch_size=batch_size, epochs=1, verbose=0)

        for layer in model.layers:
            layer_weights = layer.get_weights()
            layer_weights_gathered = mpi_comm.gather(layer_weights, root=0)
            if mpi_rank == 0:
                for j in range(len(layer_weights)):
                    # aggregate
                    layer_weights[j] = np.mean( np.array( [weights[j] for weights in layer_weights_gathered] ) , axis=0)
            else:
                layer_weights = None
            layer_weights = mpi_comm.bcast(layer_weights, root=0)
            layer.set_weights(layer_weights)

        if mpi_rank == 0:
            # validation
            [loss, accuracy] = model.evaluate(val_X, val_Y, batch_size=1024, verbose=2)
            print("Cross entropy: %0.2f, accuracy: %0.2f" % (loss, accuracy), flush=True)
            model.save(os.path.basename(__file__) +'.hdf5')
            epoch_pbar.update(i+1)

    print('rank {}: done'.format((mpi_rank)), flush=True)