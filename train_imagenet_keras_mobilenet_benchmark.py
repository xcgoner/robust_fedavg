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

    B = np.zeros(shape=(batch_size, 160, 160, 3))
    L = np.zeros(shape=(batch_size))

    # print(batch_size)
    # print(len(training_images))

    while index < batch_size and current_index < len(training_images):
        try:
            # print(training_images[current_index])
            img = load_img(training_images[current_index], target_size=(160, 160))
            B[index] = img_to_array(img, dtype='float16') / 255.0
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

    return B, keras.utils.to_categorical(L, num_classes)

def get_test_batch():
    index = 0

    global current_index_test

    B = np.zeros(shape=(batch_size, 160, 160, 3))
    L = np.zeros(shape=(batch_size))

    while index < batch_size and current_index_test < len(testing_images):
        try:
            img = load_img(testing_images[current_index_test], target_size=(160, 160))
            B[index] = img_to_array(img, dtype='float16') / 255.0
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

    return B, keras.utils.to_categorical(L, num_classes)

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, help="dir of the data", required=True)
    parser.add_argument("-b", "--batchsize", type=int, help="batchsize", default=128)
    parser.add_argument("-e", "--epochs", type=int, help="epochs", default=100)
    parser.add_argument("-v", "--interval", type=int, help="log interval", default=10)
    args = parser.parse_args()

    train_dir = os.path.join(args.dir, 'train')
    val_dir = os.path.join(args.dir, 'val')

    print(train_dir)

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
            # print(absolute_filename)
            training_images.append(absolute_filename)
            training_labels.append(category_dict[category])

    for category in category_list:
        category_dir = os.path.join(val_dir, category)
        for filename in sorted(listdir(category_dir)):
            absolute_filename = os.path.join(category_dir, filename)
            testing_images.append(absolute_filename)
            testing_labels.append(category_dict[category])

    # create model
    input_tensor = Input(shape=(160, 160, 3))
    model = MobileNetV2(alpha=0.75, weights=None, include_top=True, input_tensor=input_tensor)
    model.compile(loss      = 'categorical_crossentropy',
                  optimizer = RMSprop(lr=0.045, momentum=0.9, decay = 0.00004),
                  metrics   = ['accuracy'])

    batch_size = args.batchsize
    epochs = args.epochs
    num_classes = 1000

    # # create log directory
    # TIME = datetime.datetime.today().strftime("%Y%m%d_%H:%M")
    # log_path = './log_2/' + TIME + '/'
    # if not os.path.exists(log_path):
    #     os.makedirs(log_path)

    nice_n = np.ceil(len(training_images) / batch_size) * batch_size

    perm = list(range(len(training_images)))

    print('training start')

    for i in range(0, epochs):
        print('epoch {}/{}'.format(i+1, epochs))
        current_index = 0
        current_index_test = 0
        random.shuffle(perm)
        training_images = [training_images[index] for index in perm]
        training_labels = [training_labels[index] for index in perm]
        batch_no = 0
        while batch_no < int(nice_n / batch_size):
            start_time = time.time()
            [b, l] = get_train_batch()
            # print('data loaded')
            # print(b.shape)
            # print(l.shape)
            [loss, accuracy] = model.train_on_batch(b, l)
            end_time = time.time()
            print('Train: batch {}/{} loss: {} accuracy: {} time: {}ms'.format(int(current_index / batch_size), int(nice_n / batch_size), loss, accuracy, 1000 * (end_time - start_time)), flush=True)
            # if batch_no % args.interval == 0:
            batch_no += 1
        
        # b_t, l_t = get_test_batch()
        # loss, accuracy = model.test_on_batch(b_t, l_t)
        # print('Val: accuracy: {}'.format(accuracy), flush=True)

        # if i % 10 == 0:
        #     vgg16.save("./save/vgg16_model_2.e{epoch:02d}-l{loss:.2f}-a{acc:.2f}.hdf5".format(**{"epoch":i,"loss":loss, "acc":accuracy}))