# pretrain the auto-encoder model

from collections import OrderedDict
import os
import sys
import warnings

import argparse
import logging
import h5py as h5
import numpy as np
import pandas as pd
import scipy.io

import six
from six.moves import range
import matplotlib.pyplot as plt
#from dna import *

from sklearn.metrics import roc_auc_score, confusion_matrix
from keras.preprocessing import sequence
from keras.optimizers import RMSprop,Adam, SGD
from keras.models import Sequential, Model
from keras.layers.core import  Dropout, Activation, Flatten
from keras.regularizers import l1,l2,l1_l2
from keras.constraints import maxnorm
#from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, MaxPooling1D, Dense, LSTM, Bidirectional, BatchNormalization, MaxPooling2D, AveragePooling1D, Input, Multiply, Add, UpSampling1D
from sklearn.metrics import mean_squared_error as mse
import scipy.stats as st
#from keras.utils import plot_model
#from keras.utils.layer_utils import print_layer_shapes
# fix random seed for reproducibility
from random import shuffle
np.random.seed(1369)


def PREPROCESS(lines):
			
    data_n = len(lines)
    SEQ = np.zeros((data_n, 40, 4), dtype=int)
    for l in range(0, data_n):
        data = lines[l].split('\n')
        seq = data[0]
        for i in range(40):
            if seq[i] in "Aa":
                SEQ[l-1, i, 0] = 1
            elif seq[i] in "Cc":
                SEQ[l-1, i, 1] = 1
            elif seq[i] in "Gg":
                SEQ[l-1, i, 2] = 1
            elif seq[i] in "Tt":
                SEQ[l-1, i, 3] = 1
    return SEQ
    

if __name__ == '__main__':
        print('a')
        print ("Loading train data")
        FILE = open("sequences.txt", "r")
        data = FILE.readlines()
        print(len(data))
        shuffle(data[0:])
        SEQ = PREPROCESS(data)
        #SEQ = SEQ[0:10000000]
        print(SEQ.shape)
        print(len(SEQ))
        FILE.close()
        trainsize = int(0.7* len(SEQ))
        #valsize = trainsize + int(0.1*len(data))   
        train_x = SEQ[0:trainsize,:]
        #train_nu = CA[0:trainsize]
        #train_y = score[0:trainsize]
        print(train_x.shape)
        #print(train_y.shape)
        #print(train_nu.shape)
        
        val_x = SEQ[trainsize:,:]
        #val_y = score[trainsize:valsize]
        #val_nu = CA[trainsize:valsize]
        print(val_x.shape)
        #print(val_y.shape)
        #print(val_nu.shape)
        #test_x = SEQ[valsize:,:]
        #test_y = score[valsize:]
        #test_nu = CA[valsize:]
        #print(test_x.shape)
        #print(test_y.shape)
        #print(val_nu.shape)
        
        # model for seq
        SEQ = Input(shape=(40,4))
        conv_1 = Conv1D(activation="relu", padding="same", strides=1, filters=20, kernel_size=5, kernel_initializer='glorot_uniform',kernel_regularizer = l2(0.0001))(SEQ)
        bat_norm1 = BatchNormalization()(conv_1)
        pool = MaxPooling1D(pool_size=(2))(bat_norm1)
        conv_2 = Conv1D(activation="relu", padding="same", strides=1, filters=40, kernel_size=8, kernel_initializer='glorot_uniform',kernel_regularizer = l2(0.0001))(pool)
        bat_norm2 = BatchNormalization()(conv_2)
        pool_1 = AveragePooling1D(pool_size=(2))(bat_norm2)
        enc = pool_1
        dec_pool_1 =  UpSampling1D(size=2)(enc)
        dec_bat_norm2 = BatchNormalization()(dec_pool_1)
        dec_conv_2  = Conv1D(activation="relu", padding="same", strides=1, filters=40, kernel_size=8, kernel_initializer='glorot_uniform',kernel_regularizer = l2(0.0001))(dec_bat_norm2)
        dec_pool = UpSampling1D(size=2)(dec_conv_2)
        dec_conv_1 = Conv1D(activation="relu", padding="same", strides=1, filters=20, kernel_size=5, kernel_initializer='glorot_uniform',kernel_regularizer = l2(0.0001))(dec_pool)
        dec = Conv1D(activation="relu", padding="same", strides=1, filters=4, kernel_size=5, kernel_initializer='glorot_uniform',kernel_regularizer = l2(0.0001))(dec_pool)
        model = Model(inputs = SEQ, outputs= dec)
        model.summary()
        adam = Adam(lr = 0.001)
        model.compile(loss='binary_crossentropy', optimizer=adam)
        checkpointer = ModelCheckpoint(filepath="autoenc_epoch.h5", verbose=1, monitor='val_loss',save_best_only=True)
        history = model.fit(train_x, train_x, batch_size=64, epochs=6, shuffle=True, validation_data=(val_x, val_x), callbacks=[checkpointer])
        model.save('auto_encode.h5')
	model.save_weights("auto_encode_wtt.h5")
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()    
