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
			
    data_n = len(lines) - 1
    SEQ = np.zeros((data_n, 40, 4), dtype=int)
    CA = np.zeros((data_n, 1), dtype=float)
    Score = np.zeros((data_n, 1), dtype=float)	
    
    for l in range(1, data_n+1):
        data = lines[l].split(',')
        seq = data[2]
        Score[l-1] = float(data[6])
        CA[l-1] = float(data[5])
        for i in range(40):
            if seq[i] in "Aa":
                SEQ[l-1, i, 0] = 1
            elif seq[i] in "Cc":
                SEQ[l-1, i, 1] = 1
            elif seq[i] in "Gg":
                SEQ[l-1, i, 2] = 1
            elif seq[i] in "Tt":
                SEQ[l-1, i, 3] = 1
        #CA[l-1,0] = int(data[2])*100
	
    return SEQ, CA, Score

if __name__ == '__main__':
        print('a')
        print ("Loading train data")
        FILE = open("data_yeast_grp3.csv", "r")
        data = FILE.readlines()
        print(len(data))
        shuffle(data[1:])
        SEQ, CA, score = PREPROCESS(data)
        score = np.dot(score,-1)
        score = st.zscore(score) 
        print(SEQ.shape)
        #SEQ = np.transpose(np.array(SEQ),axes=(0,2,1))
        #print(SEQ.shape)
        FILE.close()
        trainsize = int(0.6* len(data))
        valsize = trainsize + int(0.2*len(data))   
        train_x = SEQ[0:trainsize,:]
        train_nu = CA[0:trainsize]
        train_y = score[0:trainsize]
        print(train_x.shape)
        print(train_y.shape)
        print(train_nu.shape)
        
        val_x = SEQ[trainsize:valsize,:]
        val_y = score[trainsize:valsize]
        val_nu = CA[trainsize:valsize]
        print(val_x.shape)
        print(val_y.shape)
        print(val_nu.shape)
        test_x = SEQ[valsize:,:]
        test_y = score[valsize:]
        test_nu = CA[valsize:]
        print(test_x.shape)
        print(test_y.shape)
        print(val_nu.shape)
        
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
        checkpointer = ModelCheckpoint(filepath="autoenc.h5", verbose=1, monitor='val_loss',save_best_only=True)
        model.fit(train_x, train_x, batch_size=64, epochs=250, shuffle=True, validation_data=(val_x, val_x), callbacks=[checkpointer])
        #model.save(auto_enc.h5)
		
