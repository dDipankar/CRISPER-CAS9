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
			
    data_n = len(lines) - 1
    SEQ = np.zeros((data_n, 40, 4), dtype=int)
    #CA = np.zeros((data_n, 1), dtype=float)
    #Score = np.zeros((data_n, 1), dtype=float)	
    #lines = lines[1:]
    shuffle(lines)
    for l in range(0, data_n):
        data = lines[l]
        seq = data
        #Score[l-1] = float(data[6])
        #CA[l-1] = float(data[5])
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
	
    return SEQ

    

if __name__ == '__main__':
 print ("Loading train data")
 FILE = open("sequence_SFLI.txt", "r")
 data = FILE.readlines()
 print(len(data))
 SEQ_in = PREPROCESS(data)

 #score = st.zscore(score) 
 print(SEQ_in.shape)
 FILE.close()
 
 # model for seq
 SEQ = Input(shape=(40,4))
 conv_1 = Conv1D(activation="relu", padding="same", strides=1, filters=20, kernel_size=5, kernel_regularizer = l2(0.0001))(SEQ)
 bat_norm1 = BatchNormalization()(conv_1)
 pool = MaxPooling1D(pool_size=(2))(bat_norm1)
 conv_2 = Conv1D(activation="relu", padding="same", strides=1, filters=40, kernel_size=8, kernel_regularizer = l2(0.0001))(pool)
 bat_norm2 = BatchNormalization()(conv_2)
 pool_1 = AveragePooling1D(pool_size=(2))(bat_norm2)
 flatten = Flatten()(pool_1)
 dropout_1 = Dropout(0.5)(flatten)
 dense_1 = Dense(80, activation='relu', kernel_initializer='glorot_uniform')(dropout_1)
 dropout_2 = Dropout(0.5)(dense_1)
 dense_2 = Dense(units=40,  activation="relu",kernel_initializer='glorot_uniform')(dropout_2)
 dropout_3 = Dropout(0.3)(dense_2)
 dense_3 = Dense(units=40,  activation="relu",kernel_initializer='glorot_uniform')(dropout_3)
 
 out = Dense(units=1,  activation="linear")(dense_3) 
 model = Model(inputs = SEQ, outputs= out)
 model.summary()
 model.load_weights("seqonly_wtt.h5")
 pred_y = model.predict(SEQ_in)
 np.savetxt("activity_score_SFLI.csv", pred_y, delimiter= ",")
 
