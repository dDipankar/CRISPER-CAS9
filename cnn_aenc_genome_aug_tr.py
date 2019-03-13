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
from keras.models import load_model
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

def PREPROCESS_aug(lines):
			
    data_n = len(lines) - 1
    SEQ = np.zeros((data_n, 40, 4), dtype=int)
    CA = np.zeros((data_n, 1), dtype=float)
    Score = np.zeros((data_n, 1), dtype=float)	
    
    for l in range(1, data_n+1):
        data = lines[l].split(',')
        seq = data[0]
        Score[l-1] = float(data[2])
        CA[l-1] = float(data[3])
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
 
 print ("Loading train data")
 FILE = open("aug_data_yeast.csv", "r")
 data = FILE.readlines()
 print(len(data))
 SEQ, CA, score = PREPROCESS_aug(data)
 score = np.dot(score,-1)
 score = st.zscore(score) 
 FILE.close()
 print(SEQ.shape)
 print(score.shape)
 print(CA.shape)
 score = np.dot(score,-1)
 score = st.zscore(score) 
 trainsize = int(0.95* len(data))
 train_x = SEQ[0:trainsize,:]
 train_nu = CA[0:trainsize]
 train_y = score[0:trainsize]
 print(train_x.shape)
 print(train_y.shape)
 print(train_nu.shape)
 valsize = trainsize + int(0.03*len(data))
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
 print(test_nu.shape)
 print('loading model')		
 basemodel = load_model('auto_encode.h5')
 #basemodel.summary()
 basemodel.layers.pop()
 basemodel.layers.pop()
 basemodel.layers.pop()
 basemodel.layers.pop()
 basemodel.layers.pop()
 #basemodel.summary()
 #print(basemodel.layers)
 
 for layer in basemodel.layers:
	 layer.trainable = True 
 #model = basemodel.output
 flatten = Flatten()(basemodel.layers[-1].output)
 dropout_1 = Dropout(0.5)(flatten)
 dense_1 = Dense(80, activation='relu', kernel_initializer='glorot_uniform')(dropout_1)
 dropout_2 = Dropout(0.5)(dense_1)
 dense_2 = Dense(units=40,  activation="relu",kernel_initializer='glorot_uniform')(dropout_2)
 dropout_3 = Dropout(0.3)(dense_2)
 dense_3 = Dense(units=40,  activation="relu",kernel_initializer='glorot_uniform')(dropout_3)
 out = Dense(units=1,  activation="linear")(dense_3)
 model_seq = Model(inputs = basemodel.layers[0].output, output = out) 
 model_seq.summary()
 #model for epigenetics feature
 #NU = Input(shape=(1,))
 #dense1_nu = Dense(units=40,  activation="relu",kernel_initializer='glorot_uniform')(NU)
 #mult = Multiply()([dense_3, dense1_nu])
 
 adam = Adam(lr = 0.001)
 model_seq.compile(loss='mean_squared_error', optimizer=adam)
 checkpointer = ModelCheckpoint(filepath="cas9_seq.hdf5",verbose=1, monitor='val_loss',save_best_only=True)
 earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
 model_seq.fit(train_x, train_y, batch_size=64, epochs=150, shuffle=True, validation_data=( val_x, val_y), callbacks=[checkpointer,earlystopper])
 pred_y = model_seq.predict(test_x)
 print('testset')
 print('mse ' + str(mse(test_y, pred_y)))
 print(st.spearmanr(test_y, pred_y))
 y_pred_tr = model_seq.predict(train_x)
 print(st.spearmanr(train_y, y_pred_tr))
 np.savetxt("train.csv", train_y, delimiter= ",")
 np.savetxt("trainpred.csv", y_pred_tr, delimiter = ",")
 np.savetxt("test.csv" , test_y, delimiter = ",")
 np.savetxt("testpred.csv", pred_y, delimiter = ",")	
