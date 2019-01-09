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
from random import shuffle
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
from keras.layers import Conv1D, MaxPooling1D, Dense, LSTM, Bidirectional, BatchNormalization, MaxPooling2D, AveragePooling1D, Input, Multiply, Add, Concatenate, Dot
from sklearn.metrics import mean_squared_error as mse
import scipy.stats as st
#from keras.utils import plot_model
#from keras.utils.layer_utils import print_layer_shapes
# fix random seed for reproducibility
np.random.seed(1369)

def PREPROCESS(lines):
			
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
        FILE.close()
        shuffle(data[1:])
        #data = data[0:10]
        #print(len(data))
        #print(data[0])
   
        SEQ, CA, score = PREPROCESS(data)
        
     
        #score = np.dot(score,-1)
        #score = st.zscore(score) 
        print(SEQ.shape)
        print(score.shape)
        print(CA.shape)
        #SEQ = np.transpose(np.array(SEQ),axes=(0,2,1))
        #print(SEQ.shape)
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
       
        # model for seq
        SEQ = Input(shape=(40,4))
        conv_1 = Conv1D(activation="relu", padding="valid", strides=1, filters=60, kernel_size=5, kernel_initializer='glorot_uniform')(SEQ)
        bat_norm1 = BatchNormalization()(conv_1)
        pool = MaxPooling1D(pool_size=(2))(bat_norm1)
        conv_2 = Conv1D(activation="relu", padding="valid", strides=1, filters=80, kernel_size=8, kernel_initializer='glorot_uniform')(pool)
        bat_norm2 = BatchNormalization()(conv_2)
        pool_1 = AveragePooling1D(pool_size=(2))(bat_norm2)
        flatten = Flatten()(pool_1)
        dropout_1 = Dropout(0.7)(flatten)
        dense_1 = Dense(150, activation='relu', kernel_initializer='glorot_uniform')(dropout_1)
        dropout_2 = Dropout(0.7)(dense_1)
        dense_2 = Dense(units=75,  activation="relu",kernel_initializer='glorot_uniform')(dropout_2)
        dropout_3 = Dropout(0.3)(dense_2)
        dense_3 = Dense(units=60,  activation="relu",kernel_initializer='glorot_uniform')(dropout_3)
        
        #model for epigenetics feature
        NU = Input(shape=(1,))
        dense1_nu = Dense(units=60,  activation="relu",kernel_initializer='glorot_uniform')(NU)
        mult = Multiply()([dense_3, dense1_nu])
        out = Dense(units=1,  activation="linear")(mult)
        
        
        #dense_out = Dense(units=1,  activation="linear")(dense_3)
        model = Model(inputs = [SEQ,NU], outputs= out)
        model.summary()
        
        #adam = SGD(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
        #adam = SGD(lr=0.01, momentum=0.99, decay=0.01, nesterov=False)
        adam = Adam(lr = 0.001)
        model.compile(loss='mean_squared_error', optimizer=adam)
        checkpointer = ModelCheckpoint(filepath="cas9.hdf5",verbose=1, monitor='val_loss',save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=400, verbose=1)
        model.fit([train_x,train_nu], train_y, batch_size=64, epochs=500, shuffle=True, validation_data=( [val_x,val_nu], val_y), callbacks=[checkpointer,earlystopper])
        pred_y = model.predict([test_x,test_nu])
        print('testset')
        print('mse ' + str(mse(test_y, pred_y)))
        print(st.spearmanr(test_y, pred_y))
        y_pred_tr = model.predict([train_x,train_nu])
        print(st.spearmanr(train_y, y_pred_tr))
        np.savetxt("train.csv", train_y, delimiter= ",")
        np.savetxt("trainpred.csv", y_pred_tr, delimiter = ",")
        np.savetxt("test.csv" , test_y, delimiter = ",")
        np.savetxt("testpred.csv", pred_y, delimiter = ",")    
		
