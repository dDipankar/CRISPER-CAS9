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
from keras.layers import Conv1D, MaxPooling1D, Dense, LSTM, Bidirectional, BatchNormalization, MaxPooling2D, AveragePooling1D, Input, Multiply
from sklearn.metrics import mean_squared_error as mse
import scipy.stats as st
#from keras.utils import plot_model
#from keras.utils.layer_utils import print_layer_shapes
# fix random seed for reproducibility
np.random.seed(1369)

from numpy import *
import sys;  

from keras.models import Model
from keras.layers import Input
from keras.layers.merge import Multiply
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, AveragePooling1D

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
        SEQ, CA, score = PREPROCESS(data)
        score = np.dot(score,-1)
        score = st.zscore(score) 
        print(SEQ.shape)
        #SEQ = np.transpose(np.array(SEQ),axes=(0,2,1))
        #print(SEQ.shape)
        FILE.close()
        train_x = SEQ[0:33000,:]
        train_nu = CA[0:33000]
        train_y = score[0:33000]
        print(train_x.shape)
        print(train_y.shape)
        print(train_nu.shape)
        print(train_nu[0:10])
        val_x = SEQ[33000:35000,:]
        val_y = score[33000:35000]
        val_nu = CA[33000:35000]
        print(val_x.shape)
        print(val_y.shape)
        print(val_nu.shape)
        print(val_nu[0:10])
        test_x = SEQ[35000:,:]
        test_y = score[35000:]
        test_nu = CA[35000:]
        print(test_x.shape)
        print(test_y.shape)
        print(test_nu.shape)
        print(test_nu[0:10])
        
        # model for seq
       
        DeepCpf1_Input_SEQ = Input(shape=(40,4))
        DeepCpf1_C1 = Convolution1D(80, 5, activation='relu')(DeepCpf1_Input_SEQ)
        DeepCpf1_P1 = AveragePooling1D(2)(DeepCpf1_C1)
        DeepCpf1_F = Flatten()(DeepCpf1_P1)
        DeepCpf1_DO1= Dropout(0.3)(DeepCpf1_F)
        DeepCpf1_D1 = Dense(80, activation='relu')(DeepCpf1_DO1)
        DeepCpf1_DO2= Dropout(0.3)(DeepCpf1_D1)
        DeepCpf1_D2 = Dense(40, activation='relu')(DeepCpf1_DO2)
        DeepCpf1_DO3= Dropout(0.3)(DeepCpf1_D2)
        DeepCpf1_D3_SEQ = Dense(40, activation='relu')(DeepCpf1_DO3)
    
        DeepCpf1_Input_CA = Input(shape=(1,))
        DeepCpf1_D3_CA = Dense(40, activation='relu')(DeepCpf1_Input_CA)
        DeepCpf1_M = Multiply()([DeepCpf1_D3_SEQ, DeepCpf1_D3_CA])
    
        DeepCpf1_DO4= Dropout(0.3)(DeepCpf1_M)
        DeepCpf1_Output = Dense(1, activation='linear')(DeepCpf1_DO4)
        DeepCpf1 = Model(inputs=[DeepCpf1_Input_SEQ, DeepCpf1_Input_CA], outputs=[DeepCpf1_Output])
        DeepCpf1.summary()
        
        
        adam = Adam(lr = 0.001)
        DeepCpf1.compile(loss='mean_squared_error', optimizer=adam)
        checkpointer = ModelCheckpoint(filepath="cas9.hdf5",verbose=1, monitor='val_loss',save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        DeepCpf1.fit([train_x,train_nu], train_y, batch_size=64, epochs=150, shuffle=True, validation_data=( [val_x,val_nu], val_y), callbacks=[checkpointer,earlystopper])
        pred_y = DeepCpf1.predict([test_x,test_nu])
        print('testset')
        print('mse ' + str(mse(test_y, pred_y)))
        print(st.spearmanr(test_y, pred_y))
        y_pred_tr = DeepCpf1.predict([train_x,train_nu])
        print(st.spearmanr(train_y, y_pred_tr))
        np.savetxt("train.csv", train_y, delimiter= ",")
        np.savetxt("trainpred.csv", y_pred_tr, delimiter = ",")
        np.savetxt("test.csv" , test_y, delimiter = ",")
        np.savetxt("testpred.csv", pred_y, delimiter = ",")
       
        
        
        
        
