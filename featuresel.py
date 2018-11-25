import numpy as np
import pandas as pd
import scipy.stats as st
import sys
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
import scipy.stats as st
from sklearn.metrics import mean_squared_error as mse
from scipy import stats

from keras.preprocessing import sequence
from keras.optimizers import RMSprop,Adam, Adadelta, Nadam, Adamax, SGD, Adagrad
from keras.models import Sequential
from keras.layers.core import  Dropout, Activation, Flatten
from keras.regularizers import l1,l2,l1_l2
from keras.constraints import maxnorm
#from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, MaxPooling1D, Dense, LSTM, Bidirectional

def features_selected(data_file):
	data_file = data_file.values
	#np.random.shuffle(data_file)
	X,Y = data_file[0:33000,0:709], data_file[0:33000,709]
	#print(data_file.shape)
	regr = linear_model.ElasticNet(random_state=0, alpha=0.005, l1_ratio =1)
	regr.fit(X, Y)
	coef = regr.coef_
	intercept = regr.intercept_
	parameters = {"coef": coef,
				  "intercept":intercept}
	return parameters		
	
def features_selected_gbr(data_file):
	data_file = data_file.values
	X = data_file[:,0:688]
	y = data_file[:,688]
	y = np.dot(y,-1) 						# now more positve the score is better cutter it is   
	y = (y - y.min()) / (y.max() - y.min()) # 0-1 normalization. 1 means the best cutter
	X_tr = X[0:33000,:]
	y_tr = y[0:33000]
	print(X_tr.shape)
	print(y_tr.shape)	
	X_test = X[33000:, :]
	y_test = y[33000:]
	print(X_test.shape)
	print(y_test.shape)	
	regr=GradientBoostingRegressor(random_state=0)
	regr.fit(X_tr, y_tr)
	coef = regr.feature_importances_
	return coef
		  
def train_linearReg(data_file, features):
	features = ( features!= 0)
	data_file = data_file.values
	X = data_file[:,0:688]
	y = data_file[:,688]
	y = np.dot(y,-1) 						# now more positve the score is better cutter it is   
	y = (y - y.min()) / (y.max() - y.min()) # 0-1 normalization. 1 means the best cutter
	X_tr = X[0:33000,features]
	y_tr = y[0:33000]
	print(X_tr.shape)
	print(y_tr.shape)
	X_test = X[33000:,features]
	y_test = y[33000:]
	print(X_test.shape)
	print(y_test.shape)
	regr = linear_model.ElasticNet(random_state=0, alpha=0.0003, l1_ratio =1)
	regr.fit(X_tr, y_tr)
	y_pred = regr.predict(X_test)
	#print(y_pred[0:20])
	print('testset')
	print(st.spearmanr(y_test, y_pred))
	print(st.pearsonr(y_test, y_pred))
	y_pred_tr = regr.predict(X_tr)
	print('trainset')
	print(st.spearmanr(y_tr, y_pred_tr))
	print(st.pearsonr(y_tr, y_pred_tr))
	
def train_logisticReg(data_file, features):
	print('lr')
	features = ( features!= 0)
	data_file = data_file.values
	X = data_file[:,0:688]
	y = data_file[:,689]
	#y = np.dot(y,-1) 						# now more positve the score is better cutter it is   
	#y = (y - y.min()) / (y.max() - y.min()) # 0-1 normalization. 1 means the best cutter
	X_tr = X[0:33000,:]
	y_tr = y[0:33000]
	#ind = (y_tr<=0.5)
	#indx = (y_tr>0.5)
	#y_tr[ind] = 0
	#y_tr[indx] = 1
	y_tr = np.array(y_tr, dtype = 'int64')
	#y_tr.astype(int)
	print(X_tr.shape)
	print(y_tr.shape)	
	print('here')
	print(y_tr[0:5])
	frq = np.bincount(y_tr)
	print(frq)
	X_test = X[33000:,:]
	y_test = y[33000:]
	y_test = np.array(y_test, dtype = 'int64')
	print(X_test.shape)
	print(y_test.shape)
	frq = np.bincount(y_test)
	print(frq)
	regr = linear_model.LogisticRegression(random_state=0, penalty = 'l1')
	regr.fit(X_tr, y_tr)
	y_pred = regr.predict(X_test)
	#print(y_pred[0:20])
	print('testset')
	print(st.spearmanr(y_test, y_pred))
	print(st.pearsonr(y_test, y_pred))
	y_pred_tr = regr.predict(X_tr)
	print('trainset')
	print(st.spearmanr(y_tr, y_pred_tr))
	print(st.pearsonr(y_tr, y_pred_tr))

def train_svm(data_file, features):
	features = ( features!= 0)
	data_file = data_file.values
	X = data_file[:,0:688]
	y = data_file[:,689]
	#y = np.dot(y,-1) 						# now more positve the score is better cutter it is   
	#y = (y - y.min()) / (y.max() - y.min()) # 0-1 normalization. 1 means the best cutter
	X_tr = X[0:33000,features]
	y_tr = y[0:33000]
	#ind = (y_tr<=0.5)
	#indx = (y_tr>0.5)
	#y_tr[ind] = 0
	#y_tr[indx] = 1
	#y_tr.astype(int)
	y_tr = np.array(y_tr, dtype = 'int64')
	print(X_tr.shape)
	print(y_tr.shape)	
	X_test = X[33000:,features]
	y_test = y[33000:]
	y_test = np.array(y_test, dtype = 'int64')
	print(X_test.shape)
	print(y_test.shape)
	regr = SVC(random_state=0,kernel = 'poly')
	regr.fit(X_tr, y_tr)
	y_pred = regr.predict(X_test)
	#print(y_pred[0:20])
	print('testset')
	print(st.spearmanr(y_test, y_pred))
	print(st.pearsonr(y_test, y_pred))
	y_pred_tr = regr.predict(X_tr)
	print('trainset')
	print(st.spearmanr(y_tr, y_pred_tr))
	print(st.pearsonr(y_tr, y_pred_tr))


def train_svmlinear(data_file, features):
	features = ( features!= 0)
	data_file = data_file.values
	X = data_file[:,0:688]
	y = data_file[:,689]
	#y = np.dot(y,-1) 						# now more positve the score is better cutter it is   
	#y = (y - y.min()) / (y.max() - y.min()) # 0-1 normalization. 1 means the best cutter
	X_tr = X[0:33000,:]
	y_tr = y[0:33000]
	#ind = (y_tr<=0.5)
	#indx = (y_tr>0.5)
	#y_tr[ind] = 0
	#y_tr[indx] = 1
	#y_tr.astype(int)
	y_tr = np.array(y_tr, dtype = 'int64')
	print(X_tr.shape)
	print(y_tr.shape)	
	X_test = X[33000:,:]
	y_test = y[33000:]
	y_test = np.array(y_test, dtype = 'int64')
	print(X_test.shape)
	print(y_test.shape)
	regr = LinearSVC(random_state=0, penalty='l2')
	regr.fit(X_tr, y_tr)
	y_pred = regr.predict(X_test)
	#print(y_pred[0:20])
	print('testset')
	print(st.spearmanr(y_test, y_pred))
	print(st.pearsonr(y_test, y_pred))
	y_pred_tr = regr.predict(X_tr)
	print('trainset')
	print(st.spearmanr(y_tr, y_pred_tr))
	print(st.pearsonr(y_tr, y_pred_tr))	
	
	
def train_rf(data_file, features):
	features = ( features!= 0)
	data_file = data_file.values
	X = data_file[:,0:688]
	y = data_file[:,688]
	y = np.dot(y,-1) 						# now more positve the score is better cutter it is   
	y = (y - y.min()) / (y.max() - y.min()) # 0-1 normalization. 1 means the best cutter
	X_tr = X[0:33000,features]
	y_tr = y[0:33000]
	#ind = (y_tr<=0.5)
	#indx = (y_tr>0.5)
	#y_tr[ind] = 0
	#y_tr[indx] = 1
	#y_tr.astype(int)
	#y_tr = np.array(y_tr, dtype = 'int64')
	print(X_tr.shape)
	print(y_tr.shape)	
	#frq = np.bincount(y_tr)
	#print(frq)
	X_test = X[33000:, features]
	y_test = y[33000:]
	#y_test = np.array(y_test, dtype = 'int64')
	print(X_test.shape)
	print(y_test.shape)	
	#frq = np.bincount(y_test)
	#print(frq)
	regr=RandomForestRegressor(n_estimators=100, max_depth=3, random_state=0)
	regr.fit(X_tr, y_tr)
	y_pred = regr.predict(X_test)
	#print(y_pred[0:40])

	print('testset')
	print(st.spearmanr(y_test, y_pred))
	print(st.pearsonr(y_test, y_pred))
	y_pred_tr = regr.predict(X_tr)
	print('trainset')
	print(st.spearmanr(y_tr, y_pred_tr))
	print(st.pearsonr(y_tr, y_pred_tr))
	print('feature importance')
	print(regr.feature_importances_)
	
def train_gbr(data_file, features):
	features = ( features!= 0)
	#print('non zero coeff'+str(len(features)))
	
	data_file = data_file.values
	X = data_file[:,0:709]
	y = data_file[:,688]
	y = np.dot(y,-1) 						# now more positve the score is better cutter it is   
	y = (y - y.min()) / (y.max() - y.min()) # 0-1 normalization. 1 means the best cutter
	X_tr = X[0:33000,:]
	y_tr = y[0:33000]
	#ind = (y_tr<=0.5)
	#indx = (y_tr>0.5)
	#y_tr[ind] = 0
	#y_tr[indx] = 1
	#y_tr.astype(int)
	#y_tr = np.array(y_tr, dtype = 'int64')
	print(X_tr.shape)
	print(y_tr.shape)	
	#frq = np.bincount(y_tr)
	#print(frq)
	X_test = X[33000:,:]
	y_test = y[33000:]
	#y_test = np.array(y_test, dtype = 'int64')
	print(X_test.shape)
	print(y_test.shape)	
	#frq = np.bincount(y_test)
	#print(frq)
	regr=GradientBoostingRegressor(random_state=0)
	regr.fit(X_tr, y_tr)
	y_pred = regr.predict(X_test)
	#print(y_pred[0:40])

	print('testset')
	print(st.spearmanr(y_test, y_pred))
	print(st.pearsonr(y_test, y_pred))
	y_pred_tr = regr.predict(X_tr)
	print('trainset')
	print(st.spearmanr(y_tr, y_pred_tr))
	print(st.pearsonr(y_tr, y_pred_tr))
	print('feature importance')
	print(regr.feature_importances_)
	
def train_ffnn(data_file, features):
	#features = ( features!= 0)	
	data_file = data_file.values
	input_df = pd.read_table('data_yeast_mod.csv',sep=',', header = 'infer')
	nc = input_df.ix[:, 'Nucleosome occupancy']
	nc = nc.values
	nc = nc.reshape((len(nc),1))
	#print(nc.shape)
	
	X = data_file[:,0:709]
	X = np.concatenate((X, nc), axis=1)
	y = data_file[:,709]
	y = np.dot(y,-1) 						# now more positve the score is better cutter it is   
	#y = (y - y.min()) / (y.max() - y.min()) # 0-1 normalization. 1 means the best cutter
	#X = stats.zscore(X)
	y= stats.zscore(y)
	X_tr = X[0:31000,:]
	y_tr = y[0:31000]
	X_val = X[31000:33000,:]
	y_val = y[31000:33000]
	X_test = X[33000:,:]
	y_test = y[33000:]
	model = Sequential()
	model.add(Dense(units=256, input_dim=710, activation="relu",kernel_initializer='glorot_uniform')) 
	model.add(Dropout(0.5))
	model.add(Dense(units=180,  activation="relu",kernel_initializer='glorot_uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(units= 60, activation="relu",kernel_initializer='glorot_uniform')) 
	#model.add(Dropout(0.7))
	model.add(Dense(units=1,  activation="linear"))  
	model.summary()
	adam = RMSprop(lr = 0.0001)
	model.compile(loss='mean_squared_error', optimizer=adam)
	checkpointer = ModelCheckpoint(filepath="Cas9.hdf5", verbose=1, monitor='val_loss',save_best_only=True)
	earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
	model.fit(X_tr, y_tr, batch_size=64, epochs=150, shuffle=True, validation_data=(X_val, y_val), callbacks=[checkpointer,earlystopper])
	#model.fit(X_train_s, Y_train_s, batch_size=12, epochs=50, shuffle=True, validation_data=( X_val_s, Y_val_s), callbacks=[checkpointer,earlystopper])
	y_pred = model.predict(X_test)
	#y_pred = y_pred.tolist()
	#y_test = y_test.tolist()
	print('testset')
	print('mse ' + str(mse(y_test,y_pred)))
	print(st.spearmanr(y_test, y_pred))
	#print(st.pearsonr(y_test, y_pred))
	y_pred_tr = model.predict(X_tr)
	#y_pred_tr = y_pred_tr.tolist()
	#y_tr = y_tr.tolist()
	print('trainset')
	print(st.spearmanr(y_tr, y_pred_tr))
	#print(st.pearsonr(y_tr, y_pred_tr))
	#print('feature importance')
	#print(y_pred)
	np.savetxt("train.csv", y_tr, delimiter=",")
	np.savetxt("trainpred.csv", y_pred_tr, delimiter=",")
	np.savetxt("test.csv", y_test, delimiter=",")
	np.savetxt("testpred.csv", y_pred, delimiter=",")
	
if __name__ == '__main__':
	data_file = pd.read_csv('features_yeast.csv', header=None, skiprows=0)
	#parameters = features_selected(data_file)
	#coef = parameters["coef"]
	#print(np.count_nonzero(coef))
	coef =0 
	train_ffnn(data_file, coef)
	#intercept = parameters["intercept"]
	
	
	#train_gbr(data_file, coef)
	'''
	param_coef = features_selected_gbr(data_file)
	train_gbr(data_file, param_coef)
	'''
	#train_rf(data_file, coef)
	
	#train_svm(data_file, coef)
	#train_linearReg(data_file, coef)
	#print('before')
	#train_logisticReg(data_file, coef)
	
	#train_svmlinear(data_file, coef)
	'''
	data_file = pd.read_csv('data_latest.csv', header='infer')
	label = data_file.loc[:,'sgRNALabel']
	frq = np.bincount(label)	
	print(frq)
	'''
	
