import pandas as pd
import numpy as np
import csv
import tensorflow as tf
import math
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from keras.models import Sequential, load_model
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import initializers
from sklearn.model_selection import train_test_split
from datetime import datetime, date
from preprocessing import each_day_revenue, make_X, make_cancel, make_adr 
import os
import pickle as pk
import dill

modelname = 'cancel_mergeNN'

def predict_cancel_mergeNN():
	#if os.path.exists(modelname):
	#	model = load_model(modelname)
	#	return model
	feature, date = make_X()
	cancel = make_cancel()
	x_train = []
	y_train = []
	train_date = []
	i = 0

	while i < len(feature):
		j = i + 1
		canc_num = cancel[i][1]
		while j < len(feature) and feature[i] == feature[j]:
			canc_num += cancel[i][1]
			j += 1
		canc_num /= (j - i)
		x_train.append(feature[i])
		y_train.append([1 - canc_num, canc_num])
		train_date.append(date[i])
		i = j
	x_train = np.array(x_train)
	y_train = np.array(y_train)
	
#	X, Xval, Y, Yval = train_test_split(x_train, y_train, test_size=0.1, shuffle = True)
	X,Y = x_train, y_train

#	score = 0
#	mae = 0
#	for i in range(len(Xval)):
#		score += (res[i][1] - Yval[i][1]) * (res[i][1] - Yval[i][1])
#		mae += abs(res[i][1] - Yval[i][1])
#	score /= len(res)
#	mae /= len(res)
#	score = math.sqrt(score)
#	print(score)
#	print(mae)
	
	initer = initializers.HeNormal()
	model = Sequential()
	model.add(Dropout(0.1, input_dim = 123))
	model.add(Dense(units = 256, kernel_initializer = initer, activation = 'relu'))
	model.add(Dropout(0.2))
	model.add(Dense(units = 512, kernel_initializer = initer, activation = 'relu'))
	model.add(Dropout(0.1))
	model.add(Dense(units = 256, kernel_initializer = initer, activation = 'relu'))
	model.add(Dense(units = 2, activation = 'softmax'))
	model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
	
#	model.fit(X, Y, batch_size = 128, epochs = 100, validation_data = (Xval, Yval))
	model.fit(X, Y, batch_size = 128, epochs = 100)

#	res = model.predict(Xval)
#	score = 0
#	for i in range(len(res)):
#		score += abs(res[i][0] - Yval[i][0])
#	score /= len(res)
#	print(score)

	model.save(modelname)

	return model

if __name__ == "__main__":
	os.environ['CUDA_VISIBLE_DEVICES'] = '3'
	predict_cancel_mergeNN()



















