import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from sklearn import svm
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from sklearn.ensemble import GradientBoostingRegressor
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import initializers
from sklearn.cluster import KMeans
from preprocessing import each_day_revenue, make_adr, revenue_on_right_day
import os

train = pd.read_csv("./train.csv")
train_label = pd.read_csv("./train_label.csv")
test = pd.read_csv("./test.csv")
test_label = pd.read_csv("./test_nolabel.csv")

def predict_level_DS():
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	revenue = each_day_revenue()
	adr = make_adr()
# the transformation between adr and level
	y1 = [revenue[i][1] for i in range(len(revenue))]
	res_label = pd.DataFrame(train_label, columns = ["label"]).values.tolist()
	y2 = [res_label[i][0] for i in range(len(res_label))]

	split = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
	zipped_lists = zip(y1, y2)
	sorted_pairs = sorted(zipped_lists)
	tuples = zip(*sorted_pairs)
	y1, y2 = [ list(tp) for tp in  tuples]
	split_point = []
	j = 0
#	for i in range(len(y1)):
#		print([y1[i], y2[i]])
	for i in range(len(split)):
		while True:
			if y2[j] == split[i][0] and y2[j + 1] == split[i][1]:
				split_point.append(y1[j])
				split_point.append(y1[j + 1])
				break
			j += 1
#	for i in range(len(split_point)):
#		print(split_point[i])

#	print(len(X))
#	print(len(Xval))
#	print(len(y1))
#	for i in range(len(res)):
#		print([res[i], Yval[i]])
	return split_point
	
if __name__ == "__main__":
	predict_level_DS()
'''
	for i in range(len(res_label)):
		curr = []
		for j in range(10):
			if res_label[i][0] == j:
				curr.append(1)
			else:
				curr.append(0)
		y2.append(curr)
	initer = initializers.HeNormal()
	adr2score = Sequential()
	adr2score.add(Dropout(0.1, input_dim = 3))
	adr2score.add(Dense(units = 256, kernel_initializer = initer, activation = 'relu'))
	adr2score.add(Dropout(0.2))
	adr2score.add(Dense(units = 256, kernel_initializer = initer, activation = 'relu'))
	adr2score.add(Dropout(0.1))
	adr2score.add(Dense(units = 256, kernel_initializer = initer, activation = 'relu'))
	adr2score.add(Dense(units = 10, kernel_initializer = initer, activation = 'softmax'))
	adr2score.compile(loss = 'mae', optimizer = 'Adam', metrics = ['accuracy'])

	current_best_value = 10000000000
	for i in range(50):
		score = 0
		adr2score.fit(X_train, y_train, batch_size = 16, epochs = 1, shuffle = True)
		res = adr2score.predict(X_val)
		score = 0
		for i in range(len(res)):
			idx = np.where(res[i] == np.amax(res[i]))[0][0]
			expect = np.where(y_val[i] == np.amax(y_val[i]))[0][0]
			print(idx)
			score += abs(idx - expect)
		if current_best_value > score:
			current_best_value = score
		a = current_best_value / len(y_val)
		print(a)
	
	current_best_value = float(current_best_value)
	current_best_value /= len(y_val)
	print(current_best_value)
'''
'''
	svm_split = []
	for i in range(9):
		ny = []
		nyval = []
		for j in range(len(y_train)):
			if y_train[j][0] > i:
				ny.append(1)
			else:
				ny.append(0)
		for j in range(len(y_val)):
			if y_val[j][0] >= i:
				nyval.append([1])
			else:
				nyval.append([0])
		model = svm.SVC(C = 100)
		model.fit(X_train, ny)
		svm_split.append(model)
		a = model.score(X_val, nyval)
#		print(a)
	score = 0
	
	for i in range(len(X_val)):
		curr = 0
		for j in range(9):
			curr += svm_split[j].predict([X_val[i]])[0]
#		print([curr, y_val[i][0]])
		score += abs(curr - y_val[i][0])
	score /= len(X_val)
#	print(score)
	return svm_split
'''
























