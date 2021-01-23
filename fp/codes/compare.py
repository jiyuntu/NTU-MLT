import numpy as np
import csv
import pandas as pd
import sys

file1 = str(sys.argv[1])
file2 = str(sys.argv[2])

if __name__ == '__main__':
	a = pd.read_csv(file1)
	b = pd.read_csv(file2)
	list1 = pd.DataFrame(a, columns = ['label']).values.tolist()
	list2 = pd.DataFrame(b, columns = ['label']).values.tolist()
	score = 0
	scoreabs = 0
	for i in range(len(list1)):
		score += (list1[i][0] - list2[i][0])
		scoreabs += abs(list1[i][0] - list2[i][0])
	score /= len(list1)
	scoreabs /= len(list1)
	print(score)
	print(scoreabs)
