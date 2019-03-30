import numpy as np
import math
import matplotlib.pyplot as plt
import heapq


def distance(test, train): # using Euclidean
	Sum = 0
	for i in range(0, len(train)):
		Sum += math.pow(test[i] - train[i], 2)
	return math.sqrt(Sum)

def knn(test, X, Y, k):
	countTrue = 0
	countFalse = 0
	distances = []
	for train in X:
		distances.append(distance(test, train))
	minKpoints = map(distances.index, heapq.nsmallest(k, distances))
	for i in minKpoints:
		if Y[i] == 1:
			countTrue += 1
		else:
			countFalse += 1
	return countTrue > countFalse

def transform(X):
	Z = []
	for i in range(len(X)):
		z1 = math.sqrt(math.pow(X[i][0], 2) + math.pow(X[i][1], 2))
		if X[i][0] == 0:
			z2 = np.arctan(math.inf)
		else:
			z2 = np.arctan(X[i][1]/X[i][0])
		Z.append([z1, z2])
	return Z

if __name__ == '__main__':

	k = 3

	x1False = [1.0, 0.0, 0.0, -1.0]
	x2False = [0.0, 1.0, -1.0, 0.0]
	x1True = [0.0, 0.0, -2.0]
	x2True = [2.0, -2.0, 0.0]

	X = []
	Y = []
	for i in range(0, len(x1True)):
		X.append([x1True[i], x2True[i]])
		Y.append(1)
	for i in range(0, len(x1False)):
		X.append([x1False[i], x2False[i]])
		Y.append(-1)

	x1true = []
	x2true = []
	x1false = []
	x2false = []

	for i in np.arange(-4, 4, 0.02):
		for j in np.arange(-4, 4, 0.02):
			if knn([i,j], X, Y, k):
				x1true.append(i)
				x2true.append(j)
			else:
				x1false.append(i)
				x2false.append(j)


	plt.scatter(x1true, x2true, s = 10, color = 'orange', label = 'h(x) = +1')
	plt.scatter(x1false, x2false, s = 10, color = 'pink', label = 'h(x) = -1')
	plt.scatter(x1True, x2True, s = 15, marker = 'o', color = 'blue', label = 'y = +1')
	plt.scatter(x1False, x2False, s = 15, marker = 'x', color = 'red', label = 'y = -1')
	plt.legend()
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.title('{}NN Plot'.format(k))
	plt.show()