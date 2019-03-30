import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import cKDTree

def generateTrain(n):
	n = int(math.sqrt(n))
	x1 = [i for i in range(n)]
	x2 = [i for i in range(n)]
	X1 = []
	X2 = []
	for i in x1:
		for j in x2:
			X1.append(i/100)
			X2.append(j/100)
	return X1, X2

def generateTrainGaussian(n):
	centersX1 = np.random.random_sample((10,))
	centersX2 = np.random.random_sample((10,))
	x1 = []
	x2 = []
	mu, sigma = 0.5, 0.1 # mean and standard deviation
	for i in range(len(centersX1)):
		t1 = np.random.normal(centersX1[i], sigma, (int(n/10),1)) # normal distribution for x1
		t2 = np.random.normal(centersX2[i], sigma, (int(n/10),1)) # normal distribution for x2
		for i in range(len(t1)):
			x1.append(float(t1[i]))
			x2.append(float(t2[i]))
	return x1, x2

def generateTest(n):
	return np.random.random_sample((10000,2))


def distance(test, train): # using Euclidean
	Sum = 0
	for i in range(0, len(train)):
		Sum += math.pow(test[i] - train[i], 2)
	return math.sqrt(Sum)

def distanceToSet(point, centers):
	distances = []
	for center in centers:
		distances.append(distance(point, center))
	return min(distances)

def simpleGreedy(X, n):
	# return n centers
	random = np.random.randint(len(X))
	centers = [X[random]]
	while len(centers) < n:
		maxDistance = -1
		maxPoint = []
		for point in X:
			if point in centers:
				continue
			if distanceToSet(point, centers) > maxDistance:
				maxPoint = point
				maxDistance = distanceToSet(point, centers)
		centers.append(maxPoint)
	return centers

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

def nearestNeighborForce(X, testPoint):
	minDistance = 1000000
	NN = []
	for trainPoint in X:
		Distance = distance(testPoint, trainPoint)
		if Distance < minDistance:
			NN = trainPoint
			minDistance = Distance
	return NN

def catrgorize(X, centers):
	regions = [[] for i in range(len(centers))]
	# print(centers)
	voronoiKdtree = cKDTree(centers)
	for point in X:
		pointDist, regionIndex = voronoiKdtree.query(point)
		regions[regionIndex].append(point)
	return regions

def nearestCenter(centers, testPoint):
	minDistance = 10000
	nCenter = 11
	for i in range(len(centers)):
		Distance = distance(testPoint, centers[i])
		if Distance < minDistance:
			nCenter = i
			minDistance = Distance
	return nCenter

def nearestNeighborPartition(centers, regions, testPoint):
	nCenter = nearestCenter(centers, testPoint)
	NN = nearestNeighborForce(regions[nCenter], testPoint)
	return NN



if __name__ == '__main__':
	n = 10000

	trainx1, trainx2 = generateTrainGaussian(n)



	trainX = [[trainx1[i], trainx2[i]] for i in range(len(trainx1))]

	testX = generateTest(n)

	startTime = time.time()
	centers = simpleGreedy(trainX, 10)
	regions = catrgorize(trainX, centers)
	NNs1 = []
	for testPoint in testX:
		NNs1.append(nearestNeighborPartition(centers, regions, testPoint))
	endTime = time.time()
	print('Time cost for finding the nearest neighbor with partition: {}'.format(endTime-startTime))

	NNs2 = []

	startTime = time.time()
	for testPoint in testX:
		NNs2.append(nearestNeighborForce(trainX, testPoint))
	endTime = time.time()

	print('Time cost for finding the nearest neighbor with brute force: {}'.format(endTime-startTime))

	print('length of NNs1: {}, length of NNs2: {}'.format(len(NNs1), len(NNs2)))
	count = 0
	for i in range(len(NNs1)):
		if NNs1[i] == NNs2[i]:
			count += 1

	print('out of {} test points, {} of results matches'.format(len(NNs1), count))


	centers = np.array(centers)
	vor = Voronoi(centers)
	voronoi_plot_2d(vor)


	plt.scatter(trainx1, trainx2, s = 1)
	plt.scatter([centers[i][0] for i in range(len(centers))], [centers[i][1] for i in range(len(centers))], s = 15)
	plt.show()