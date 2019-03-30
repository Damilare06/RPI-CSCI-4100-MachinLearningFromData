import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from random import shuffle

order = 8


def parsearr(arr):
    newarr = np.zeros((16, 16))
    for i in range(0, 16):
        for j in range(0, 16):
            newarr[i][j] = arr[i*16+j]
    return newarr

def comparearrs(arr1, arr2):
    sum = 256
    for i in range(len(arr1)):
        for j in range(len(arr1)):
            if(arr1[i][j] != arr2[i][j]):
                sum -= abs(arr1[i][j] - arr2[i][j])
    return sum/256

def horizontalreflect(arr):
    return(arr[::-1])

def verticalreflect(arr):
    newarr = np.zeros((16, 16))    
    for i in range(0, 16):
        newarr[i] = arr[i][::-1]
    return(newarr)

def thetafunc(s):
    return math.exp(s) / (1 + math.exp(s))


def read(fin):
	f1 = open(fin,'r')
	list1 = []
	listNot1 = []
	for line in f1:
		temp = line.split()
		if temp[0] == '1.0000':
			del temp[0]
			list1.append([float(i) for i in temp])
		elif temp[0] != '1.0000':
			del temp[0]
			listNot1.append([float(i) for i in temp])
	return list1, listNot1


def get_intensity(data):
	intensity = []
	for digit in data:
		temp = 0
		for i in digit:
			temp += i
		intensity.append(temp/256)
	return intensity


	
def get_symmetry(data):
	symmetry = []
	for digit in data:
		temp = 0
		for i in range(0,256,16):
			for j in range(i, i+8):
				if digit[j] == digit[2*i+15-j]:
					temp += 1
		arr = parsearr(digit)
		symmetry.append(comparearrs(arr, horizontalreflect(arr))/2 + (comparearrs(arr, verticalreflect(arr)))/2)
	return symmetry
	

	'''
	def get_symmetry(data):
	symmetry = []
	for digit in data:
		temp = 0
		for i in range(0,256,16):
			for j in range(i, i+8):
				if digit[j] == digit[2*i+15-j]:
					temp += 1
		symmetry.append(temp/128)
	return symmetry
	'''


def get_parameter(data):
	Min = min(data)
	Max = max(data)
	shift = (Max-Min)/2-Max
	scale =1/(Max+shift)
	return shift, scale


def normalization(data, shift, scale):
	for i in range(0, len(data)):
		data[i] = scale*(data[i]+shift)
	return data


def constructDataSet(x1Data1, x2Data1, x1DataNot1, x2DataNot1):
	testData = []
	for i in range(0, len(x1Data1)):
		temp = []
		temp.append(x1Data1[i])
		temp.append(x2Data1[i])
		temp.append(1)
		testData.append(temp)

	for i in range(0, len(x1DataNot1)):
		temp = []
		temp.append(x1DataNot1[i])
		temp.append(x2DataNot1[i])
		temp.append(-1)
		testData.append(temp)
	shuffle(testData)
	temp = np.random.randint(len(testData), size=300)
	temp[::-1].sort()

	trainData = []
	for i in temp:
		trainData.append(testData[i])
		del testData[i]

	return trainData, testData


def L(x,k):
	if k == 0:
		return 1
	elif k == 1:
		return x
	else:
		return ((2*k-1)/k)*x*L(x,k-1)-((k-1)/k)*x*L(x,k-2)

def featureTransformation(data, n): # nth order
	x1 = data[0]
	x2 = data[1]
	result = []
	for i in range(n+1):
		for j in range(i+1):
			result.append(L(x1,i-j)*L(x2,j))
	result.append(data[2])
	return result

def f(x1, x2, n, w):
	one_data = []
	result = 0
	index = 0
	for i in range(n + 1):
		for j in range(n - i + 1):
			# output = L(i,element[1])* L(j, element[2])
			result += w[index] * L(x1, i) * L(x2, i)
			index += 1
	# print(result)
	return  result

def linearRegression(dataSet, Lambda):
	# Linear Regression
	x = [i[0:45] for i in dataSet]
	y = [i[45] for i in dataSet]
	X = np.array(x)
	Y = np.array(y)

	Z = X
	ZT = X.T
	ZTZ = ZT.dot(X)
	ZTZI = inv(ZTZ + Lambda * np.identity(ZTZ.shape[0]))
	weight = ZTZI.dot(ZT).dot(Y)
	return weight

def eight_order_L(x1, x2, weight,order):
    one_data = []
    result = 0
    index = 0
    for i in range(order + 1):
        for j in range(order - i + 1):
            # output = L(i,element[1])* L(j, element[2])
            result += weight[index] * L(x1, i) * L(x2, j)
            index += 1
    # print(result)
    return  result


def plot(data, w):
	x1 = []
	y1 = []
	xNot1 = []
	yNot1 = []

	for i in range(len(data)):
		if data[i][2] == 1:
			x1.append(data[i][0])
			y1.append(data[i][1])
		else:
			xNot1.append(data[i][0])
			yNot1.append(data[i][1])

	X = np.arange(-1, 1, 0.01)
	Y = np.arange(-1, 1, 0.01)
	X, Y = np.meshgrid(X, Y)
	print(X)
	plt.contour(X, Y, eight_order_L(X, Y, w, 8), levels = [0], colors = 'orange')
	plt.scatter(x1, y1, s = 12, c = '', edgecolor = 'blue', marker = 'o', label = 'digit 1')
	plt.scatter(xNot1, yNot1, s = 12, c = 'red', marker = 'x', label = 'digit not 1')
	plt.legend()
	plt.xlabel('intensity')
	plt.ylabel('symmetry')
	plt.show()

def sign(x):
	if x > 0:
		return 1
	else:
		return -1



def CV(Lambda, trainSet, testData):
	wreg = linearRegression(trainSet, Lambda)
	y_test = f(testData[1], testData[2], 8, wreg)
	if testData[-1] == sign(y_test):
		return 0
	else:
		return (y_test - testData[-1])**2
		# return 1
def Etest(Lambda, trainSet, testSet):
	wreg = linearRegression(trainSet, Lambda)
	Eout = 0
	for testData in testSet:
		y_test = f(testData[0], testData[1], 8, wreg)
		if testData[2] == sign(y_test):
			continue
		else:
			Eout += (y_test - testData[2])**2
			# Eout += 1
	return Eout/len(testSet)

if __name__ == '__main__':

	# load data
	train1, trainNot1 = read('ZipDigits.train')
	test1, testNot1 = read('ZipDigits.test')

	# Data preprocessing
	data1 = train1 + test1
	dataNot1 = trainNot1 + testNot1

	# features selection
	intensity1 = get_intensity(data1)
	intensityNot1 = get_intensity(dataNot1)
	symmetry1 = get_symmetry(data1)
	symmetryNot1 = get_symmetry(dataNot1)

	# input normalization
	shift, scale = get_parameter(intensity1+intensityNot1)
	intensity1 = normalization(intensity1, shift, scale)
	intensityNot1 = normalization(intensityNot1, shift, scale)
	shift, scale = get_parameter(symmetry1+symmetryNot1)
	symmetry1 = normalization(symmetry1, shift, scale)
	symmetryNot1 = normalization(symmetryNot1, shift, scale)

	# construnct data set
	trainSet, testSet = constructDataSet(intensity1, symmetry1, intensityNot1, symmetryNot1)

	# Problem 1
	tSet = [] # Data set after transformation
	for Data in trainSet:
		tSet.append(featureTransformation(Data, order))
	print('Dimension of Z after 8th order transformation is {}*{}'.format(len(tSet), len(tSet[0])-1))
	# Problem 2
	
	wreg0 = linearRegression(tSet, 0)
	print(wreg0)
	plot(trainSet, wreg0)
	'''
	# Problem 3
	wreg2 = linearRegression(tSet, 2)
	print(wreg2)
	plot(trainSet, wreg2)


	# Problem 4
	minEcv = 100
	Ecv = []
	Eout = []
	bestLambda = 0 
	for n in np.arange(0.1, 2.1, 0.1):
		E_n = 0
		for j in range(0, 300):
			temp = [item for item in tSet]
			testData = temp[j]
			del temp[j]
			E_n += CV(n, temp, testData)
		Ecv.append(E_n/300)
		if(E_n/300 < minEcv):
			minEcv = E_n/300
			bestLambda = n
		Eout.append(Etest(n, tSet, testSet))
	

	x = np.arange(0.1, 2.1, 0.1)
	plt.figure()
	plt.plot(x, Ecv, color = 'red', label = 'Ecv')
	plt.plot(x, Eout, color = 'blue', label = 'Eout')
	plt.legend()
	plt.show()

	# Problem 5
	print('The best lambda is {} with the minimum Ecv {}'.format(bestLambda, minEcv))

	bestWreg = linearRegression(tSet, bestLambda)
	print(bestWreg)
	plot(trainSet, bestWreg)

	# Problem 6
	Etest = Etest(bestLambda, tSet, testSet)
	print('Estimate Eout is {}.'.format(Etest))
	'''