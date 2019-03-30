import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import math
import heapq
import classify as cls
import nn
import lengendreTransfrom as lt

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


def get_symmetry(digit):
    arr = parsearr(digit)

    symmetry = 0
    for i in range(0,256,16):
        for j in range(i, i+8):
            if digit[j] == digit[2*i+15-j]:
                symmetry += 1

    # symmetry = comparearrs(arr, horizontalreflect(arr))/2 + (comparearrs(arr, verticalreflect(arr)))/2
    return symmetry


def get_intensity(digit):
    intensity = 0
    for i in digit:
        intensity += i
    return intensity/256

def dataProcess(files):
    digit = []
    for file in files:
        raw_data = np.loadtxt(file)
        data = raw_data[:, 1:]
        for index in range(len(data)):
            symmetry = get_symmetry(data[index])
            intensity = get_intensity(data[index])
            digit.append([int(raw_data[index, 0]) == 1, intensity, symmetry])
    digit = np.array(digit)
    return digit

def normalize(dataSet):
    # transfer feature one by one
    dataSet = dataSet.astype(np.float32)
    for i in range(1, len(dataSet[0])):
        max1 = np.max(dataSet[:,i])
        min1 = np.min(dataSet[:,i])
        diff = np.max(dataSet[:,i]) + np.min(dataSet[:,i])
        dataSet[:,i] = 1.0*(dataSet[:,i] - min1 - (max1-min1)/2) / ((max1-min1)/2)
    return dataSet

def splitData(dataSet):
    index = np.array(random.sample(range(len(dataSet)), 300))
    testIndex = np.delete(np.arange(len(dataSet)), index)
    trainSet = dataSet[index]
    testSet = dataSet[testIndex]
    return trainSet, testSet

def analysisData(dataSet):
    digit1, digitNot1 = [], []
    for i in range(len(dataSet)):
        if dataSet[i][0] == 1:
            digit1.append([dataSet[i][1],dataSet[i][2]])
        else:
            digitNot1.append([dataSet[i][1], dataSet[i][2]])
    return digit1, digitNot1

def distance(a, b): # using Euclidean
    Sum = 0
    for i in range(1, len(a)):

        Sum += math.pow(b[i] - a[i], 2)
    return math.sqrt(Sum)

def knn(testPoint, trainSet, k):
    countTrue = 0
    countFalse = 0
    distances = []
    for train in trainSet:
        distances.append(distance(testPoint, train))
    minKpoints = map(distances.index, heapq.nsmallest(k, distances))
    for i in minKpoints:
        if trainSet[i][0] == 1:
            countTrue += 1
        else:
            countFalse += 1
    return countTrue > countFalse

def cv(dataSet, k):
    Error = 0
    for i in range(len(dataSet)):
        temp = dataSet.tolist()
        del temp[i]
        Error += (knn(dataSet[i], temp, k) != dataSet[i][0])
    return Error/len(dataSet)

def calcError(dataSet, trainSet, k):
    Error = 0
    for data in dataSet:
        Error += (knn(data, trainSet, k) != data[0])
    return Error/len(dataSet)




# point format: (y, x1, x2)
if __name__ == '__main__':
    files = ['ZipDigits.train' , 'ZipDigits.test']
    digit = dataProcess(files)
    digit = normalize(digit)

    trainSet, testSet = splitData(digit)
    

    # Problem 1:
    print('=============Problem 1=============')
    kList = range(1, 50)
    EcvList = []
    bestK = 0
    minEcv = 1
    for k in kList:
        Ecv = cv(trainSet, k)
        if Ecv < minEcv:
            bestK = k
            minEcv = Ecv
        EcvList.append(Ecv)
        # EcvList.append(CV(trainSet, k))
    
    Ein = calcError(trainSet, trainSet, bestK)
    Etest = calcError(testSet, trainSet, bestK)
    print('best K: ', bestK)
    print('minimum Ecv:', minEcv)
    print('Ein:', Ein)
    print('Etest:', Etest)

    plt.plot(kList, EcvList)
    plt.xlabel('k')
    plt.ylabel('Ecv')
    plt.show()

    x1true = []
    x2true = []
    x1false = []
    x2false = []
    for i in np.arange(-1, 1.02, 0.02):
        for j in np.arange(-1, 1.02, 0.02):
            if knn([True, i, j], trainSet, bestK):
                x1true.append(i)
                x2true.append(j)
            else:
            	x1false.append(i)
            	x2false.append(j)

    digit1, digitNot1 = analysisData(trainSet)

    plt.scatter(x1true, x2true, s = 12, color = 'orange', label = 'h(x) == 1')
    plt.scatter(x1false, x2false, s = 12, color = 'pink', label = 'h(x) != 1')
    plt.scatter([digit1[i][0] for i in range(len(digit1))], [digit1[i][1] for i in range(len(digit1))], s = 12, c = '', edgecolor = 'blue', marker = 'o', label = 'digit 1')
    plt.scatter([digitNot1[i][0] for i in range(len(digitNot1))], [digitNot1[i][1] for i in range(len(digitNot1))], s = 12, c = 'red', marker = 'x', label = 'digit not 1')
    plt.legend()
    plt.xlabel('intensity')
    plt.ylabel('symmetry')
    plt.show()

    # Problem 2:
    train_data = []
    test_data = []
    for point in trainSet:
        if point[0] == 0:
            point[0] = -1
        else:
            point[0] = 1
        train_data.append([[point[1], point[2]],point[0]])
    for point in testSet:
        test_data.append([[point[1], point[2]],point[0]])

    print('=============Problem 2=============')
    min_k = 0
    min_e_cv = 1
    e_cv_list = []
    k_list = range(1,50)
    print("Selecting K...")
    for k in k_list:
        #if k%10==0:print k
        e_cv = nn.CrossValidRBF(k, train_data)
        e_cv_list.append(e_cv)
        if e_cv < min_e_cv:
            min_e_cv = e_cv
            min_k = k
            #print "update k:", k, "e_cv:", e_cv
            
    centers, w_lin = nn.RBF_Learn(train_data, min_k)   
    print("best k and cv:", min_k, min_e_cv)
    r=2/(np.sqrt(min_k))
    print("Ein:", nn.RBFAccuracyCalc(w_lin, centers, r, train_data))
    print("etest:", nn.RBFAccuracyCalc(w_lin, centers, r, test_data))
    plt.plot(k_list, e_cv_list)
    plt.xlabel('k')
    plt.ylabel('Ecv')
    plt.show()
    


    x1true = []
    x2true = []
    x1false = []
    x2false = []
    for i in np.arange(-1, 1.02, 0.02):
        for j in np.arange(-1, 1.02, 0.02):
            if nn.RBF_Classify(w_lin, centers, [i,j], r) == 1:
                x1true.append(i)
                x2true.append(j)
            else:
                x1false.append(i)
                x2false.append(j)

    digit1, digitNot1 = analysisData(trainSet)

    plt.scatter(x1true, x2true, s = 12, color = 'orange', label = 'h(x) == 1')
    plt.scatter(x1false, x2false, s = 12, color = 'pink', label = 'h(x) != 1')
    plt.scatter([digit1[i][0] for i in range(len(digit1))], [digit1[i][1] for i in range(len(digit1))], s = 12, c = '', edgecolor = 'blue', marker = 'o', label = 'digit 1')
    plt.scatter([digitNot1[i][0] for i in range(len(digitNot1))], [digitNot1[i][1] for i in range(len(digitNot1))], s = 12, c = 'red', marker = 'x', label = 'digit not 1')
    plt.legend()
    plt.xlabel('intensity')
    plt.ylabel('symmetry')
    plt.show()
