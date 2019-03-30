import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import ndimage
import math
from numpy.linalg import inv

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
    return math.exp(s) / (1+ math.exp(s))


def read(fin):
	f1 = open(fin,'r')
	list1 = []
	list5 = []
	for line in f1:
		temp = line.split()
		if temp[0] == '1.0000':
			del temp[0]
			list1.append([float(i) for i in temp])
		elif temp[0] == '5.0000':
			del temp[0]
			list5.append([float(i) for i in temp])
	return list1, list5

def get_intensity(data):
	intensity = []
	for digit in data:
		temp = 0
		for i in digit:
			temp += i
		intensity.append(temp/len(digit))
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

# Pocket PLA part

#定义sign函数
def sign(x):
    if x>0:
        return 1
    else:
        return -1


def CountError(x,w):
	n=x.shape[1]-1
	count=0
	errordata = []
	for i in x:
		if sign(i[:n].dot(w))*i[-1]<0:
			count+=1
			errordata.append(i)
	return count, errordata


def PocketPLA(x, k, maxnum, wlin):
	#n为数据维度,m为数据数量
	m,n=x.shape
	#注意多了一个标志位
	n-=1
	#初始化向量
	w=wlin

	#错误率最小的向量
	w0=wlin
	error, errordata=CountError(x,w)
	#记录每次的错误
	Error=[10**10]
	if error==0:
		pass
	else:
		#记录次数
		j=0
		while (j<maxnum or error==0):
			#随机选取数据
			k=np.random.randint(0,len(errordata))
			i=errordata[k]
			#得到w(t+1)
			w=w0+k*i[-1]*i[:n]
			error1, errordata1=CountError(x,w)
			#如果w(t+1)比w(t)效果好，则更新w(t)
			if error > error1:
				w0=w[:]
				error=error1
				errordata = errordata1
				print(error)
			Error.append(error)
			j+=1
	return w0,Error


if __name__ == '__main__':


	train1, train5 = read('ZipDigits.train')
	test1, test5 = read('ZipDigits.test')
	# part (b) and (c)
	intensity1 = get_intensity(train1)
	intensity5 = get_intensity(train5)
	symmetry1 = get_symmetry(train1)
	symmetry5 = get_symmetry(train5)

	top = []
	bottom = []

	for i in range(0,len(intensity1)):
		top.append([intensity1[i],symmetry1[i]])

	for i in range(0,len(intensity5)):
		bottom.append([intensity5[i],symmetry5[i]])


	# Linear Regression
	# pre processing
	x1=[[1]+i for i in top]
	x2=[[1]+i for i in bottom]
	y1 = [1]*len(top)
	y2 = [-1]*len(bottom)

	X = np.array(x1 + x2)
	Y = np.array(y1 + y2)
	wlin=inv(X.T.dot(X)).dot(X.T).dot(Y)
	print('W_lin from Linear Regression is: [{}, {}, {}]'.format(wlin[0],wlin[1],wlin[2]))


	# PLA
	# pre processing
	x1=[[1]+i+[1] for i in top]
	x2=[[1]+i+[-1] for i in bottom]
	data=x1+x2
	data=np.array(data)
	np.random.shuffle(data)

	#迭代次数
	num = 1000
	#产生结果
	w,error= PocketPLA(data,1,num, wlin)

	print('Final error: {}\nFinal w: {}, {}, {}\nEin = {}'.format(error[-1], w[0], w[1],w[2], error[-1]/len(data)))


	Einbound = float(math.sqrt(1/float(2*len(data)) * math.log(2*len(Y)/0.05)))
	print('Einbound: {}'.format(Einbound))

	X3=[-1,0]
	Y3=[-(w[0]+w[1]*i)/w[2] for i in X3]
	plt.plot(X3,Y3, c = 'orange')
	plt.scatter(intensity1, symmetry1, s = 12, c = '', edgecolor = 'blue', marker = 'o', label = 'digit 1')
	plt.scatter(intensity5, symmetry5, s = 12, c = 'red', marker = 'x', label = 'digit 5')
	plt.legend()
	plt.xlabel('intensity')
	plt.ylabel('symmetry')
	plt.title('Train Result')
	plt.show()




	# run test data
	tintensity1 = get_intensity(test1)
	tintensity5 = get_intensity(test5)
	tsymmetry1 = get_symmetry(test1)
	tsymmetry5 = get_symmetry(test5)

	top = []
	bottom = []

	for i in range(0,len(tintensity1)):
		top.append([tintensity1[i],tsymmetry1[i]])

	for i in range(0,len(tintensity5)):
		bottom.append([tintensity5[i],tsymmetry5[i]])

	x1=[[1]+i+[1] for i in top]
	x2=[[1]+i+[-1] for i in bottom]
	data=x1+x2
	data=np.array(data)

	error, errordata = CountError(data,w)
	Etest = error/len(data)
	print("Etest:", Etest)
	Etestbound = float(math.sqrt(1/float(2*len(data.shape[0])) * math.log(2*len(Y)/0.05)))
	print('Etestbound: {}'.format(Etestbound))

	plt.plot(X3,Y3,c = 'orange')
	plt.scatter(tintensity1, tsymmetry1, s = 12, c = '', edgecolor = 'blue', marker = 'o', label = 'digit 1')
	plt.scatter(tintensity5, tsymmetry5, s = 12, c = 'red', marker = 'x', label = 'digit 5')
	plt.legend()
	plt.xlabel('intensity')
	plt.ylabel('symmetry')
	plt.title('Test Result')
	plt.show()