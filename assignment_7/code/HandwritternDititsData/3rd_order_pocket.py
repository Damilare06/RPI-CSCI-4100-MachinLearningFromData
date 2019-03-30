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

#定义计算错误个数的函数
def CountError(x,w):
    n=x.shape[1]-1
    count=0
    for i in x:
        t1 = np.asarray(i[:n])
        t2 = np.asarray(w)
        t3 = np.inner(t1,t2)
        print(t3, i[-1])
        if sign(t3)*i[-1]<0:
            count+=1
    return count

def f(x,y,w):
    return w[0]+w[1]*x+w[2]*y+w[3]*x*y+w[4]*(x**2)+w[5]*(y**2)+w[6]*(x**3)+w[7]*(x**2)*y+w[8]*x*(y**2)+w[9]*y**3

def transform(data):
    result=[]
    for i in data:
        x1=i[1]
        x2=i[2]
        flag=i[-1]
        x=[1,x1,x2,x1*x2,x1**2,x2**2,x1**3,(x1**2)*x2,x1*(x2**2),x2**3,flag]
        result.append(x)
    return np.array(result)

def PocketPLA(x,k,maxnum, wlin):
    #n为数据维度,m为数据数量
    m,n=x.shape
    #注意多了一个标志位
    n-=1
    #初始化向量
    w=wlin
    #错误率最小的向量
    w0=wlin
    error=CountError(x,w)
    #记录每次的错误
    Error=[]
    if error==0:
        pass
    else:
        #记录次数
        j=0
        while (j<maxnum or error==0):
            #随机选取数据
            k=np.random.randint(0,m)
            i=x[k]
            #得到w(t+1)
            w=w0+k*i[-1]*i[:n]
            error1=CountError(x,w)
            #如果w(t+1)比w(t)效果好，则更新w(t)
            if error>error1:
                w0=w[:]
                error=error1

            Error.append(error)
            j+=1
            if(j == maxnum):
            	break
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
		bottom.append([intensity1[i],symmetry1[i]])



	# Linear Regression
	# pre processing
	x1 = [[1]+i+[1] for i in top]
	x2 = [[1]+i+[-1] for i in bottom]
	data=x1+x2
	data = np.array(data)

	newdata = transform(data)

	Xnew = newdata[:,:-1]
	Ynew = newdata[:,-1]


	# 数据数目
	n = 2000
	# 定义x, y
	x = np.linspace(-1, 0, n)
	y = np.linspace(0, 1, n)
	w1=inv(Xnew.T.dot(Xnew)).dot(Xnew.T).dot(Ynew)
	print(w1)

	# 生成网格数据
	X, Y = np.meshgrid(x, y)
	print(CountError(newdata, w1))

	plt.contour(X, Y, f(X, Y, w1), 1, colors = 'red')
	plt.scatter(intensity1, symmetry1, s = 12, c = '', edgecolor = 'blue', marker = 'o', label = 'digit 1')
	plt.scatter(intensity5, symmetry5, s = 12, c = 'red', marker = 'x', label = 'digit 5')
	plt.legend()
	plt.xlabel('intensity')
	plt.ylabel('symmetry')
	plt.show()


	'''
	# PLA
	# 迭代次数
	num = 1000
	# 产生结果
	print('PLA starts here')
	w,error= PocketPLA(newdata, 1, num, w1)

	print('Final error: {}\nFinal w: {}\nEin = {}'.format(error[-1], w, error[-1]/newdata.shape[0]))

	plt.contour(X, Y, f(X, Y, w), 1, colors = 'red')
	plt.scatter(intensity1, symmetry1, s = 12, c = '', edgecolor = 'blue', marker = 'o', label = 'digit 1')
	plt.scatter(intensity5, symmetry5, s = 12, c = 'red', marker = 'x', label = 'digit 5')
	plt.legend()
	plt.xlabel('intensity')
	plt.ylabel('symmetry')
	plt.show()


	# run test data
	tintensity1 = get_intensity(test1)
	tintensity5 = get_intensity(test5)
	tsymmetry1 = get_symmetry(test1)
	tsymmetry5 = get_symmetry(test5)

	top = []
	bottom = []

	for i in range(0,len(intensity1)):
		top.append(three_order(intensity1[i],symmetry1[i]))

	for i in range(0,len(intensity5)):
		top.append(three_order(intensity1[i],symmetry1[i]))

	x1=[[1]+i+[1] for i in top]
	x2=[[1]+i+[-1] for i in bottom]
	data=x1+x2
	data=np.array(data)

	Eout = CountError(data,w)/len(data)
	print("Eout:", Eout)

	plt.plot(X3,Y3)
	plt.scatter(tintensity1, tsymmetry1, s = 12, c = '', edgecolor = 'blue', marker = 'o', label = 'digit 1')
	plt.scatter(tintensity5, tsymmetry5, s = 12, c = 'red', marker = 'x', label = 'digit 5')
	plt.legend()
	plt.xlabel('intensity')
	plt.ylabel('symmetry')
	plt.show()	
	'''