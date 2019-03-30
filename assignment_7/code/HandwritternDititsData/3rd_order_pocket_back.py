import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import ndimage
import math
from numpy.linalg import inv

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
		intensity.append(temp+250)
	return intensity

def get_symmetry(data):
	symmetry = []
	for digit in data:
		temp = 0
		for i in range(0,256,16):
			for j in range(i, i+8):
				if digit[j] == digit[2*i+15-j]:
					temp += 1
		symmetry.append(temp)
	return symmetry

# Pocket PLA part

def f(x,y,w):
    return w[0]+w[1]*x+w[2]*y+w[3]*x*y+w[4]*(x**2)+w[5]*(y**2)+w[6]*(x**3)+w[7]*(x**2)*y+w[8]*x*(y**2)+w[9]*y**3

def solve(data, w):
	output = data[-1]
	x = data[1]
	y = data[2]

	a = w[9]
	b = w[5]+w[8]*x
	c = w[2]+w[4]*x+w[7]*(x**2)
	d = w[0]+w[1]*x+w[3]*(x**2)+w[6]*(x**3)

	u = (9*a*b*c-27*(a**2)*d-2*(b**3))
	v = (3*( 4*a*(c**3)- b*b*c*c-18*a*b*c*d+27*a*a*d*d+4*b*b*b*d ))**(1./3)/(18*(a**2))
	m = 0
	n = 0
	if abs(u+v) >= abs(u-v):
		m = (u+v)**(1./3)
	else:
		m = (u-v)**(1./3)

	if abs(m) != 0:
		n = (b*b-3*a*c)/(9*a*m)

	y1 = m+n-(b/(3*a))
	print(y1, y, output)
	return sign(y1,y,output)

#定义sign函数
def sign(x1,x2,y):
    if y == 1:
        return x1<x2
    else:
        return x1>x2

#定义计算错误个数的函数
def CountError(x,w):
    #数据维度
    n=x.shape[1]-1
    #记录错误次数
    count=0
    for i in x:
        if solve(i,w) == False:
            count+=1
    return count


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

    print("# of data: {}".format(len(x)))
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
            print(j, error)
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

	minx = min(intensity1)
	maxx = max(intensity5)
	miny = min(symmetry5)
	maxy = max(symmetry1)


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
	np.random.shuffle(data)

	newdata = transform(data)

	Xnew = newdata[:,:-1]
	Ynew = newdata[:,-1]
	wlin=inv(Xnew.T.dot(Xnew)).dot(Xnew.T).dot(Ynew)


	# 数据数目
	n = len(intensity1)+len(intensity5)
	print(len(intensity1))
	print(len(intensity5))
	t = 1
	# 定义x, y
	x = np.linspace(0, 260, n)
	y = np.linspace(0, 120, n)
	# 生成网格数据
	X, Y = np.meshgrid(x, y)
	print(CountError(newdata, wlin))

	'''
	plt.contour(X, Y, f(X, Y, wlin), 1, colors = 'red')
	plt.scatter(intensity1, symmetry1, s = 12, c = '', edgecolor = 'blue', marker = 'o', label = 'digit 1')
	plt.scatter(intensity5, symmetry5, s = 12, c = 'red', marker = 'x', label = 'digit 5')
	plt.legend()
	plt.xlabel('intensity')
	plt.ylabel('symmetry')
	plt.show()
	



	# PLA
	# 迭代次数
	num = 100
	# 产生结果
	print('PLA starts here')
	w,error= PocketPLA(newdata, 1, num,wlin)

	#print('Final error: {}\nFinal w: {}\nEin = {}'.format(error[-1], w, error[-1]/newdata.shape[0]))

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