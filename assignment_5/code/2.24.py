import math
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import integrate

def f(x):
	return x**2

#模拟求出系数
def simulation(n):
	a=0
	b=0
	#模拟n次
	for i in range(n):
		#产生-1,1之间均匀分布随机数
		x1=random.uniform(-1,1)
		x2=random.uniform(-1,1)
		y1=f(x1)
		y2=f(x2)
		a1=(x1+x2)
		b1=-x1*x2
		a+=a1
		b+=b1
	return(a/n,b/n)


#定义误差函数
def Eout(x,x1,x2):
	y1=f(x1)
	y2=f(x2)
	a=(x1+x2)
	b=-x1*x2
	y=f(x)
	y1=a*x+b
	return 1/8*(y-y1)**2
integrate.tplquad(Eout, -1, 1, lambda x: -1, lambda x: 1,lambda x, y: -1, lambda x, y: 1)

def bias(x):
	y1=a*x+b
	y2=f(x)
	return (y1-y2)**2/2

def var(x,x1,x2):
	yavg=a*x+b
	y1=f(x1)
	y2=f(x2)
	a1=(x1+x2)
	b1=-x1*x2
	yrea=a1*x+b1
	return 1/8*(yavg-yrea)**2




if __name__ == '__main__':
	'''
	n=range(10**4,10**5+1,10**4)
	result=[]
	for i in n:
		temp=simulation(i)
		print(temp)
		result.append(temp)
	'''

	plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
	plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
	# a,b=result[-1]
	a, b = simulation(1000)
	x1=np.arange(-1,1.1,0.1)
	y1=[f(i) for i in x1]	
	x2=[-1,1]
	y2=[a*i+b for i in x2]
	plt.plot(x1,y1)
	plt.plot(x2,y2)
	Eout=integrate.tplquad(Eout, -1, 1, lambda x: -1, lambda x: 1,lambda x, y: -1, lambda x, y: 1)[0]
	bias = integrate.quad(bias,-1,1)[0]
	var = integrate.tplquad(var, -1, 1, lambda x: -1, lambda x: 1,lambda x, y: -1, lambda x, y: 1)[0]

	print("Eout: {}\nbias: {}\nvar: {}".format(Eout,bias,var))


	plt.show()



Eout: 0.5333333333333333
bias: 0.19569357380857
var: 0.33338149781948495

