import numpy as np
import math
import matplotlib.pyplot as plt
import heapq


def generate(rad,thk,sep,n,x1=0,y1=0):
    # top origin
    X1=x1
    Y1=y1

    # bot origin
    X2=X1+rad+thk/2
    Y2=Y1-sep

    # points
    top=[]
    bottom=[]
    
    r1=rad+thk
    r2=rad
    
    cnt=1
    while(cnt<=n):
        x=np.random.uniform(-r1,r1)
        y=np.random.uniform(-r1,r1)
        d=x**2+y**2
        if(d>=r2**2 and d<=r1**2):
            if (y>0):
                top.append([X1+x,Y1+y])
                cnt+=1
            else:
                bottom.append([X2+x,Y2+y])
                cnt+=1
        else:
            continue
    return top,bottom


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


if __name__ == '__main__':
	k = 3
	rad = 10
	thk = 5
	sep = 5
	top, bottom = generate(rad,thk,sep,1000)

	x1True = [float(i[0]) for i in top]
	x2True = [float(i[1]) for i in top]
	x1False = [float(i[0]) for i in bottom]
	x2False = [float(i[1]) for i in bottom]

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

	for i in np.arange(-20, 30, 0.5):
		for j in np.arange(-25, 18, 0.5):
			if knn([i,j], X, Y, k):
				x1true.append(i)
				x2true.append(j)
			else:
				x1false.append(i)
				x2false.append(j)

	plt.scatter(x1true, x2true, s = 20, color = 'orange', label = 'h(x) = +1')
	plt.scatter(x1false, x2false, s = 20, color = 'pink', label = 'h(x) = -1')
	plt.scatter(x1True, x2True, s = 7, marker = 'o', color = 'blue', label = 'y = +1')
	plt.scatter(x1False, x2False, s = 7, marker = 'x', color = 'red', label = 'y = -1')
	plt.legend()
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.title('{}NN Plot'.format(k))
	plt.show()