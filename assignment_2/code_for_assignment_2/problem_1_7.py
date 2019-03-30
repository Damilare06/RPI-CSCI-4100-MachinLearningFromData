import random
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import comb

def Pmax(N,x,u):
	low=int(3-6*x)
	up=int(3+6*x)
	s=0
	for k in range(low,up+1):
		s+=comb(N,k)*(u**k)*((1-u)**(N-k))
	return 1-s**2

x=np.arange(0,1,0.01)
y1=np.array([Pmax(6,i,0.5) for i in x])
y2=np.array([2*math.exp(-2*6*(i**2)) for i in x])

plt.plot(x,y1,label="P[max|v_i-u_i|>e]")
plt.plot(x,y2,label="Hoeffding Value")
plt.xlabel("Eplison")
plt.ylabel("Probability")
plt.legend()
plt.show()

