from random import randint
import math
import matplotlib.pyplot as plt
import numpy as np

def experiement():
	n_coin = 1000
	times = 10
	coins = []
	for i in range(0, n_coin):
		n_head = 0
		for j in range(0, times):
			flip = randint(0,1)
			if flip == 1:
				n_head += 1
		coins.append(n_head)
	c_1 = coins[0]
	c_rand = coins[randint(0,n_coin-1)]
	c_min = min(coins)
	# print("v_1: {}\nv_rand: {}\nv_min: {}".format(c_1, c_rand, c_min))
	return c_1, c_rand, c_min


def Hoeffding(R_1, R_rand, R_min, n_exp):
	epsilon = np.arange(0,2,0.01)
	Z_1 = np.zeros(len(epsilon))
	Z_rand = np.zeros(len(epsilon))
	Z_min = np.zeros(len(epsilon))

	for i in range(len(epsilon)):
		for j in range(11):
			if abs((j-5)/10.0)>epsilon[i]:
				Z_1[i] += R_1[j]
				Z_rand[i] += R_rand[j]
				Z_min[i] += R_min[j]

	Z_1 = Z_1/n_exp
	Z_min = Z_min/n_exp
	Z_rand = Z_rand/n_exp



	hoeffding = np.array([2*math.exp(-2*(i**2)*10) for i in epsilon])

	plt.plot(epsilon,hoeffding,label="Hoeffding Value")
	plt.plot(epsilon,Z_1,label="P(|v_1 - u| > e)")
	plt.plot(epsilon,Z_rand,label="P(|v_rand - u| > e)")
	plt.plot(epsilon,Z_min,label="P(|v_min - u| > e)")
	plt.xlabel('epsilon')
	plt.ylabel('probability')
	plt.legend()
	plt.show()



if __name__ == '__main__':
	n_exp = 10000
	R_1 = [0]*11
	R_rand = [0]*11
	R_min = [0]*11
	for i in range(0, n_exp):
		c_1, c_rand, c_min = experiement()
		R_1[c_1] += 1
		R_rand[c_rand] += 1
		R_min[c_min] += 1
		if i%100 == 0:
			print(i)

	plt.bar(range(11),R_1)
	plt.title("Result of c_1")
	plt.show()

	plt.bar(range(11),R_rand)
	plt.title("Result of c_rand")
	plt.show()

	plt.bar(range(11),R_min)
	plt.title("Result of c_min")
	plt.show()

	Hoeffding(R_1, R_rand, R_min, n_exp)



	'''
	rng = np.random.RandomState(10)
	a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))
	plt.hist(a, bins='auto')  # arguments are passed to np.histogram
	plt.title("Histogram with 'auto' bins")
	plt.show()
	'''