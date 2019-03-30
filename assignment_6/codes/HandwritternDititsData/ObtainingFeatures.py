import matplotlib.pyplot as plt
import numpy as np

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
		intensity.append(temp)
	return intensity

def get_symmetry(data):
	symmetry = []
	for digit in data:
		temp = 0
		for i in range(0,256,16):
			for j in range(i, i+8):
				if digit[j] == digit[2*i+15-j]:
					temp += 1
		print(temp)
		symmetry.append(temp)
	return symmetry

if __name__ == '__main__':
 	

	matrix = np.loadtxt('ZipDigits.train', usecols=range(257))
	count = 0
	for i in matrix:
		if i[0] == 1.0 or i[0] == 5.0:
			count += 1

	m = np.zeros((count, 257), dtype=float)

	index = 0
	for i in range(0, matrix.shape[0]):
		if matrix[i][0] == 1.0 or matrix[i][0] == 5.0:
			m[index] = matrix[i]
			index += 1


	digit = m[:, 0]
	grayscale = m[:, 1:]
	img1 = []
	img5 = []



	for i in range(0, grayscale.shape[0]):
		if digit[i] == 1.0:
			img = np.resize(grayscale[i], (16, 16))
			img1.append(img)
		else:
			img = np.resize(grayscale[i], (16, 16))
			img5.append(img)
	temp1 = img1[3]
	temp5 = img5[3]
	for i in range(0,16):
		for j in range(0,16):
			temp1[i][j] *= -1
			temp5[i][j] *= -1

	plt.imshow(temp1, cmap='gray')
	plt.show()
	plt.imshow(temp5, cmap='gray')
	plt.show()


	'''
	train1, train5 = read('ZipDigits.train')
	
	# part (a)
	x = [i for i in range(1,257)]
	y1 = train1[0]
	y2 = train5[0]
	plt.scatter(x, y1, s = 3, c = 'red')
	plt.show()
	plt.scatter(x, y2, s = 3, c = 'green')
	plt.show()
	
	# part (b) and (c)
	intensity1 = get_intensity(train1)
	intensity5 = get_intensity(train5)
	symmetry1 = get_symmetry(train1)
	symmetry5 = get_symmetry(train5)

	plt.scatter(intensity1, symmetry1, s = 12, c = '', edgecolor = 'blue', marker = 'o', label = 'digit 1')
	plt.scatter(intensity5, symmetry5, s = 12, c = 'red', marker = 'x', label = 'digit 5')
	plt.legend()
	plt.xlabel('intensity')
	plt.ylabel('symmetry')
	plt.show()
	'''