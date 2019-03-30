
def eucildeanDistance(x1, x2):
	return ((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)**0.5

def cosSimilarity(x1, x2):
	top = x1[0]*x2[0] + x1[1]*x2[1]
	bot = ((x1[0]**2+x1[1]**2)*(x2[0]**2+x2[1]**2))**0.5
	return top/bot


if __name__ == '__main__':
	x = [1+2455,1+545]
	y = [1000+2455,1000+545]

	d1 = eucildeanDistance(x, y)
	d2 = cosSimilarity(x, y)
			
	print('vector1: {}, vector2: {},\n\tEucildean Distance: {}\n\tCosine Similarity: {}\n'.format(x, y, d1, d2))