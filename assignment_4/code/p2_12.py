import math

n = 2
print(2**2)
while(True):
	# if (8/n*math.log(4*((2*n)**10+1)/0.05))**0.5 > 0.05:
	print(n**2-n+1)
	if n**2-n+1==2**n:

		n += 1
	else:
		break
print(n)