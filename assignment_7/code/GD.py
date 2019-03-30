from sympy import *
from matplotlib import pyplot as plt

x = Symbol('x')
y = Symbol('y')
z = Symbol('z')

f = x**2 + 2*y**2 + 2*sin(2*pi*x)*sin(2*pi*y)

# First partial derivative with respect to x
fpx = f.diff(x)

# First partial derivative with respect to y
fpy = f.diff(y)




maxIterations = 50
def GD(x0, y0, rate, maxIterations):
    f = x**2 + 2*y**2 + 2*sin(2*pi*x)*sin(2*pi*y)

    # First partial derivative with respect to x
    fpx = f.diff(x)

    # First partial derivative with respect to y
    fpy = f.diff(y)
    X = []
    Y = []
    out = []
    iterations = 0
    for i in range (maxIterations):
        xn = x0 - rate*fpx.evalf(subs={x:x0, y:y0})
        yn = y0 - rate*fpy.evalf(subs={x:x0, y:y0})
        #If the number of iterations goes up too much, maybe x (and/or y0)
        #is diverging! Let's stop the loop and try to understand.
        iterations += 1
        value = f.evalf(subs={x:xn, y:yn})
        out.append(value)
        X.append(xn)
        Y.append(yn)
        #Simultaneous update
        x0 = xn
        y0 = yn
    m = min(out)
    i = out.index(m)
    print(X[i], Y[i], m)


GD(0.1,0.1,0.01,50)
GD(1,1,0.01,50)
GD(-0.5,-0.5,0.01,50)
GD(-1,-1,0.01,50)
