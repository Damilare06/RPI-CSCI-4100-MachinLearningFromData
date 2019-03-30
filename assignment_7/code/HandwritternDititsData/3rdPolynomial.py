import numpy as np
from matplotlib import pyplot as plt
import scipy

import math
from numpy.linalg import inv

def showimage(arr):
    data = np.zeros( (16,16,3), dtype=np.uint8)
    for i in range(0, 16):
        for j in range(0, 16):
            greyscale = arr[i][j]
            data[i][j] = [greyscale,greyscale,greyscale]
    
    img = plt.imshow(data, interpolation='nearest')
    
    plt.show()
   
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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
 
def sign(x):
    if x>0:
        return 1
    else:
        return -1

def CountError(x,w):
    n=x.shape[1]-1
    count=0
    for i in x:
        if sign(np.inner(i[:n],w))*i[-1]<0:
            count+=1
    return count


def PocketPLA(x,k,maxnum):
    m,n=x.shape
    n-=1
    w=np.zeros(n)
    w0=np.zeros(n)
    error=CountError(x,w)
    Error=[]
    if error==0:
        pass
    else:
        j=0
        while (j<maxnum or error==0):
            k=np.random.randint(0,m)
            i=x[k]
            w=w0+k*i[-1]*i[:n]
            error1=CountError(x,w)
            if error>error1:
                w0=w[:]
                error=error1
            Error.append(error)
            j+=1
    return w0,Error

trainfile = np.loadtxt('train.txt')
testfile = np.loadtxt('test.txt')
 
true_arr = []
false_arr = []
true_arr2 = []
false_arr2 = []
count_train = 0
count_test = 0

for picdata in trainfile:
    if (picdata[0] == 1 or picdata[0] == 5):
        count_test += 1
        
        data = picdata[1:]
        arr = parsearr(data)
        
        intensity = sum(data)/len(data)
        symmetry = comparearrs(arr, horizontalreflect(arr))/2 + (comparearrs(arr, verticalreflect(arr)))/2
           
        if (picdata[0] == 1):
             
            true_arr.append([symmetry,intensity])
            
        elif (picdata[0] == 5):
              
            false_arr.append([symmetry,intensity])
            
for picdata in testfile:
    
    if (picdata[0] == 1 or picdata[0] == 5):
        count_train += 1
        
        data = picdata[1:]
        #showimage(parsearr(data)) 
        arr = parsearr(data)    
        intensity = sum(data)/len(data)
        symmetry = comparearrs(arr, horizontalreflect(arr))/2 + (comparearrs(arr, verticalreflect(arr)))/2    
        if (picdata[0] == 1):
          
            true_arr2.append([symmetry,intensity])
            
        elif (picdata[0] == 5):
              
            false_arr2.append([symmetry,intensity]) 
            
  
def transform(data):
    result=[]
    for i in data:
        x1 = i[1]
        x2 = i[2]
        flag = i[-1]
        x=[1,x1,x2,x1*x2,x1**2,x2**2,x1**3,(x1**2)*x2,x1*(x2**2),x2**3,flag]
        result.append(x)
    return np.array(result)

def f(x, y, w):
    return w[0]+w[1]*x+w[2]*y+w[3]*x*y+w[4]*(x**2)+w[5]*(y**2)+w[6]*(x**3)+w[7]*(x**2)*y+w[8]*x* (y**2)+w[9]*y**3

 
X1=[i[0] for i in true_arr]
Y1=[i[1] for i in true_arr]

X2=[i[0] for i in false_arr]
Y2=[i[1] for i in false_arr]

plt.scatter(X1, Y1,c='blue', s=10,linewidths=0)        
plt.scatter(X2, Y2,c='r',marker = 'x', s=1, linewidths=10)

x1=[[1]+i+[1] for i in true_arr]
x2=[[1]+i+[-1] for i in false_arr]
data=x1+x2
data=np.array(data)
np.random.shuffle(data)
newdata=transform(data)

Xnew=newdata[:,:-1]
Ynew=newdata[:,-1]
n = 2000
t = 1

x = np.linspace(-t, t, n)
y = np.linspace(-t, t, n)

w1=inv(Xnew.T.dot(Xnew)).dot(Xnew.T).dot(Ynew)
X, Y = np.meshgrid(x, y)

plt.contour(X, Y, f(X, Y, w1), 1, colors = 'orange')
plt.scatter(X1, Y1,c='blue', s=10,linewidths=0)        
plt.scatter(X2, Y2,c='r',marker = 'x', s=1, linewidths=10)
plt.axis([-0.2, 1, -1, 1]) 
plt.title('Train Result')
plt.xlabel("Intensity")
plt.ylabel("Symmetry")
plt.show()


X1=[i[0] for i in true_arr2]
Y1=[i[1] for i in true_arr2]
X2=[i[0] for i in false_arr2]
Y2=[i[1] for i in false_arr2]
plt.contour(X, Y, f(X, Y, w1), 1, colors = 'orange')
plt.scatter(X1, Y1,c='blue', s=10,linewidths=0)        
plt.scatter(X2, Y2,c='r',marker = 'x', s=1, linewidths=10)
plt.axis([-0.2, 1, -1, 1]) 
plt.title('Test Result')
plt.xlabel("Intensity")
plt.ylabel("Symmetry")
plt.show()
error = CountError(newdata,w1)
print(error)
print('Ein:'+str(float(CountError(newdata,w1))/float(newdata.shape[0])))
print('Ein BOUND:'+ str(float(math.sqrt(1/float(2*newdata.shape[0]) * math.log(2*len(Y)/0.05)))))

x1=[[1]+i+[1] for i in true_arr2]
x2=[[1]+i+[-1] for i in false_arr2]
data=x1+x2
data=np.array(data)
np.random.shuffle(data)
newdata=transform(data)
print('Ein:'+str(float(CountError(newdata,w1))/float(newdata.shape[0])))
print('Etest BOUND:'+ str(float(math.sqrt(1/float(2*newdata.shape[0]) * math.log(2*len(Y)/0.05)))))

print(len(Y))