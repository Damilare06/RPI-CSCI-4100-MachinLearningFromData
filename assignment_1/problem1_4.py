import numpy as np
import matplotlib.pyplot as plt
import math
import random

def create_a_random_line():
    a = random.randint(-5,5)
    b = random.randint(-5,5)
    c = random.randint(-5,5)
    # x = np.arange(-5,5, 0.1)
    # y = 1.0*a * x/b + 1.0/b
    return a,b,c

def check_data(a, b,c, a_g, b_g,c_g,  x, y):
    result = True
    a1 = a * x + b * y + c
    b1 = a_g * x + b_g * y + c_g
    if (a * x + b * y + c > 0 and a_g * x + b_g * y + c_g <= 0 ):
        a_g += x
        b_g += y
        c_g += 1
        result = False
    elif (a * x + b * y + c <= 0 and a_g * x + b_g * y + c_g >= 0 ):
        a_g -= x
        b_g -= y
        c_g -= 1
        result = False

    return result, a_g,b_g,c_g

if __name__ == '__main__':
    difference = 0
    for i in range(10):
        number = 10
        # random.seed(5)
        # np.random.seed(22)
        # use random function to get two random integer
        x = np.arange(-5,5, 0.1)
        a_f,b_f,c_f = create_a_random_line()
        y_f = - (a_f*x + c_f) / b_f
        xdata = np.random.uniform(-5,5,number)
        ydata = np.random.uniform(-5,5,number)
        # a_g,b_g,c_g = create_a_random_line()
        a_g = 0
        b_g = 0
        c_g = 0
        x_pos = []
        y_pos = []
        x_neg = []
        y_neg = []

        for i in range(number):
            if  a_f*xdata[i] + b_f*ydata[i] + c_f > 0 :
                x_pos.append(xdata[i])
                y_pos.append(ydata[i])
            else :
                x_neg.append(xdata[i])
                y_neg.append(ydata[i])

        target_line, = plt.plot(x,y_f, label='target function')
        # plt.text( 1.0*(5+b)/a, 5, "target function")
        # hypothesis_line, = plt.plot(x_g,y_g,"r--",label='hypothesis g')

        plt.scatter(x_neg,y_neg, color = "red")
        plt.scatter(x_pos,y_pos, color = "green")
        plt.axis([-5, 5, -5, 5])
        # plt.legend([target_line,hypothesis_line],["target function f", "hypothesis g"])
        plt.show()

        finish = False
        count = 0
        while not finish:


            # if count == 8:
            # print(b_g)
            y_g = - (a_g * x + c_g)/b_g
            target_line, = plt.plot(x, y_f, label='target function')
            hypothesis_line, = plt.plot(x, y_g, "r--", label='hypothesis g')
            plt.scatter(x_neg, y_neg, color="red")
            plt.scatter(x_pos, y_pos, color="green")
            plt.axis([-5, 5, -5, 5])
            # plt.show()
            # input("enter return")
            plt.close("all")
            count = 0
            for i in range(number):
                result,a_g,b_g,c_g = check_data(a_f, b_f,c_f, a_g, b_g,c_g, xdata[i], ydata[i])
                if result == False:
                    pass
                else:
                    count += 1
            # print(count)
            if count == number:
                break


        y_g = - (a_g * x + c_g)/b_g

        for i in range(len(x)):
            difference += (y_g[i]-y_f[i]) ** 2

        target_line, = plt.plot(x, y_f, label='target function')
        hypothesis_line, = plt.plot(x, y_g, "r--", label='hypothesis g')
        plt.scatter(x_neg, y_neg, color="red")
        plt.scatter(x_pos, y_pos, color="green")
        plt.axis([-5, 5, -5, 5])
        plt.legend([target_line,hypothesis_line],["target", 'hypothesis: {}x + {}y + {} = 0'.format(a_g, b_g, c_g)])
        plt.title("{}th iteration".format(count))
        plt.show()

        plt.close("all")
        print("difference ", difference)
        exit(1)
