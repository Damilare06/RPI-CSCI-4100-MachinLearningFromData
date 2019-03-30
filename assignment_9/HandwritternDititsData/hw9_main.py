import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from scipy.special import legendre
from numpy.linalg import inv




def L(k,x):
    if(k==0):
        return 1
    elif(k==1):
        return x
    else:
        return (2*k-1)/k*(x*L(k-1,x))-(k-1)/k*L(k-2,x)

def eight_order_L(x1, x2, weight,order):
    one_data = []
    result = 0
    index = 0
    for i in range(order + 1):
        for j in range(order - i + 1):
            # output = L(i,element[1])* L(j, element[2])
            result += weight[index] * L(i, x1) * L(j, x2)
            index += 1
    # print(result)
    return  result



def split_data(data_set):
    index = np.array(random.sample(range(len(data_set)), 300))
    test_index = np.delete(np.arange(len(data_set)), index)
    train_data = data_set[index]
    test_data = data_set[test_index]
    return train_data, test_data


def normalize(data_set):
    # transfer feature one by one
    data_set = data_set.astype(np.float32)
    for i in range(1, len(data_set[0])):
        max1 = np.max(data_set[:,i])
        min1 = np.min(data_set[:,i])
        diff = np.max(data_set[:,i]) + np.min(data_set[:,i])
        print(diff)
        data_set[:,i] = 1.0*(data_set[:,i] - min1 - (max1-min1)/2) / ((max1-min1)/2)
    return data_set


def data_process(files):
    digit1, not_digit1 = [], []
    for file in files:
        raw_data = np.loadtxt(file)
        data = raw_data[:, 1:]
        # target_vector = raw_data[index, 0]
        for index in range(len(data)):
            # print(index)
            number = data[index].reshape((16, 16))
            # cv2.imshow("test", number)
            # cv2.waitKey(0)
            # feature 1. whether vertical symmetric
            number_flip = cv2.flip(number, 0)
            # more count means more unsymmetrical
            count = len(np.where(number != number_flip)[0])
            # number_final = number_flip - number

            # this is feature for pixel range
            # pixel_index = np.where(number > -1.0)[1]
            # pixel_range = max(pixel_index) - min(pixel_index)

            intensity = len(np.where(number > -1.0)[0])

            digit1.append([int(raw_data[index, 0]) == 1, count, intensity])
            # else:
            #     digit1.append((0, count, pixel_range))
    # not_digit1 = np.array(not_digit1)
    digit1 = np.array(digit1)
    return digit1 # , not_digit1


def feature_transform(train_data, order):
    new_data = []
    number = 0
    for element in train_data:
        one_data = []
        for i in range(order+1):
            for j in range(order-i+1):
                # output = L(i,element[1])* L(j, element[2])
                one_data.append( L(i,element[1])* L(j, element[2]) )
        new_data.append(one_data)
    return new_data

def plot_dicision_boundary(data_matrix,train_data,all_data ,order,withRegu, regularization=0 ):
    Z = data_matrix
    ZT = data_matrix.T
    ZTZ = ZT.dot(data_matrix)
    ZTZI = inv(ZTZ + regularization * np.identity(ZTZ.shape[0]))
    weight = ZTZI.dot(ZT).dot(train_data[:,0])
    # print("this is sum of weight", np.sum(weight))

    # print(weight)
    x1 = np.arange(-1, 1, 0.01)
    x2 = np.arange(-1, 1, 0.01)
    # fig = plt.figure()
    # fig.add_subplot(1, 2, 1)
    x1, x2 = np.meshgrid(x1, x2)
    plt.contour(x1, x2, eight_order_L(x1, x2, weight, order),levels=[0])
    # print(train_data[np.where(train_data[:,0]==-1), 1])

    plt.scatter(train_data[np.where(train_data[:, 0] == 1), 1][0], train_data[np.where(train_data[:, 0] == 1), 2][0],color="Blue")
    plt.scatter(train_data[np.where(train_data[:,0]==-1), 1][0], train_data[np.where(train_data[:,0]==-1), 2][0] , color="Red")
    plt.title("Eighth order Legendre polynomials model to seperate of {} data {}".format("training", withRegu))
    plt.xlabel("symmetric")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()

    # fig.add_subplot(1, 2, 2)
    plt.scatter(all_data[np.where(all_data[:,0]==1), 1][0], all_data[np.where(all_data[:,0]==1), 2][0] , color="Blue")
    plt.scatter(all_data[np.where(all_data[:,0]==-1), 1][0], all_data[np.where(all_data[:,0]==-1), 2][0] , color="Red")
    plt.contour(x1, x2, eight_order_L(x1, x2, weight, order), levels=[0])
    plt.title("Eighth order Legendre polynomials model to seperate of {} data {}".format("all", withRegu))
    plt.xlabel("symmetric")
    plt.ylabel("Intensity")
    plt.show()


def cross_validation(data_matrix1,test_data_matrix1, train_data1,test_data1):
    # for every lambda from 0 ~ 2, I can get one E_cv and E_test

    # regularization = np.arange(0, 0.00001, 10**-8)
    regularization = np.arange(0.01,2,0.01)
    E_cv_min = 999
    lambda_min = 999
    train_error = []
    test_error = []
    error = []
    data = [(data_matrix1, train_data1), (test_data_matrix1, test_data1)]
    for constant in regularization:
        for index in range(2) :
            (data_matrix, train_data) =data[index]
            Z = data_matrix
            ZT = data_matrix.T
            ZTZ = ZT.dot(data_matrix)
            ZTZI = inv(ZTZ + constant * np.identity(ZTZ.shape[0]) )
            H = Z.dot(ZTZI).dot(ZT)
            # pseudo_inverse = (temp + constant * np.identity(temp.shape[0]) ).dot(data_matrix.T)

            # W_reg = H.dot(test_data[:,0])
            # E_aug =

            y_prediction = H.dot(train_data[:,0])
            E_cv = 0
            error_with_lambda = []
            for i in range(len(train_data[:,0])):
                E_cv += ( (y_prediction[i] - train_data[i,0]) / (1-H[i,i]) ) ** 2
                # temp1 = (y_prediction[i] - train_data[i,0])
                # temp2 = (1-H[i,i])
            E_cv = E_cv / len(train_data[:,0])
            # if E_cv < E_cv_min:
            #     E_cv_min = E_cv
            #     lambda_min = constant
            if E_cv < E_cv_min and index == 0:
                E_cv_min = E_cv
                lambda_min = constant
                print("Lambda_min is ", lambda_min)
            print(E_cv)
            # if index:
            error.append((index, constant, E_cv))
            # else:
            #     train_error.append((constant,E_cv))
        #     print(E_cv, constant)
        # print()
        # print()
    error = np.array(error)
    plt.scatter(error[np.where(error[:, 0] == 0)][:,1],error[np.where(error[:, 0] == 0)][:,2], color="Blue", label='training')
    plt.scatter(error[np.where(error[:, 0] == 1)][:,1],error[np.where(error[:, 0] == 1)][:,2], color="red", label="testing")
    plt.title("E_cv versus lambda and E_test versus lambda")
    plt.legend()
    plt.show()

    return lambda_min


def  new_cross_validation(data_matrix1,test_data_matrix, train_data1,test_data):
    # for every lambda from 0 ~ 2, I can get one E_cv and E_test

    # regularization = np.arange(0, 0.00001, 10**-8)
    regularization = np.arange(0.01,2,0.02)
    E_cv_min = 999
    lambda_min = 999
    error = []
    for constant in regularization:
        print(len(data_matrix1[:,0]))
        for i in range(len(data_matrix1[:,0])):
            test_data_matrix = np.array([data_matrix1[i,:]])
            test_data = np.array([train_data1[i,:]])
            train_data_matrix = data_matrix1[:,:]
            train_data = train_data1[:,:]

            train_data_matrix = np.delete(train_data_matrix,i, 0)
            train_data = np.delete(train_data,1,0)
            data = [(train_data_matrix, train_data), (test_data_matrix, test_data)  ]
            for index in range(2) :
                (data_matrix, train_data) =data[index]
                Z = data_matrix
                ZT = data_matrix.T
                ZTZ = ZT.dot(data_matrix)
                ZTZI = inv(ZTZ + constant * np.identity(ZTZ.shape[0]) )
                H = Z.dot(ZTZI).dot(ZT)
                # pseudo_inverse = (temp + constant * np.identity(temp.shape[0]) ).dot(data_matrix.T)

                # W_reg = H.dot(test_data[:,0])
                # E_aug =

                y_prediction = H.dot(train_data[:,0])
                E_cv = 0
                error_with_lambda = []
                for i in range(len(train_data[:,0])):
                    E_cv += ( (y_prediction[i] - train_data[i,0]) / (1-H[i,i]) ) ** 2
                    # temp1 = (y_prediction[i] - train_data[i,0])
                    # temp2 = (1-H[i,i])
                E_cv = E_cv / len(train_data[:,0])
                if E_cv < E_cv_min and index == 0:
                    E_cv_min = E_cv
                    lambda_min = constant
                    print("Lambda_min is ", lambda_min)
                print(E_cv)
                # if index:
                error.append((index, constant, E_cv))
                # else:
                #     train_error.append((constant,E_cv))
            #     print(E_cv, constant)
            # print()
            # print()
    error = np.array(error)
    plt.scatter(error[np.where(error[:, 0] == 0)][:,1],error[np.where(error[:, 0] == 0)][:,2], color="Blue", label='training')
    plt.scatter(error[np.where(error[:, 0] == 1)][:,1],error[np.where(error[:, 0] == 1)][:,2], color="red", label="testing")
    plt.title("E_cv versus lambda and E_test versus lambda")
    plt.legend()
    plt.show()

    return lambda_min

    pass


if __name__ == '__main__':
    train_file = "ZipDigits.train"
    test_file = "ZipDigits.test"
    order = 8
    data_set = data_process([train_file,test_file])
    ######### normalize feature
    data_set = normalize(data_set)
    data_set[np.where(data_set[:,0]==1 ),0 ] = 1
    data_set[np.where(data_set[:,0]==0 ),0 ] = -1
    train_data, test_data = split_data(data_set)

    data_matrix = np.array(feature_transform(train_data, order))
    test_data_matrix = np.array(feature_transform(test_data, order))

    # plot_dicision_boundary(data_matrix, train_data, data_set, order, "without regularity", 0)

    # with regularization
    # plot_dicision_boundary(data_matrix, train_data, data_set, order, "with regu" , 2)

    # cross validation
    min_lambda = cross_validation(data_matrix,test_data_matrix, train_data,test_data)
    print(min_lambda)

    # new cross validation 299 vs 1
    # min_lambda = new_cross_validation(data_matrix,data_matrix, train_data, test_data)


    # plot_dicision_boundary(data_matrix, train_data, data_set, order, "with min_lambda regu" , min_lambda)
    # print(min_lambda)




    # data_matrix = np.transpose(new_data)





    #
    # x = np.arange(-1, 1, 0.05)
    # y = [[] for i in range(6)]
    # for k in range(6):
    #     y[k] = [L(k, i) for i in x]
    #     plt.plot(x, y[k], label="L" + str(k) + "(x)")
    # plt.legend()
    # plt.title("Legendre Polynomial")
    # plt.show()

