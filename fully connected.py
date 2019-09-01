import numpy as np
import pandas as pd

def nonlin(x, deriv=False):
    if (deriv == True):
        return (x * (1 - x))
    return 1 / (1 + np.exp(-x))

def testing(l0,y):
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    acc = l2[np.argmax(l2)] / np.sum(l2) * 100
#     print('real : ',np.argmax(y),' pred : ',np.argmax(l2),'', round(acc,2) ,'%')
    if(np.argmax(y) == np.argmax(l2)):
        return 1
    return 0

def labelling(label, dim):
    leng = len(label)
    array = np.zeros((leng,dim))
    for l in range(leng):
        for i in range(dim):
            if(label[l] == i):
                array[l][i] = 1        
    return array

def replaceone(x):
    array = np.zeros((len(x), len(x[0])))
    for i in range(len(x)):
        for j in range(len(x[0])):
            if(x[i][j] > 0):
                array[i][j] = 1
    return array

data = np.array([[3,   1.5],
        [2,   1],
        [4,   1.5],
        [3,   1],
        [3.5, 0.5],
        [2,   0.5],
        [5.5, 1],
        [1,   1]])

data = pd.read_csv('mnist_test.csv', header = None)
y = labelling(data.iloc[:,0].values, 10)
data = data.iloc[:,1:].values
data = replaceone(data)
# target bisa diisi yaw roll pitch
# y = np.array([[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0]])

np.random.seed(1)

syn0 = 2 * np.random.random((784,16)) - 1
syn1 = 2 * np.random.random((16,10)) - 1

epoch = 50 * len(data)
for j in range(epoch):
    #layers
    ri = np.random.randint(len(data))
    l0 = np.array([data[ri]])
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    
    #backpropagation
    l2_error = np.array([y[ri]]) - l2
    l2_delta = l2_error * nonlin(l2, deriv = True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv = True)

    # updating Synapses
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    if(epoch % 100 == 0):
        print(j/epoch*100,'%')
benar = 0
for i in range(epoch):
    ri = np.random.randint(len(data))
    benar += testing(data[ri],y[ri])
    print('truth rate : ', benar / epoch * 100, '%')
