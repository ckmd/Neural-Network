import pickle
import filters
import numpy as np
import pandas as pd
import NumPyCNN as numpycnn

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

def removeOverflow(x):
    array = x
    for i in range(len(x)):
        for j in range(len(x[0])):
            if(x[i][j] > 709):
                array[i][j] = 709
            elif(x[i][j] < -708):
                array[i][j] = -708
    return array

data = pd.read_csv('mnist_test.csv', header = None)
y = labelling(data.iloc[:,0].values, 10)
data = data.iloc[:,1:].values
length = len(data)

# get Synapse
syn0 = open("syn0.pickle", "rb")
syn0 = pickle.load(syn0)
syn1 = open("syn1.pickle", "rb")
syn1 = pickle.load(syn1)

# Testing and Counting truth Rate
epoch = length
benar = 0
for i in range(epoch):
    ri = np.random.randint(length)
    singleData = np.reshape(data[ri], (-1, 28)) # reshape into 28 x 28
    l1_feature_map = numpycnn.conv(singleData, filters.filter)
    # ReLu layer
    l1_feature_map_relu = numpycnn.relu(l1_feature_map)
    # Pooling Layer
    l1_feature_map_relu_pool = numpycnn.pooling(l1_feature_map_relu, 2, 2)
    
    # Forward Propagation
    final = np.array([l1_feature_map_relu_pool.ravel()])
    benar += testing(final.ravel(),y[ri])
    print('process : ',i/epoch*100,'% truth rate : ', benar / epoch * 100, '%')
