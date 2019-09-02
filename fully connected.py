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

data = pd.read_csv('mnist_train.csv', header = None)
y = labelling(data.iloc[:,0].values, 10)
data = data.iloc[:,1:].values
# data = replaceone(data)
# target bisa diisi yaw roll pitch
# y = np.array([[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0]])

np.random.seed(1)
# synapse
syn0 = 2 * np.random.random((507,16)) - 1
syn1 = 2 * np.random.random((16,10)) - 1

# filter / kernel
l1_filter = np.zeros((3,3,3))
l1_filter[0, :, :] = np.array([[[-1, 0, 1], 
                                [-1, 0, 1], 
                                [-1, 0, 1]]])
l1_filter[1, :, :] = np.array([[[1,   1,  1], 
                                [0,   0,  0], 
                                [-1, -1, -1]]])
l1_filter[2, :, :] = np.array([[[1,   0,  1], 
                                [0,   1,  0], 
                                [1,   0,  1]]])
length = len(data)
print(length)
epoch = 1 * length
for j in range(epoch):
    # print(j)
    ri = np.random.randint(length)
    # convoluting layer
    singleData = np.reshape(data[ri], (-1, 28)) # reshape into 28 x 28
    l1_feature_map = numpycnn.conv(singleData, l1_filter)
    # ReLu layer
    l1_feature_map_relu = numpycnn.relu(l1_feature_map)
    # Pooling Layer
    l1_feature_map_relu_pool = numpycnn.pooling(l1_feature_map_relu, 2, 2)
    
    # Forward Propagation
    l0 = replaceone(np.array([l1_feature_map_relu_pool.ravel()]))
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    
    # backpropagation
    l2_error = np.array([y[ri]]) - l2
    l2_delta = l2_error * nonlin(l2, deriv = True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv = True)

    # updating Synapses
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    if(j % 10 == 0):
        print(j/epoch*100,'%',np.amax(l0),np.amax(l1),np.amax(l2))
# Testing and Counting truth Rate
benar = 0
for i in range(epoch):
    ri = np.random.randint(length)
    singleData = np.reshape(data[ri], (-1, 28)) # reshape into 28 x 28
    l1_feature_map = numpycnn.conv(singleData, l1_filter)
    # ReLu layer
    l1_feature_map_relu = numpycnn.relu(l1_feature_map)
    # Pooling Layer
    l1_feature_map_relu_pool = numpycnn.pooling(l1_feature_map_relu, 2, 2)
    
    # Forward Propagation
    final = replaceone(np.array([l1_feature_map_relu_pool.ravel()]))
    benar += testing(final.ravel(),y[ri])
    print('process : ',i/epoch*100,'% truth rate : ', benar / epoch * 100, '%')
