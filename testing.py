import pickle, filters, function
import numpy as np
import pandas as pd
import NumPyCNN as numpycnn
import face_detect as fd

def testing(l0,y):
    l1 = function.nonlin(np.dot(l0, syn0))
    l2 = function.nonlin(np.dot(l1, syn1))
    # acc = l2[np.argmax(l2)] / np.sum(l2) * 100
#     print('real : ',np.argmax(y),' pred : ',np.argmax(l2),'', round(acc,2) ,'%')
    return y - np.round(l2,2)
    # if(np.argmax(y) == np.argmax(l2)):
    #     return 1
    # return 0

# data = pd.read_csv('mnist_test.csv', header = None)
# y = function.labelling(data.iloc[:,0].values, 10)
# data = data.iloc[:,1:].values
data = fd.data
y = fd.label
length = len(data)

# get Synapse
syn0 = open("syn0.pickle", "rb")
syn0 = pickle.load(syn0)
syn1 = open("syn1.pickle", "rb")
syn1 = pickle.load(syn1)

# Testing and Counting truth Rate
epoch = length
benar = 0
runn = 0
for i in range(epoch):
    ri = np.random.randint(length)
    # singleData = np.reshape(data[ri], (-1, 28)) # reshape into 28 x 28
    singleData = data[ri]
    l1_feature_map = numpycnn.conv(singleData, filters.filter)
    # ReLu layer
    l1_feature_map_relu = numpycnn.relu(l1_feature_map)
    # Pooling Layer
    l1_feature_map_relu_pool = numpycnn.pooling(l1_feature_map_relu, 2, 2)
    
    # Forward Propagation
    final = np.array([l1_feature_map_relu_pool.ravel()])
    print(testing(final.ravel(), y[ri]))
    # benar += testing(final.ravel(),y[ri])
    # runn += 1
    # print('process : ',round(i/epoch*100,2),'% truth rate : ', round(benar / runn * 100,2), '%')
