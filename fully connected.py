import numpy as np
import pandas as pd
import NumPyCNN as numpycnn
import time, pickle, filters, function

start = time.time()

data = pd.read_csv('mnist_test.csv', header = None)
y = function.labelling(data.iloc[:,0].values, 10)
data = data.iloc[:,1:].values
# data = replaceone(data)
# target bisa diisi yaw roll pitch
# y = np.array([[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0]])

np.random.seed(1)
# synapse
syn0 = 2 * np.random.random((1014,100)) - 1
syn1 = 2 * np.random.random((100,10)) - 1

length = len(data)
print(length)
epoch = 1 * length
for j in range(epoch):
    # print(j)
    ri = np.random.randint(length)
    # convoluting layer
    singleData = np.reshape(data[ri], (-1, 28)) # reshape into 28 x 28
    l1_feature_map = numpycnn.conv(singleData, filters.l1_filter)
    # ReLu layer
    l1_feature_map_relu = numpycnn.relu(l1_feature_map)
    # Pooling Layer
    l1_feature_map_relu_pool = numpycnn.pooling(l1_feature_map_relu, 2, 2)
    
    # Forward Propagation
    l0 = np.array([l1_feature_map_relu_pool.ravel()])
    l1 = function.nonlin(np.dot(l0, syn0))
    l2 = function.nonlin(np.dot(l1, syn1))
    
    # backpropagation
    l2_error = np.array([y[ri]]) - l2
    l2_delta = l2_error * function.nonlin(l2, deriv = True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * function.nonlin(l1, deriv = True)

    # updating Synapses
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    if(j % 10 == 0):
        current = time.time()
        # print(round((current - start),1),'s',round((j/epoch*100),2),'%')
        print(round((current - start),1),'s',round((j/epoch*100),2),'%', round(np.amin(l0),2), round(np.amax(l0),2), round(np.amin(l1),2), round(np.amax(l1),2), round(np.amin(l2),2), round(np.amax(l2),2))

# save final synapse into pickle
pickle_out = open("syn0.pickle", "wb")
pickle.dump(syn0, pickle_out)

pickle_out = open("syn1.pickle", "wb")
pickle.dump(syn1, pickle_out)

pickle_out.close()

end = time.time()
print(end - start)