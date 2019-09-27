import numpy as np
import pandas as pd
import NumPyCNN as numpycnn
import time, pickle, filters, function
import face_detect as fd

start = time.time()

# data = pd.read_csv('mnist_test.csv', header = None)
# y = function.labelling(data.iloc[:,0].values, 10)
# data = data.iloc[:,1:].values
data = fd.data
y = fd.label
# target bisa diisi yaw roll pitch
# y = np.array([[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0]])

np.random.seed(1)
# synapse
syn0 = 2 * np.random.random((4608,100)) - 1
syn1 = 2 * np.random.random((100,5)) - 1

length = len(data)
print(length)
epoch = 1 * length
for j in range(1):
    # print(j)
    ri = np.random.randint(length)
    # convoluting layer
    # singleData = np.reshape(data[ri], (-1, 28)) # reshape into 28 x 28 pnly for MNIST
    singleData = data[ri]
    l1_feature_map = numpycnn.conv(singleData, filters.filter1)
    l1_feature_map_relu = numpycnn.relu(l1_feature_map)
    l1_feature_map_relu_pool = numpycnn.pooling(l1_feature_map_relu, 2, 2)
    print(l1_feature_map_relu_pool.shape, np.amax(l1_feature_map_relu_pool), np.amin(l1_feature_map_relu_pool))

    feature_input = []
    for conv2 in l1_feature_map_relu_pool.T:
        l2_feature_map = numpycnn.conv(conv2, filters.filter2)
        l2_feature_map_relu = numpycnn.relu(l2_feature_map)
        l2_feature_map_relu_pool = numpycnn.pooling(l2_feature_map_relu, 2, 2)
        print(l2_feature_map_relu_pool.shape, np.amax(l2_feature_map_relu_pool), np.amin(l2_feature_map_relu_pool))

        for conv3 in l2_feature_map_relu_pool.T:
            l3_feature_map = numpycnn.conv(conv3, filters.filter3)
            l3_feature_map_relu = numpycnn.relu(l3_feature_map)
            l3_feature_map_relu_pool = numpycnn.pooling(l3_feature_map_relu, 2, 2)
            print(l3_feature_map_relu_pool.shape, np.amax(l3_feature_map_relu_pool), np.amin(l3_feature_map_relu_pool))

            for conv4 in l3_feature_map_relu_pool.T:
                l4_feature_map = numpycnn.conv(conv4, filters.filter4)
                l4_feature_map_relu = numpycnn.relu(l4_feature_map)
                l4_feature_map_relu_pool = numpycnn.pooling(l4_feature_map_relu, 2, 2)
                print(l4_feature_map_relu_pool.shape, np.amax(l4_feature_map_relu_pool), np.amin(l4_feature_map_relu_pool))

                for conv_final in l4_feature_map_relu_pool.T:
                    feature_input.append(conv_final)

    feature_input = np.array(feature_input).ravel()
    # Forward Propagation
    l0 = np.array([feature_input])
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
    if(j % 1 == 0):
        current = time.time()
        print(round((current - start),1),'s',round((j/epoch*100),2),'%')
        # print(round((current - start),1),'s',round((j/epoch*100),2),'%', round(np.amin(l0),2), round(np.amax(l0),2), round(np.amin(l1),2), round(np.amax(l1),2), round(np.amin(l2),2), round(np.amax(l2),2))

# save final synapse into pickle
pickle_out = open("syn0.pickle", "wb")
pickle.dump(syn0, pickle_out)

pickle_out = open("syn1.pickle", "wb")
pickle.dump(syn1, pickle_out)

pickle_out.close()

end = time.time()
print(end - start)