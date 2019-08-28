import pandas as pd
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

def count_error(point, w, bias):
    z = bias
    for i in range(len(point) - 1):
        z += point[i] * w[i]
    pred = sigmoid(z)
    if(pred < 0.5):
        result = 0
    else:
        result = 1
    if(result == point[len(point) - 1]):
        return 1
    else:
        return 0

def conv(layer, filter):
    stride = 1
    data = layer[0]    # Layer[0] is data
    label = layer[1]    # layer[1] is label
    convoluted = []
    for i in range(len(data) - len(filter) + stride):
        newRow = []
        for j in range(len(data[i]) - len(filter) + stride):
            newDot = data[i][j]*filter[0][0] + data[i][j+1]*filter[0][1] + data[i][j+2]*filter[0][2] + data[i+1][j]*filter[1][0] + data[i+1][j+1]*filter[1][1] + data[i+1][j+2]*filter[1][2] + data[i+2][j]*filter[2][0] + data[i+2][j+1]*filter[2][1] + data[i+2][j+2]*filter[2][2]
            newRow.append(newDot)
        convoluted.append(newRow)
    convoluted = sum(convoluted,[]) # Merubah 2D list menjadi 1D list
    convoluted.append(label)
    # print(convoluted)
    return convoluted

# Belum Berhasil
def pool(layer, stride):
    data = layer[0]
    label = layer[1]
    pooled = []
    for i in range(len(data)):
        newRow = []
        for j in range(len(data[i])):
            if(i % stride == 0 and j % stride == 0):
                newDot = (data[i][j] + data[i][j+1] + data[i+1][j] + data[i+1][j+1])/4.0
                newRow.append(newDot)
        pooled.append(newRow)
    return pooled

# Reading XO data 5x5, index 0 s.d. 24 is data, index 25 is label
dataset = pd.read_excel('XO Dataset.xlsx')
startrow = 0
arrayOfData = []
for i in range(10):
    endrow = startrow + 5
    data = dataset.iloc[startrow:endrow,:5].to_numpy()#.ravel() # 0->5, 6->11, 12->17 and so on
    label = dataset.iloc[endrow-1,5]
    # label = label[np.logical_not(np.isnan(label))] # removing nan values
    arrayOfData.append([data,label])
    # print(label)
    startrow += 6

# Read 3x3 convolutional Filter from Xlsx
filter = pd.read_excel('XO Filter.xlsx')
arrayOfFilter = []
start = 0
for i in range(3):
    end = start + 3
    data = filter.iloc[start:end].to_numpy()#.ravel() # 0->3, 4->7 8->11 and so on
    arrayOfFilter.append(data)
    # print(data)
    start += 4

# print(arrayOfData)
# Convolution Layer
convoluted = []
for i in range(len(arrayOfData)):
    for j in range(len(arrayOfFilter)):
        convolve = conv(arrayOfData[i],arrayOfFilter[j])
        convoluted.append(convolve)

# Pooling Layer
# pooled = pool(convoluted,1)
# print(pooled)
# print(convoluted)

# Initiate random Weight and Bias
w = []
bias = np.random.randn()
for i in range(len(convoluted[0])-1):
    w.append(np.random.randn())

print(convoluted[0])

# Training Process
learning_rate = 0.1
# recomended 100000 for best result
training_times = 100000
for i in range(training_times):
    print('training process : ', (i+1)/training_times*100,'%')
    ri = np.random.randint(len(convoluted))
    point = convoluted[ri]
    lengthPoint = len(point) - 1
    z = bias
    for j in range(lengthPoint):
        z += point[j] * w[j]

    # create range between 0 and 1
    pred = sigmoid(z)
    target = point[lengthPoint]
    # calculating squarefoot of pred - target
    cost = np.square(pred - target)

    # change error and change weight
    # derivative cost
    dcost_pred = 2 * (pred - target)
    # derivative pred
    dpred_dz = sigmoid_p(z)

    dcost_dz = dcost_pred * dpred_dz

    dz_dw = [None]*lengthPoint
    dcost_dw = [None]*lengthPoint
    for j in range(len(point) - 1):
        dz_dw[j] = point[j]
        # Chaining Process
        dcost_dw[j] = dcost_dz * dz_dw[j]
        # Updating Weight
        w[j] = w[j] - learning_rate * dcost_dw[j]
    dz_db = 1

    # chaining process
    dcost_db = dcost_dz * dz_db
    # Update Bias
    bias = bias - learning_rate * dcost_db
    # print(bias)

# testing and counting error rate
true = 0
for i in range(10):
    ri = np.random.randint(len(data))
    point = convoluted[ri]
    true += count_error(point, w, bias)
    accuracy = true / (i+1) * 100
    print(i+1,'accuracy : ',accuracy,'%')