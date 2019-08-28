import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

def which_flower(length, width):
    z = length * w1 + width * w2 + bias
    pred = sigmoid(z)
    print(pred)

def count_error(point):
    z = point[0] * w1 + point[1] * w2 + bias
    pred = sigmoid(z)
    if(pred < 0.5):
        result = 0
    else:
        result = 1
    if(result == point[2]):
        return 1
    else:
        return 0

data = [[3,   1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5, 0.5, 1],
        [2,   0.5, 0],
        [5.5, 1,   1],
        [1,   1,   0]]
mystery = [4.5, 1]

# Initiate random Weight and Bias
w1 = np.random.randn()
w2 = np.random.randn()
bias = np.random.randn()

# Training Process
learning_rate = 0.1
# recomended 100000 for best result
training_times = 100000
for i in range(training_times):
    print('training process : ', (i+1)/training_times*100,'%')
    ri = np.random.randint(len(data))
    point = data[ri]

    z = point[0] * w1 + point[1] * w2 + bias
    # create range between 0 and 1
    pred = sigmoid(z)
    target = point[2]
    # calculating squarefoot of pred - target
    cost = np.square(pred - target)

    # change error and change weight
    # derivative cost
    dcost_pred = 2 * (pred - target)
    # derivative pred
    dpred_dz = sigmoid_p(z)

    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1

    dcost_dz = dcost_pred * dpred_dz

    # chaining process
    dcost_dw1 = dcost_dz * dz_dw1
    dcost_dw2 = dcost_dz * dz_dw2
    dcost_db = dcost_dz * dz_db

    # Updating Weight and Bias
    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    bias = bias - learning_rate * dcost_db

# Testing Process
# which_flower(3.5,0.5)

# counting error rate
true = 0
for i in range(10):
    ri = np.random.randint(len(data))
    point = data[ri]
    true += count_error(point)
    accuracy = true / (i+1) * 100
    print(i+1,'accuracy : ',accuracy,'%')