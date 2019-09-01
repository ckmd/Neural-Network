import numpy as np

def activation(x):
    return 1 / (1 + np.exp(-x))
    # return np.round(act,2)

def activationDerivative(x):
    return (x * (1 - x))

def relu(x):
    return np.maximum(x,0)

def which_flower(data):
    outh1 = activation(np.dot(data, hlayer1) + bias1)
    outo1 = activation(np.dot(outh1, softmax) + bias2)
    print(data,'-',outo1)

# untuk belajar
bias1 = 0.35
bias2 = 0.6
# bias3 = [0.1,0.0]

hlayer1 = np.array([
    [0.15, 0.2],[0.25,0.3]    
])
# hlayer2 = np.array([
#     [0.3, -0.1],[0.2, 0.1]
# ])
softmax = np.array([
    [0.4, 0.45],[0.5,0.55]
])
# hlayer1 = np.random.rand(2,2) # node input, node akhir
# hlayer2 = np.random.rand(2,2) # node input, node akhir
# softmax = np.random.rand(2,2) # node input, node akhir

# bias1 = np.random.rand()
# bias2 = np.random.rand()
bias3 = np.random.rand()

input = np.array([[0.05,0.1],[2,2],[1,2],[2,1],[5,5],[6,6],[6,5],[5,6]])
# y = np.array([[0.01,0.99],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1]])
y = np.array([[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0]])

data = [[3,   1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5, 0.5, 1],
        [2,   0.5, 0],
        [5.5, 1,   1],
        [1,   1,   0]]

learningRate = 0.5
# Training Process
for i in range(100000):
    ri = np.random.randint(len(data))
    # temporary synapse
    syn2 = softmax
    # ForwardProp
    outh0 = data[2][:2] # dapatkan indeks 0 dan 1
    outh1 = activation(np.dot(outh0, hlayer1) + bias1)
    # h2Result = activation(np.dot(outh1, hlayer2) + bias2)
    outo1 = activation(np.dot(outh1, softmax) + bias2)
    totalError = np.sum(((y[2] - outo1)**2)/2)
    # Backprop
    # update softmax network
    dz_totalError = outo1 - y[2]
    dz_outo1 = activationDerivative(outo1)
    delta_outo1 = dz_totalError * dz_outo1 * outh1
    # delta_outo1 = 0.082167041 # dari contoh
    softmax -= learningRate * delta_outo1

    # update hidden network 1    
    dz_H1error = dz_totalError * dz_outo1
    dz_H1totalError = np.sum(dz_H1error * syn2, axis = 1)
    dz_outh1 = activationDerivative(outh1)
    delta_h1 = dz_H1totalError * dz_outh1 * outh0
    hlayer1 -= learningRate * delta_h1
    if(i % 10000 == 0):
        print(totalError)
    # print(np.argmax(outo1))
    # softmaxDelta = totalError * activationDerivative(outo1)

    # h2Error = np.dot(softmaxDelta,softmax.T)
    # h2Delta = h2Error * activationDerivative(h2Result)

    # h1Error = np.dot(h2Delta,hlayer2.T)
    # h1Delta = h1Error * activationDerivative(outh1)

    # # updating synapse
    # softmax += np.dot(h2Result.T,softmaxDelta)
    # hlayer2 += np.dot(outh1.T,h2Delta)
    # hlayer1 += np.dot(outh0.T,h1Delta)

# Testing Process
# for i in range(50):
#     ri = np.random.randint(len(data))
which_flower(data[2][:2])
which_flower(data[4][:2])
which_flower(data[5][:2])
which_flower(data[7][:2])