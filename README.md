# Neural-Network for Final Project

https://matthewmazur.files.wordpress.com/2018/03/neural_network-7.png

# 3 layer NN already tested with NMIST number dataset
- Layer 1  : 784
- Layer 2  : 16
- Layer 3  : 10
- Accuracy : 95,098 % - 98,007 %
- Tested on September 2, 2019

# updated to the 6 layer CNN already tested with mnist dataset
- Dataset  : 784 parsed into 28 x 28 image
- Layer 1  : Convoluted into 26 x 26 with 3 x 3 x 3 filter / kernel
- Layer 2  : Relu Layer
- Layer 3  : Pooling into 13 x 13 x 3
- Layer 4  : FullConn 507 nodes
- Layer 5  : FullConn 16 nodes
- Layer 6  : Softmax 10

# How to Use Neural Network
- run the fully connected.py file for training the synapses based on input data
- run the testing.py for the testing section or see the result of training

# Drawbacks
- have limitation in activation sigmoid function
- rescale the higher number than one become one