import numpy as np
import matplotlib.pyplot as plt
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
np.random.seed(1)

# 深层神经网络的参数初始化
def initialize_parameters_deep(layer_dims):
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        
        parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros(shape = (layer_dims[l],1))
        
        
    return parameters

# 计算 Z, cache(A, W, b)
def linear_forward(A,W,b):
    
    
    Z = np.dot(W, A) + b
    
    cache = (A, W, b)
    
    return Z, cache

 # 第1-(L-1)层使用relu, 在第L层使用sigmoid 函数

def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        
    elif activation == "relu":
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
        
    cache = (linear_cache, activation_cache)
    
    return  A, cache  
        
# forward propagation:

def L_model_forward(X, parameters):
    
    caches = []
    A = X
    
    L = len(parameters) // 2
    
    for l in range(1, L):
        
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)],
                                                     parameters["b" + str(l)],
                                                     "relu")
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)],
                                             parameters["b" + str(L)],
                                             "sigmoid")
    
    caches.append(cache)
    
    
    return AL, caches

# compute cost:
def compute_cost(AL, Y):
    
    
    m = Y.shape[1]
    
    loss = np.multiply(Y,np.log(Al)) + np.multiply((1-Y),np,log(1-AL))
    
    cost = -(1 / m) * np.sum(loss)
    
    
    
    return cost

# compute dW, db, dA: 
def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = (1 / m) * np.dot(dZ, cache[0].T)
    db = (1 / m) * np.sum(dZ, axis = 1, keepdims = true)
    dA_prev = np.dot(cache[1].T, dZ)
    
    return dA_prev, dW, db    
# 根据激活函数不同，进行dZ的计算， 进而进行更新dW, db, dA:
def linear_activation_backward(dA, cache, activation):
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        
        dZ = relu_backward(dA, activation_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, linear_cache)
        
        
        
    
        
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db
# backward propagation: 注意代码结构，不好理解:

def L_model_backward(AL, Y, caches):
    
    grads = {}
    L = len(caches)
    
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    
    current_cache = caches[L-1]
    
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(sigmoid_backward(dAL, current_cache[1]),current_cache[0] )
    
    # loop for L-1 layers.
    
    for l in reversed(range(L-1)):
        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_backward(relu_backward(dAL, current_cache[1]),current_cache[0])
        
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + l)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
    return grads

# update parameters:

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2
    
    for l in range(L):
        
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    
    return parameters




