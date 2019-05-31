import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import sklearn
import sklearn.datasets
import scipy.io
from testCases import * 
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters

train_X, train_Y, test_X, test_Y = load_2D_dataset()

print("train_X : " + str(train_X.shape))
print("train_Y : " + str(train_Y.shape))
print("test_X : " + str(test_X.shape))
print("test_Y : " + str(test_Y.shape))

def forward_propagation_with_dropout(X, parameters, keep_prob):
    
    np.random.seed(1)
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = D1 < keep_prob
    A1 = A1 * D1
    A1 = A1 / keep_prob
    
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob
    
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = (1. / m) * np.dot(dZ3, A2.T)
    db3 = (1. / m) * np.sum(dZ3, axis = 1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dA2 = dA2 * D2
    dA2 = dA2 / keep_prob
    
    dZ2 = np.multiply(dA2, np.int64(A2 > 0)) # if A2 > 0, np.int64() return 1, else return 0
    dW2 = (1. / m) * np.dot(dZ2, A1.T)
    db2 = (1. / m) * np.sum(dZ2, axis = 1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1 * D1
    dA1 = dA1 / keep_prob
    
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1. / m ) * np.dot(dZ1, X.T)
    db1 = (1. / m) * np.sum(dZ1, axis = 1, keepdims = True)
    
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    
    
    return gradients




def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
    
    grads = {}
    costs = []
    
    m = X.shape[1]
    
    layer_dims = [X.shape[0], 20, 3, 1]
    
    parameters = initialize_parameters(layer_dims)
    
    
    for i in range(num_iterations):
        
        
        if keep_prob == 1:
            
            A3, cache = forward_propagation(X, parameters)
            
        elif keep_prob < 1:
            
            A3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        
        if lambd == 0:
            
            cost = compute_cost(A3, Y)
            
        else:
            
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
            
        
        assert(lambd == 0 or keep_prob == 1) 
        
        if lambd == 0 and keep_prob == 1:
            
            grads = backward_propagation(X, Y, cache)
            
        elif lambd != 0:
            
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
            
        elif keep_prob < 1:
            
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 1000 == 0:
            
            print("Cost After iterations {} : {}".format(i, cost))
            
            costs.append(cost)
            
    plt.plot(costs)
    plt.title("Learning rate = " + str(learning_rate))
    plt.ylabel("Cost")
    plt.xlabel("iterations(x1,000)")
    #plt.show()
    
    plt.savefig('/home/guanlingh/local2server/homework6/image/h6-3.jpg')
    
    return parameters

parameters = model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3)
print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)





