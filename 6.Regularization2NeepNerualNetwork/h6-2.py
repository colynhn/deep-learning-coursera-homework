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

def backward_propagation_with_regularization(X, Y, cache, lambd):
   
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
  
    dW3 = 1. / m * np.dot(dZ3, A2.T) + (lambd * W3) / m

    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))

    dW2 = 1. / m * np.dot(dZ2, A1.T) + (lambd * W2) / m

    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))

    dW1 = 1. / m * np.dot(dZ1, X.T) + (lambd * W1) / m

    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients




def compute_cost_with_regulations(A3, Y, parameters, lambd):
    
    m = Y.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    
    costa = compute_cost(A3, Y)
    
    L2_regulations = lambd *(np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)
    
    
    
    cost_reg = costa + L2_regulations
    
    
    return cost_reg




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
            
            cost = compute_cost_with_regulations(A3, Y, parameters, lambd)
            
        
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
    
    plt.savefig('/home/guanlingh/local2server/homework6/image/h6-2.jpg')
    
    return parameters

parameters = model(train_X, train_Y,lambd = 0.7)
print("Train set : ")
train_prediction = predict(train_X, train_Y, parameters)
print("Test set : ")
Test_prediction = predict(test_X, test_Y, parameters)







