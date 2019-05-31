import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
np.random.seed(1)

X, Y = load_planar_dataset()

def layer_sizes(X, Y):
    
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
     
    return n_x, n_h, n_y
    
def initialize_parameters(n_x, n_h, n_y):
    
    
    np.random.seed(2)
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape = (n_h,1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape = (n_y, 1))
    
    parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2" : W2,
        "b2" : b2  
    }
    
    return parameters

def forward_propagation(X, parameters):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    
    cache = {
        "Z1" : Z1,
        "A1" : A1,
        "Z2" : Z2,
        "A2" : A2
    }
    
    
    
    return A2, cache

def compute_cost(Y, A2, parameters):
    
    m = Y.shape[1]
    
    
    #W1 = parameters['W1']
    #W2 = parameters['W2']
    #loss = np.multiply(Y, np.log(A2)) + np.multiply(1-Y, np.log(1-A2))
    #cost = - np.sum(loss) / m
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m
    
    cost = np.squeeze(cost)
    
    return cost

def backward_propagation(parameters, cache, X, Y):
    
    m = X.shape[1]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    W2 = parameters["W2"]
    
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2,A1.T) / m
    db2 = np.sum(dZ2, axis = 1, keepdims = True) / m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1,2))
    dW1 = np.dot(dZ1,X.T) / m
    db1 = np.sum(dZ1,axis = 1, keepdims = True) / m
    
    
    grads = {
        "dW2" : dW2,
        "db2" : db2,
        "dW1" : dW1,
        "db1" : db1
    }
    
    return grads

def update_parameters(parameters, grads, learning_rate=1.2):
    
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    
    updated_parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2" : W2,
        "b2" : b2
    }
    
    return updated_parameters

def predict(parameters, X):
    
    A2, cache = forward_propagation(X, parameters)
    
    predictions = np.round(A2)
    
    return predictions 

# 整合以上函数，进行总的整理预测：
def nn_model(X, Y, n_h, num_iteration, learning_rate,print_cost = False):
    
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(0, num_iteration):
        
        A2, cache = forward_propagation(X, parameters)
        
        cost = compute_cost(Y, A2, parameters)
        
        grads = backward_propagation(parameters, cache, X, Y)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 1000 == 0:
            
            print("Cost after iteration %i : %f" %(i, cost))
    return parameters

parameters = nn_model(X, Y, n_h = 4, num_iteration = 10000, learning_rate = 1.2, print_cost = True)

#print(parameters)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(parameters, X)
#print(predictions)
#print(Y)
#print(np.equal(Y, predictions))
#print(np.mean(np.equal(Y, predictions)))
print("精度为 : %d" % (np.mean(np.equal(Y,predictions)) * 100) + "%")

# test hidden layer size:

plt.figure(figsize = (16, 32))
hidden_layer_sizes = [1,2,3,4,5,20,30]

for i, n_h in enumerate(hidden_layer_sizes):
    print(i)
    plt.subplot(5, 2, i+1)
    plt.title("Hidden Layer of size %d" % n_h)
    parameters = nn_model(X, Y, n_h, num_iteration = 10000, learning_rate = 1.2, print_cost = False)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = np.mean(np.equal(Y, predictions))
    #print("精度为 : %d" % (accuracy *100) + "%")
    print("Accuracy for {} hidden units : {} %".format(n_h, accuracy))
    








