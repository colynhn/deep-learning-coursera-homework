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
    
    plt.savefig('/home/guanlingh/local2server/homework6/image/h6-1.jpg')
    
    return parameters

parameters = model(train_X, train_Y)
print("Train set : ")
train_prediction = predict(train_X, train_Y, parameters)
print("Test set : ")
Test_prediction = predict(test_X, test_Y, parameters)







