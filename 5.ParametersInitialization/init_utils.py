import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_dataset()

def initialize_parameters_zeros(layer_dims):
    
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        
        parameters["W" + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters 

def initialize_parameters_random(layer_dims):
    
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 10
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
     
    return parameters

def initialize_parameters_he(layer_dims):
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)-1
    
    for l in range():
        
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    
    return parameters 

def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he"):
        
    grads = {}
    costs = [] 
    m = X.shape[1] 
    layers_dims = [X.shape[0], 10, 5, 1]
    
    if initialization == "zeros":
        
        parameters = initialize_parameters_zeros(layers_dims)
        
    elif initialization == "random":
        
        parameters = initialize_parameters_random(layers_dims)
        
    elif initialization == "he":
        
        parameters = initialize_parameters_he(layers_dims)


    for i in range(0, num_iterations):

        a3, cache = forward_propagation(X, parameters)
        
        cost = compute_loss(a3, Y)

        grads = backward_propagation(X, Y, cache)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 10 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
            
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    if initialization == "zeros":

        plt.savefig('/home/guanlingh/local2server/homework5/zeros.jpg')

    elif initialization == "random":

        plt.savefig('/home/guanlingh/local2server/homework5/random.jpg')

    elif initialization == "he":

        plt.savefig('/home/guanlingh/local2server/homework5/he.jpg')

    return parameters

print("-------------------------------------When initialization is zeros------------------------------------\n")

parameters1 = model(train_X, train_Y, initialization = "zeros")

print("Train1 : ")
train_prediction1 = predict(train_X, train_Y, parameters1)
print("Test1 : ")
test_predition1 = predict(test_X,test_Y, parameters1)

print("Train predictions1: " + str(train_prediction1))
print("Test predictions1: " + str(test_predition1))

print("-------------------------------------When initialization is random------------------------------------\n")

parameters2 = model(train_X, train_Y, initialization = "random")

print("Train2 : ")
train_prediction2 = predict(train_X, train_Y, parameters2)
print("Test2 : ")
test_predition2 = predict(test_X,test_Y, parameters2)

print("Train predictions2: " + str(train_prediction2))
print("Test predictions2: " + str(test_predition2))

print("-------------------------------------When initialization is he------------------------------------\n")

parameters3 = model(train_X, train_Y, initialization = "he")

print("Train3 : ")
train_prediction3 = predict(train_X, train_Y, parameters3)
print("Test2 : ")
test_predition3 = predict(test_X,test_Y, parameters3)

print("Train predictions3: " + str(train_prediction3))
print("Test predictions3: " + str(test_predition3))




