#importing necesary library
import time
import timeit
from timeit import default_timer as timer

import matplotlib.pyplot as plt  # for plotting purpose
import numpy as np  # for array
import pandas as pd  # for file operation
#from guppy import hpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # for converting categorical into numerical
from sklearn.preprocessing import OneHotEncoder
import tracemalloc
import os
import psutil

tracemalloc.start()


start = timeit.timeit()
start1 = time.time()
start2 = timer()
t = time.process_time()


encode = LabelEncoder()

data = pd.read_excel("data.xlsx")
data = data.sample(frac=1)

x = data['Data']
y = data['Significance']

encode = LabelEncoder()
y = encode.fit_transform(y)

all_data = np.array(x)

#binary encoded (one hot encoder)
Onehot_encoder = OneHotEncoder(sparse=False)
all_data = all_data.reshape(len(all_data), 1)
encoded_x = Onehot_encoder.fit_transform(all_data)

x_train, x_test, y_train, y_test = train_test_split(encoded_x, y, test_size= 0.15, random_state=21)

#Now transposing data for further manupulation
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T


#Now printing shape of the transposed training as well as testing data
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)

#Constructing Neural network with one hidden layer (total 3 leyars)
#initializing weights and bias
def initialize_parameters_and_layer_sizes_NN(x_train, y_train):
    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.1,
                  "bias1": np.ones((3,1)),
                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1,
                  "bias2": np.ones((y_train.shape[0],1))}
    return parameters


#defining function for forward propagation

def sigmoid(z):
    #z = np.float64(z)
    y_head = 1/(1+np.exp(-z))
    #y_head = 1/(-z).exp()
    #print("from function sigmoid","  ",y_head)
    return y_head

#Q=P*(1-np.exp(-A))
#Q = P*(-A).expm1()
#Q = -P*np.expm1(-A)


def forward_propagation_NN(x_train, parameters):
    Z1 = np.dot(parameters["weight1"],x_train) +parameters["bias1"]
    Z1 = np.float64(Z1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    #print("from function forward_propagation_NN","  A2 : ",A2, "Cache ", cache)
    return A2, cache


#now we have predicted value from forward propagation, so we need cost function as well as loss function
def compute_cost_NN(A2, Y, parameters):
    logprobs = np.multiply(np.log(A2),Y)
    cost = abs(sum(logprobs)/Y.shape[0])
    return cost


#Now time to propagate backward
def backward_propagation_NN(parameters, cache, X, Y):

    dZ2 = cache["A2"]-Y
    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[0]
    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[0]
    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))
    dW1 = np.dot(dZ1,X.T)/X.shape[1]
    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]
    grads = {"dweight1": dW1,
             "dbias1": db1,
             "dweight2": dW2,
             "dbias2": db2}
    #print("from function backward_propagation_NN","  ",grads)
    return grads

#now time to update all the parameters and weights
def update_parameters_NN(parameters, grads, learning_rate = 0.01):
    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],
                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],
                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],
                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}
    #print("from function update_parameters_NN","  ",parameters)
    return parameters

#now making prediction
def predict_NN(parameters,x_test):
    # x_test is a input for forward propagation
    A2, cache = forward_propagation_NN(x_test,parameters)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(A2.shape[1]):
        if A2[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    #print("from function predict_NN","  ",Y_prediction)
    return Y_prediction


# creating neural network model (1 hidden layer)
def two_layer_neural_network(x_train, y_train, x_test, y_test, num_iterations):
    cost_list = []
    index_list = []
    # initialize parameters and layer sizes
    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)

    for i in range(0, num_iterations):
        # forward propagation
        A2, cache = forward_propagation_NN(x_train, parameters)
        # compute cost
        cost = compute_cost_NN(A2, y_train, parameters)
        # backward propagation
        grads = backward_propagation_NN(parameters, cache, x_train, y_train)
        # update parameters
        parameters = update_parameters_NN(parameters, grads)

        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            # print (cost)
    plt.plot(index_list, cost_list)
    plt.xticks(index_list, rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()

    # predict
    y_prediction_test = predict_NN(parameters, x_test)
    y_prediction_train = predict_NN(parameters, x_train)
    y_prediction_test = y_prediction_test.flatten()
    print(y_prediction_test, y_test)

    # Print train/test Errors
   # print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    #print("from function two_layer_neural_network","  ",parameters)
    #print(confusion_matrix(y_test, y_prediction_test))
    return parameters



parameters = two_layer_neural_network(x_train, y_train, x_test, y_test, num_iterations=3000)


end = timeit.timeit()
end1 = time.time()
end2 = timer()
elapsed_time = time.process_time() - t


#h = hpy()
#print(h.heap())

#print(end - start, "    ", end1 - start1, "      ", end2 - start2, "     ", elapsed_time )

end6 = tracemalloc.take_snapshot()

process = psutil.Process(os.getpid())
print(process.memory_info().rss)
#display_top(end6)



