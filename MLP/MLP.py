from pandas import read_csv
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from random import seed
from random import randrange
from numpy import array

#import csv
filename = 'Iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width','class','v1', 'v2']
dataset_csv = read_csv(filename, names=names)

# Data Lookup
print(dataset_csv.head(20))

# program function for sigmoid and output binary
def prediction(x):
    if (x<0.5):
        return 0
    else:
        return 1

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

arr = dataset_csv.values
X = arr[:,0:4] 
Y1 = arr[:,5]
Y2 = arr[:,6]
Y = [Y1,Y2]
print(X)
print(Y)

# Split a dataset into a train and test set
def train_test_split(dataset,dataver,datavir, split=0.6):
    train = list()
    train_size = split * 50
    dataset_copy1 = list(dataset)
    dataset_copy2 = list(dataver)
    dataset_copy3 = list(datavir)
    while len(train) < train_size:
        index = randrange(len(dataset_copy1))
        train.append(dataset_copy1.pop(index))
        index = randrange(len(dataset_copy2))
        train.append(dataset_copy2.pop(index))
        index = randrange(len(dataset_copy3))
        train.append(dataset_copy3.pop(index))
    return train
 
# test train/test split
dataset = []
dataver = []
datavir = []
test_ns = []
for x1 in range(50):
    dataset.append(x1)
    dataver.append(x1+50)
    datavir.append(x1+100)
test_ns = train_test_split(dataset,dataver,datavir)
test = sorted(test_ns,key=int)
print("Test Data : ",test, len(test))
train = [x for x in dataset if x not in test]
train += [x for x in dataver if x not in test]
train += [x for x in datavir if x not in test]
print("Train Data : ", train, len(train))

for x in range(len(test)) :
    print("Data test ",x+1)
    print(X[test[x]])
    print(Y[0][test[x]])
    print(Y[1][test[x]])

# define learning rate (0.1/0.8)
lr = 0.1

theta = [[],[],[],[]]
bias = []
targethid = [[],[],[],[]]
outputhid = [[],[],[],[]]
v = [[],[],[],[]]
bias2 = []
target = [[],[]]
output = [[],[]]
error = [[],[]]
lambda_output = [[],[]]
delta_v = [[0,0],[0,0],[0,0],[0,0]]
delta_bias2 = [0,0,0,0]
lambda_hidden = [[],[],[],[]]
delta_theta = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
delta_bias = [0,0,0,0] 

targethid_test = [[],[],[],[]]
outputhid_test = [[],[],[],[]]
v_test = [[],[],[],[]]
bias2_test = []
target_test = [[],[]]
output_test = [[],[]]
error_test = [[],[]]

plot_epoch = []
plot_accuracy = []
plot_error = []
plot_accuracy_test = []
plot_error_test = []

for i in range(0,4):
    for j in range(0,4):
        theta[i].append(random.uniform(0,1))
    bias.append(random.uniform(0,1))
    
#print("Theta : ", theta)
#print("Bias : ", bias)

for i in range(0,4):
    for j in range(0,2):
        v[i].append(random.uniform(0,1))
        if i==0:
            bias2.append(random.uniform(0,1))

#print("V : ", v)
#print("Bias 2 : ", bias2)

for epoch in range(1,201):
    print("Epoch : ", epoch)
    print("--------------------------------------------------------------------")
    print("TRAIN DATA")
    accuracy_array = [0]
    accuracy_array_test = [0]
    error_array = [0]
    error_array_test = [0]
    for x in range(0,len(train)):
        for i in range(0,4):
            targethid[i] = theta[0][i]*X[train[x]][0] + theta[1][i]*X[train[x]][1] + theta[2][i]*X[train[x]][2] + theta[3][i]*X[train[x]][3] + bias[i]
        #print("Target Hidden: ",targethid)
        for i in range(0,4):
            outputhid[i] = sigmoid(targethid[i])
        #print("Output Hidden : ",outputhid)

        for i in range(0,2):
            target[i] = v[0][i]*outputhid[0] + v[1][i]*outputhid[1] + v[2][i]*outputhid[2] + v[3][i]*outputhid[3] + bias2[i]
            output[i] = sigmoid(target[i])
            error[i] = ((output[i]-Y[i][train[x]])**2)/2
            lambda_output[i] = (output[i]-Y[i][train[x]])*output[i]*(1-output[i])
        #print("Target : ",target)
        #print("Output : ",output)
        #print("Error : ",error)
        #print("Error Average : ", sum(error)/2)
        error_array[0]+=(sum(error)/2)
        #print("Lambda Output : ",lambda_output)
        
        if prediction(output[0]) == Y[0][train[x]] and prediction(output[1]) == Y[1][train[x]]:
            accuracy_array[0] += 1
        #print("Accuracy : ", accuracy_array[0])

        for i in range(0,4):
            for j in range(0,2):
                delta_v[i][j] = outputhid[i]*lambda_output[j]
                if i==0:
                    delta_bias2[j] = lambda_output[j]
        #print("Delta V : ",delta_v)
        #print("Delta Bias2 : ",delta_bias2)


        for i in range(0,4):
            lambda_hidden[i] = (lambda_output[0]*v[i][0]+lambda_output[1]*v[i][1])*outputhid[i]*(1-outputhid[i])

        #print("Lambda Hidden: ",lambda_hidden)

        for i in range(0,4):
            for j in range(0,4):
                delta_theta[i][j] = X[train[x]][i]*lambda_hidden[j]
            delta_bias[i] = lambda_hidden[i]

        #print("Delta Theta : ",delta_theta)
        #print("Delta Bias : ",delta_bias)

        #Update weight and bias
        for i in range(0,4):
            for j in range(0,2):
                v[i][j] = v[i][j] - lr*delta_v[i][j]
                if i==0:
                    bias2[j] = bias2[j] - lr*delta_bias2[j]

        #print("V+ : ", v)
        #print("Bias 2+ : ", bias2)

        for i in range(0,4):
            for j in range(0,4):
                theta[i][j] = theta[i][j] - lr*delta_theta[i][j]
            bias[j] = bias[j] - lr*delta_bias[j]

        #print("Theta+ : ", theta)
        #print("Bias+ : ", bias)
    print("Error Train : ", error_array[0]/len(train))
    print("Accuracy : ", accuracy_array[0],"/",len(train))
    print("Accuracy Train : ", accuracy_array[0]/len(train))
    print("--------------------------------------------------------------------")
    print("TEST DATA")       
    #print("Weight : ", v)
    #print("Bias : ", bias2)
    for x in range(0,len(test)):
        for i in range(0,4):
            targethid_test[i] = theta[0][i]*X[test[x]][0] + theta[1][i]*X[test[x]][1] + theta[2][i]*X[test[x]][2] + theta[3][i]*X[test[x]][3] + bias[i]
        #print("Target Hidden: ",targethid_test)
        for i in range(0,4):
            outputhid_test[i] = sigmoid(targethid_test[i])
        #print("Output Hidden : ",outputhid_test)

        for i in range(0,2):
            target_test[i] = v[0][i]*outputhid_test[0] + v[1][i]*outputhid_test[1] + v[2][i]*outputhid_test[2] + v[3][i]*outputhid_test[3] + bias2[i]
            output_test[i] = sigmoid(target_test[i])
            error_test[i] = ((output_test[i]-Y[i][test[x]])**2)/2
        #print("Target : ",target_test)
        #print("Output : ",output_test)
        #print("Error : ",error_test)
        #print("Error Average : ", sum(error_test)/2)
        error_array_test[0]+=(sum(error_test)/2)
        
        if prediction(output_test[0]) == Y[0][test[x]] and prediction(output_test[1]) == Y[1][test[x]]:
            accuracy_array_test[0] += 1
        #print("Accuracy : ", accuracy_array_test[0])
        #print("Test Data: ")
        #print("Output/Prediction 1 : ",Y[0][test[x]],prediction(output_test[0]))
        #print("Output/Prediction 2 : ",Y[1][test[x]],prediction(output_test[1]))
    print("Error Test : ", error_array_test[0]/len(test))
    print("Accuracy : ", accuracy_array_test[0],"/",len(test))
    print("Accuracy Test : ", accuracy_array_test[0]/len(test))
    
    #Array Graph
    plot_epoch.append(epoch)
    plot_error.append(error_array[0]/len(train))    
    plot_accuracy.append(accuracy_array[0]/len(train))
    plot_error_test.append(error_array_test[0]/len(test))
    plot_accuracy_test.append(accuracy_array_test[0]/len(test))
    print("---------------------------------------------------------------------------------------------------------------")
    
# print("Epoch Array : ", plot_epoch)
# print("Accuracy Array - Train : ", plot_accuracy)
# print("Error Array - Train : ", plot_error)
# print("Accuracy Array - Test : ", plot_accuracy_test)
# print("Error Array - Test : ", plot_error_test)

# Plotting Error and Accuracy Chart
plt.figure(1)
plt.plot(plot_epoch,plot_error, label='Training')
plt.plot(plot_epoch,plot_error_test, label='Test')
plt.title('Error Graph MLP - Training & Test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Scale')

plt.figure(2)
plt.plot(plot_epoch, plot_accuracy,label='Training')
plt.plot(plot_epoch, plot_accuracy_test,label='Test')
plt.title('Accuracy Chart MLP - Training & Test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Scale')