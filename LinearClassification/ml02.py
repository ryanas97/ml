pip install pandas
pip install matplotlib

from pandas import read_csv
import random
from numpy import array
import matplotlib.pyplot as plt
import math

#import csv
filename = 'Iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width','class','v1', 'v2']
dataset = read_csv(filename, names=names)

# Data Lookup
print(dataset.head(20))

# program function for sigmoid and output binary
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def output(x):
    if (x<0.5):
        return 0
    else:
        return 1
		
# split into input (X) and output (Y) variables
arr = dataset.values
X = arr[:,0:4] 
Y1 = arr[:,5]
Y2 = arr[:,6]
print(X)
print(Y1[149])

# define learning rate (0.1/0.8)
lr1 = 0.8

# linear equations classifier function

w1 = random.uniform(0, 1) 
w2 = random.uniform(0, 1) 
w3 = random.uniform(0, 1) 
w4 = random.uniform(0, 1)
b1 = random.uniform(0, 1)

dw1 = []
dw2 = []
dw3 = []
dw4 = []
db1 = []

t1 = []
sig1 = []
err1 = []
pred1 = []


w5 = random.uniform(0, 1) 
w6 = random.uniform(0, 1) 
w7 = random.uniform(0, 1) 
w8 = random.uniform(0, 1)
b2 = random.uniform(0, 1)

dw5 = []
dw6 = []
dw7 = []
dw8 = []
db2 = []

t2 = []
sig2 = []
err2 = []
pred2 = []

l1= [w1, w2, w3, w4]
l2= [w5, w6, w7, w8]

for epoch in range(1,101):
    # define accuracy variable 
    acc = 0
    for x in range(0,len(X)) :
        #Append multiple array
        t1.append(x), sig1.append(x), err1.append(x), dw1.append(x), dw2.append(x), dw3.append(x), dw4.append(x), db1.append(x)
        t2.append(x), sig2.append(x), err2.append(x), dw5.append(x), dw6.append(x), dw7.append(x), dw8.append(x), db2.append(x)
        # Target
        print("L1 : ", l1)
        t1[x] = sum(X[x]*l1) + b1
        t2[x] = sum(X[x]*l2) + b2
        # Sigmoid
        sig1[x] = sigmoid(t1[x])
        sig2[x] = sigmoid(t2[x])
        # Output
        print("Output 1: ", output(sig1[x]))
        print("Output 2: ", output(sig2[x]))
        print("Y1 : ", Y1[x])
        print("Y2 : ", Y2[x])
        if output(sig1[x]) == Y1[x] and output(sig2[x]) == Y2[x] :
            acc += 1
        print("Accuracy : ", acc)
        # Error
        err1[x] = (sig1[x]-Y1[x])**2
        print("Error 1 : ", err1[x])
        err2[x] = (sig2[x]-Y2[x])**2
        print("Error 2 : ", err2[x])
        # Partial Derivative of Weight and Bias
        dw1[x] = 2*(sig1[x]-Y1[x])*(1-sig1[x])*sig1[x]*X[x][0]
        dw2[x] = 2*(sig1[x]-Y1[x])*(1-sig1[x])*sig1[x]*X[x][1]
        dw3[x] = 2*(sig1[x]-Y1[x])*(1-sig1[x])*sig1[x]*X[x][2]
        dw4[x] = 2*(sig1[x]-Y1[x])*(1-sig1[x])*sig1[x]*X[x][3]
        db1[x] = 2*(sig1[x]-Y1[x])*(1-sig1[x])*sig1[x]*1
        dw5[x] = 2*(sig2[x]-Y2[x])*(1-sig2[x])*sig2[x]*X[x][0]
        dw6[x] = 2*(sig2[x]-Y2[x])*(1-sig2[x])*sig2[x]*X[x][1]
        dw7[x] = 2*(sig2[x]-Y2[x])*(1-sig2[x])*sig2[x]*X[x][2]
        dw8[x] = 2*(sig2[x]-Y2[x])*(1-sig2[x])*sig2[x]*X[x][3]
        db2[x] = 2*(sig2[x]-Y2[x])*(1-sig2[x])*sig2[x]*1
        # Update the new value of weight and bias for next itteration
        w1 = w1-(lr1*dw1[x])
        w2 = w2-(lr1*dw2[x])
        w3 = w3-(lr1*dw3[x])
        w4 = w4-(lr1*dw4[x])
        b1 = b1-(lr1*db1[x])
        w5 = w5-lr1*dw5[x]
        w6 = w6-lr1*dw6[x]
        w7 = w7-lr1*dw7[x]
        w8 = w8-lr1*dw8[x]
        b2 = b2-lr1*db2[x]
        l1= [w1, w2, w3, w4]
        l2= [w5, w6, w7, w8]
        print("------------------------------------------------------")
    # Error Average and Accuracy
    avg_err1 = (sum(err1[0:len(X)])/len(X))
    avg_err2 = (sum(err2[0:len(X)])/len(X))
    accuracy = (acc/len(X))
    print("Average Error1 : ", avg_err1)
    print("Average Error2 : ", avg_err2)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # Plotting Error and Accuracy Chart
    plt.figure(1)
    plt.plot(epoch, avg_err1, '-o')
 
    plt.figure(2)
    plt.plot(epoch, avg_err1, '-o')
    
    plt.figure(3)
    plt.plot(epoch, accuracy, '-o')
    