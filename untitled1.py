# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:08:07 2017

@author: umesh.vijay.pathak
"""


from scipy import ndimage
from scipy import misc
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import lasagne
from lasagne import layers
from nolearn.lasagne import NeuralNet


data = misc.imread("C:\Users\umesh.vijay.pathak\Pictures\sample.jpg")
size = data.shape

data = data[:,:,0]
widthParamFinal = data.sum(axis = 0)
compVal = int(widthParamFinal[0])
startwidth = 0
endwidth = 0

for i in widthParamFinal:
    if i == compVal:
        startwidth+=1;
    else:
        break;
        
for i in reversed(widthParamFinal):
    if i == compVal:
        endwidth+=1;
    else:
        break;
            
    
heightParamFinal = data.sum(axis = 1)
compVal2 = int(heightParamFinal[0])
startHeight = 0
endHeight = 0

for i in heightParamFinal:
    if i == compVal2:
        startHeight+=1;
    else:
        break;
        
for i in reversed(heightParamFinal):
    if i == compVal2:
        endHeight+=1;
    else:
        break;
        
data = data[(startHeight-50):(len(heightParamFinal)-endHeight+50),(startwidth-40):(len(widthParamFinal)-endwidth+40)]    
    
    
DivParamFinal = data.sum(axis = 0)

compValue = DivParamFinal[0]
breakPoints = []
flag = 0
count = 0
for i in DivParamFinal:
    count+=1
    if i == compValue and flag == 1:
        flag = 0
        breakPoints.append(count)
    elif i != compValue and flag == 0:
        flag = 1
        breakPoints.append(count)

breakPoints = breakPoints[1:-1]       
breakPoints = map(sum, zip(breakPoints[::2], breakPoints[1::2]))         
breakPoints = [1]+[(i/2) for i in breakPoints]+[len(DivParamFinal)]



df = pd.DataFrame([])

for br in range(len(breakPoints)-1):
    dataX = data[:,breakPoints[br]:breakPoints[br+1]]
    dataX = misc.imresize(dataX,(28,28))
    dataX = 255 - dataX
    df = df.append(pd.DataFrame(dataX.reshape(1,-1)))
    misc.imsave("C:\Users\umesh.vijay.pathak\Pictures\download"+str(br)+".jpg",dataX)

test = np.array(df).reshape((-1, 1, 28, 28)).astype(np.uint8)
   

dataset = pd.read_csv("C:\\Users\\umesh.vijay.pathak\\Pictures\\train.csv")
target = dataset.iloc[:,0].values.ravel()
train = dataset.iloc[:,1:].values

target = target.astype(np.uint8)
train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
 



def CNN(n_epochs):
    net1 = NeuralNet(
        layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),      #Convolutional layer.  Params defined below
        ('pool1', layers.MaxPool2DLayer),   # Like downsampling, for execution speed
        ('conv2', layers.Conv2DLayer),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],

    input_shape=(None, 1, 28, 28),
    conv1_num_filters=7, 
    conv1_filter_size=(3, 3), 
    conv1_nonlinearity=lasagne.nonlinearities.rectify,
        
    pool1_pool_size=(2, 2),
        
    conv2_num_filters=12, 
    conv2_filter_size=(2, 2),    
    conv2_nonlinearity=lasagne.nonlinearities.rectify,
        
    hidden3_num_units=1000,
    output_num_units=10, 
    output_nonlinearity=lasagne.nonlinearities.softmax,

    update_learning_rate=0.0001,
    update_momentum=0.9,

    max_epochs=n_epochs,
    verbose=1,
    )
    return net1

cnn = CNN(15).fit(train,target) # train the CNN model for 15 epochs

pred = cnn.predict(test)
