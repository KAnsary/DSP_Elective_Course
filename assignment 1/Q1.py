#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 03:57:22 2018

@author: ansary
"""
#import needed Libraries 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metric

#set seed for random variables
np.random.seed(0)

#learning Parameters
learning_rate = 0.001
training_epochs = [25000,50000,50000,25000]
display_step = 1000

# Training Data
Years=np.asarray([1990,1991,1992,1993,1994])
train_X = Years-1990
train_Y = np.asarray([19.358, 19.484, 20.293, 21.499, 23.561])
n_samples = train_X.shape[0]

##plot scattered training data
plt.figure(1)
plt.plot(Years, train_Y, 'bo', label='Original data')
#plt.title("Data")
#plt.xlabel("Years")
#plt.ylabel("Solid Waste")
#plt.show()

#regression models
Models=["Linear","Quadratic","Cubic","Power"]
ModelIdx = 0
#R_squared values
RSqr=np.zeros(4)
for ModelIdx in range(len(Models)):
    tf.reset_default_graph()
    #features
    X = tf.placeholder(tf.float32, name="X")
    #output
    Y = tf.placeholder(tf.float32, name="Y")
    
    # Set model weights
    W1 = tf.Variable(np.random.randn(),name="W1")
    W2 = tf.Variable(np.random.randn(),name="W2")
    W3 = tf.Variable(np.random.randn(),name="W3")
    b = tf.Variable(np.random.randn(),name="b")
    
    
    # Construct the model
    if ModelIdx==0: #linear
        RegrModel = tf.add(tf.multiply(X, W1), b)
    elif ModelIdx==1: #quad
        element1 = tf.multiply(tf.pow(X,2),W1)
        element2 = tf.multiply(X,W2) 
        RegrModel = element1+element2+b
    elif ModelIdx==2: #cubic
        element1 = tf.multiply(tf.pow(X,3),W1)
        element2 = tf.multiply(tf.pow(X,2),W2)
        element3 = tf.multiply(X,W3) 
        RegrModel = element1+element2+element3+b
    else: #power
        element1 = tf.pow(X,W1)
        element2 = tf.multiply(element1,W2) 
        RegrModel = element2+b       

    # Mean squared error Cost function
    cost = tf.reduce_sum(tf.pow(RegrModel-Y,2))/(2*n_samples)
    # Gradient descent optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    # Initializing the variables 
    init = tf.global_variables_initializer()
    # Start session and training
    with tf.Session() as sess:
    
        # Run the initializer
        sess.run(init)
    
        # Train through data
        for epoch in range(training_epochs[ModelIdx]):
            sess.run([optimizer, cost], feed_dict={X: train_X, Y: train_Y})
                    # Display logs per epoch step
#            if (epoch+1) % display_step == 0:
#                c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
#                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.5f}".format(c))

        print("Optimization Finished!")
        print("**********************")
        training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        RSqr[ModelIdx]=metric.r2_score(train_Y,sess.run(RegrModel,feed_dict={X: train_X}))
        print("Model:",Models[ModelIdx],"R_sqr=","{:.3f}".format(RSqr[ModelIdx]),\
              "cost=","{:.5f}".format( training_cost),'\n')
        if ModelIdx==0:
            print("Model:",Models[ModelIdx],"Parameters:W1=",sess.run(W1),"b=",sess.run(b),'\n')
            print("Prediction for 2000 is",sess.run(RegrModel,feed_dict={X: 2000-1990}),\
                  "for 2005",sess.run(RegrModel,feed_dict={X: 2005-1990}),'\n')
        elif ModelIdx==1:
            print("Model:",Models[ModelIdx],"Parameters:W1=",sess.run(W1),\
                  "W2=",sess.run(W2),"b=",sess.run(b),'\n')
            print("Prediction for 2000 is",sess.run(RegrModel,feed_dict={X: 2000-1990}),\
                  "for 2005",sess.run(RegrModel,feed_dict={X: 2005-1990}),'\n')
        elif ModelIdx==1:
            print("Model:",Models[ModelIdx],"Parameters:W1=",sess.run(W1),\
                  "W2=",sess.run(W2),"W3=",sess.run(W3),"b=",sess.run(b),'\n')
            print("Prediction for 2000 is",sess.run(RegrModel,feed_dict={X: 2000-1990}),\
                  "for 2005",sess.run(RegrModel,feed_dict={X: 2005-1990}),'\n')
        else:
            print("Model:",Models[ModelIdx],"Parameters:W1=",sess.run(W1),\
                  "W2=",sess.run(W2),"b=",sess.run(b),'\n')
            print("Prediction for 2000 is",sess.run(RegrModel,feed_dict={X: 2000-1990}),\
                  "for 2005",sess.run(RegrModel,feed_dict={X: 2005-1990}),'\n')            
        # Graphical display
        
        YAxis=np.linspace(1988,1996,100)
        XAxis=np.linspace(1988-1990,1996-1990,100)
        plt.plot(YAxis, sess.run(RegrModel,feed_dict={X: XAxis}), label=Models[ModelIdx])
        plt.title('Fitting curves') 
        plt.legend()
        plt.show()
