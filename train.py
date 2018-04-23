# -*- coding: utf-8 -*-
"""
Zen Chai
22/03/2018
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import numpy as np

# Load in MNIST Digits data where each image is 28 x 28 pixels
print("Importing MNIST Digits Data: ")
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print()
print("Training set size: {0}".format(mnist.train.images.shape[0]))
print("Test set size: {0}".format(mnist.test.images.shape[0]))
print("Validation set size: {0}".format(mnist.validation.images.shape[0]))

# Network Information
print()
print("Building fully connected layers with one hidden layers with 784 weights.")
print("Using softmax activation with 10 outputs.")
print("Using cross entropy cost function with Adam optimizer.")
ops.reset_default_graph()

# Create placeholder for training data
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y = tf.placeholder(tf.float32, shape=[None, 10], name="y")

# Initialise random weights for one layer fully connected network
tf.set_random_seed(1)   
W1 = tf.get_variable("W1", [784,100], \
                     initializer=tf.contrib.layers.xavier_initializer(seed=1))
b1 = tf.get_variable("b1", [1, 100], \
                     initializer=tf.zeros_initializer())
W2 = tf.get_variable("W2", [100,10], \
                     initializer=tf.contrib.layers.xavier_initializer(seed=1))
b2 = tf.get_variable("b2", [1, 10], \
                     initializer=tf.zeros_initializer())

# Define forward propagation
Z1 =  tf.add(tf.matmul(x, W1), b1)
A1 = tf.nn.relu(Z1)
Z2 = tf.add(tf.matmul(A1, W2), b2)

# Define cost functin
#labels = tf.transpose(y)
#logits = tf.transpose(Z)
labels = y
logits = Z2
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

# Hyperparameters Info
learningRate = 0.001
numEpochs = 20
m = mnist.train.images.shape[0]
miniBatchSize = 100
numMiniBatches = int(m / miniBatchSize)
print()
print("Learning rate: {0}".format(learningRate))
print("Number of epochs: {0}".format(numEpochs))
print("Mini batch size: {0}".format(miniBatchSize))
print("Number of mini batches: {0}".format(numMiniBatches))

# Define optimizer for backward propagation
optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)

# Initialialise session
init = tf.global_variables_initializer()
costs = []
with tf.Session() as sess:
    sess.run(init)
    
    # Execute training loop 
    for epoch in range(numEpochs):        
        epochCost = 0
        for minibatch in range(numMiniBatches):
            minibatchX, minibatchY = mnist.train.next_batch(miniBatchSize)
            _, tempCost = sess.run([optimizer, cost], feed_dict={x:minibatchX, y:minibatchY})
            epochCost += tempCost / numMiniBatches
        
        if epoch % 5 == 0:
            print("Cost after epoch %i: %f" % (epoch, epochCost))
            costs.append(epochCost)
            
    # Plot cost 
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per ten)')
    plt.title('Learning rate = '+str(learningRate))    
    plt.savefig("images/result_epoch" + str(numEpochs) + ".jpg")
    plt.show()
            
    # Evaluate validation and test sets
    pred = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))                  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
       
    print("Test set accuracy: {0}".format(accuracy.eval(feed_dict=
          {x: mnist.test.images, y: mnist.test.labels})))
    print("Validation set accuracy: {0}".format(accuracy.eval(feed_dict=
          {x: mnist.validation.images, y: mnist.validation.labels})))
    
