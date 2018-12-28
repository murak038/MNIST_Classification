""" Starter code for logistic regression model
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/
"""
import os
""" In this code, I implemented a multilayer model in order to model nonlinear relationships between the various pixels 
 in the image. Adding additional layers allows the model to capture nonlinear relationships while increasing the size
 of the hidden layers allows for the model to factor in more relationships"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load in the libraries
import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import time
tf.reset_default_graph()
# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 10

# Read in data using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets('/tmp/mnist', one_hot=True)

# Create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9.
# Features are of the type float, and labels are of the type int
x = tf.placeholder(tf.float32, [batch_size, 784], name='features')
y = tf.placeholder(tf.int32, [batch_size, 10], name='labels')




# Create weights and bias weights and biases are initialized to 0 shape of w depends on the dimension of X and
# Y so that Y = X * w + b shape of b depends on Y

# to increase the performance of the logistic regression on the MNIST dataset, two hidden layers of various sizes
# are used

# the first layer has 256 nodes
layer1 = 256
w1 = tf.Variable(initial_value=tf.random_normal([784,layer1], stddev=0.01), name='weight_1')
b1 = tf.Variable(initial_value=tf.random_normal([batch_size,layer1], stddev=0.01), name='bias_1')

# the second layer has 64 nodes
layer2 = 64
w2 = tf.Variable(initial_value=tf.random_normal([layer1,layer2], stddev=0.01), name='weight_2')
b2 = tf.Variable(initial_value=tf.random_normal([batch_size,layer2], stddev=0.01), name='bias_2')

# the output layer prior to softmax
w3 = tf.Variable(initial_value=tf.random_normal([layer2,10], stddev=0.01), name='weight_3')
b3 = tf.Variable(initial_value=tf.random_normal([batch_size,10], stddev=0.01), name='bias_3')

# build model the model that returns the logits. this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image
# DO NOT DO SOFTMAX HERE

logits1 = tf.matmul(x,w1)+b1
logits1 = tf.nn.relu(logits1)
logits2 = tf.matmul(logits1, w2) + b2
logits2 = tf.nn.relu(logits2)
logits = tf.matmul(logits2,w3) + b3


# Define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y, name='loss')
loss = tf.reduce_mean(entropy)

# Define training op using gradient descent to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    # save the output of the graph to Tensorboard
    writer = tf.summary.FileWriter('./graphs_ImprovedLogReg',sess.graph)
    # start the timer
    start_time = time.time()
    # initialize variables
    sess.run(tf.global_variables_initializer())
    # number of batches
    n_batches = int(mnist.train.num_examples / batch_size)
    for i in range(n_epochs):  # train the model n_epochs times
        total_loss = 0
        # run through batches of data
        for _ in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size) # get the next batch of data
            _, loss_batch = sess.run([optimizer, loss], feed_dict={x: X_batch, y: Y_batch}) # run the session
            total_loss += loss_batch
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))

    print('Total time: {0} seconds'.format(time.time() - start_time))

    print('Optimization Finished!')

    # test the model
    preds = tf.nn.softmax(logits) # find the prediction by taking the softmax of the logits
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1)) # are the prediction and labels the same?
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))  # calculate the accuracy

    n_batches = int(mnist.test.num_examples / batch_size)
    total_correct_preds = 0

    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run([accuracy], feed_dict={x: X_batch, y: Y_batch}) # find the accuracy of batch of data
        total_correct_preds += accuracy_batch[0]

    print('Accuracy {0}'.format(total_correct_preds / mnist.test.num_examples))
    writer.close()