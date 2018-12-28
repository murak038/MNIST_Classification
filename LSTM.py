import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import time
import utils
tf.reset_default_graph()
lstm_size = 256
batch_size = 128
n_epochs = 10
learning_rate = 0.01
dropout = 0.5

# Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets('/tmp/mnist', one_hot=True)

# Create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9.
# Features are of the type float, and labels are of the type int
graph = tf.get_default_graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, [batch_size, 784], name='features') # create placeholder for input data
    y = tf.placeholder(tf.int32, [batch_size, 10], name='labels') # create placeholder for the variable
    # create a weight and bias variable to map the output of LSTM to the labels
    w = tf.Variable(initial_value=tf.random_normal([lstm_size, 10], stddev=0.01), name='weight')
    b = tf.Variable(initial_value=tf.random_normal([10], stddev=0.01), name='bias')
    # reshape the input so that the LSTM carries out the cell functions for each row of the data
    input_x = tf.reshape(x, shape=[batch_size, 28, 28])
    # create the LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    # add a dropout layer to the LSTM
    cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = dropout)
    # the initial stage of the cell is set to zero
    initial_state = cell.zero_state(batch_size, tf.float32)
    # create a recurrent neural network with the LSTM cell defined prior
    output, _ = tf.nn.dynamic_rnn(cell, inputs=input_x, initial_state=initial_state)
    # compute logits from the output of the LSTM
    logits = tf.matmul(output[:,-1], w) + b
    # calculate the entropy and the loss of the logit values computed
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y, name='loss')
    loss = tf.reduce_mean(entropy)
    # use an AdamOptimizer to minimize the loss value
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


with tf.Session(graph = graph) as sess:
    # create a writer to get a Tensorboard graph output
    writer = tf.summary.FileWriter('./graphs_LSTM',sess.graph)
    # start timer
    start_time = time.time()
    # initialize all of the variables
    sess.run(tf.global_variables_initializer())
    # the number of total batches
    n_batches = int(mnist.train.num_examples / batch_size)
    for i in range(n_epochs):  # train the model n_epochs times
        total_loss = 0

        for _ in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size) # get the next batch of data
            _, loss_batch = sess.run([optimizer, loss], feed_dict={x: X_batch, y: Y_batch}) # train network
            total_loss += loss_batch # total loss
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))

    print('Total time: {0} seconds'.format(time.time() - start_time))

    print('Optimization Finished!')  # should be around 0.35 after 25 epochs

    # test the model
    preds = tf.nn.softmax(logits)  # find the prediction by taking the softmax of the logits
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))  # are the prediction and labels the same?
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))  # calculate the accuracy

    n_batches = int(mnist.test.num_examples / batch_size)
    total_correct_preds = 0

    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run([accuracy], feed_dict={x: X_batch, y: Y_batch})  # find the accuracy of batch of data
        total_correct_preds += accuracy_batch[0]

    print('Accuracy {0}'.format(total_correct_preds / mnist.test.num_examples))
    writer.close()
