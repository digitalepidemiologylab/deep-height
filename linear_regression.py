#!/usr/bin/env python

import tensorflow as tf
import numpy as np

import open_snp_data


train, test = open_snp_data.load_data("opensnp_data/")

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 10
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [batch_size, len(train.snps[0])]) 
y = tf.placeholder(tf.float32, [batch_size]) 

# Set model weights
W = tf.Variable(tf.zeros([batch_size, len(train.snps[0])]))
b = tf.Variable(tf.zeros([batch_size]))

# Construct model
pred = tf.add(tf.matmul(x, tf.transpose(W)),  tf.transpose(b)) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*batch_size)

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

