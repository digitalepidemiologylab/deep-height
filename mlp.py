#!/usr/bin/env python

import tensorflow as tf
import numpy as np

import open_snp_data

LOG_DIR = "./logdir"

train, test = open_snp_data.load_data("opensnp_data/", small=True)

input_dims = len(train.snps[0])

learning_rate = 0.001
training_epochs = 100
batch_size = 1
display_step = 1

n_hidden_1 = 10
n_hidden_2 = 10

n_input = input_dims

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
	
    # Create a summary to visualize the first layer ReLU activation
    tf.histogram_summary("relu1", layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Create another summary to visualize the second layer ReLU activation
    tf.histogram_summary("relu2", layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
with tf.device("/cpu:0"):
		weights = {
			'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
			'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
			'out': tf.Variable(tf.random_normal([n_hidden_2, batch_size]))
		}
		biases = {
			'b1': tf.Variable(tf.random_normal([n_hidden_1])),
			'b2': tf.Variable(tf.random_normal([n_hidden_2])),
			'out': tf.Variable(tf.random_normal([batch_size]))
		}

with tf.name_scope("Model"):
		pred = multilayer_perceptron(x, weights, biases)

with tf.name_scope("Loss"):
		cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*batch_size)

with tf.name_scope("SGD"):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
		grads = tf.gradients(cost, tf.trainable_variables())
		grads = list(zip(grads, tf.trainable_variables()))		

# Create a summary to monitor cost tensor
tf.scalar_summary("loss", cost)

# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.histogram_summary(var.name, var)
# Summarize all gradients
for grad, var in grads:
    tf.histogram_summary(var.name + '/gradient', grad)

# Merge all summaries into a single op
merged_summary_op = tf.merge_all_summaries()


init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(LOG_DIR, tf.get_default_graph())
    
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, summary = sess.run(	[optimizer, cost, merged_summary_op], 
										feed_dict={x: batch_x, y: batch_y}
									)
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
    print "Optimization Finished!"


