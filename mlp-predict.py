#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import open_snp_data
import os
import random

LOG_DIR = "./logdir"

train, test = open_snp_data.load_data("opensnp_data/", small=False, include_metadata=False)

TEST_ON = train

input_dims = len(train.snps[0])

learning_rate = 0.001
training_epochs = 150
batch_size = 1
display_step = 1
checkpoint_step = 10
DETAILED_VISUALIZATION = False

n_hidden_1 = 20
n_hidden_2 = 20

n_input = input_dims

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)

    if DETAILED_VISUALIZATION:
        # Create a summary to visualize the first layer Sigmoid activation
        tf.histogram_summary("tanh1", layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.tanh(layer_2)

    if DETAILED_VISUALIZATION:
        # Create another summary to visualize the second layer Sigmoid activation
        tf.histogram_summary("tanh2", layer_2)
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
        if DETAILED_VISUALIZATION:
            grads = tf.gradients(cost, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))		

# Create a summary to monitor cost tensor
tf.scalar_summary("loss", cost)

if DETAILED_VISUALIZATION:
    # Create summaries to visualize weights
    for var in tf.trainable_variables():
        tf.histogram_summary(var.name, var)
    # Summarize all gradients
    for grad, var in grads:
        tf.histogram_summary(var.name + '/gradient', grad)

# Merge all summaries into a single op
merged_summary_op = tf.merge_all_summaries()

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('checkpoints/')
    #ckpt.model_checkpoint_path = "checkpoints/model.ckpt-20"
    if ckpt and ckpt.model_checkpoint_path:
        # if checkpoint exists, restore the parameters and set epoch_n and i_iter
        saver.restore(sess, ckpt.model_checkpoint_path)
        epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
        i_iter = (epoch_n+1) * (train.num_examples/batch_size)
        print "Restored Epoch ", epoch_n
    else:
        # no checkpoint exists. create checkpoints directory if it does not exist.
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')

        epoch_n = 0
        init = tf.initialize_all_variables()
        sess.run(init)
    
    DATASETS = [train, test]
    for TEST_ON in DATASETS: 
        #summary_writer = tf.train.SummaryWriter(LOG_DIR, tf.get_default_graph())
        total_batch = int(TEST_ON.num_examples/batch_size)
        avg_cost = 0
        count = 0  
        predictions = []
        heights = [] 
        for i in range(total_batch):
            batch_x, batch_y = TEST_ON.next_batch(batch_size)
            batch_x = batch_x
            batch_y = batch_y

            _cost, prediction = sess.run([cost, pred], feed_dict = {x: batch_x, y: batch_y})
            predictions.append(prediction[0][0]*2)
            heights.append(batch_y[0]*2)
            print predictions[-1], heights[-1]
            count += 1
            avg_cost += _cost/count
            print _cost, avg_cost
            print "="*100
        print "Plotting graph..."
        x_min = np.amin(heights, axis=0) - 10
        x_max = np.amax(heights, axis=0) + 10
        
        y_min = np.amin(predictions, axis=0) - 10
        y_max = np.amin(predictions, axis=0) + 10

        plt.clf() 
        #plt.plot([x_min, y_min], [x_max, y_max], 'r-')

        plt.scatter(heights, predictions)
        plt.xlabel("Actual Heights")
        plt.ylabel("Predicted Heights")
        if TEST_ON == train:
            plt.savefig("train.png")
        else:
            plt.savefig("test.png")
         
