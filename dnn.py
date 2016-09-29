#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from bnf import *

import os

import open_snp_data
LOG_Y = True

train, test = open_snp_data.load_data("opensnp_data/", small=False, log_y=LOG_Y)

input_dims = len(train.snps[0])

"""
Hyperparameters
"""
training_epochs = 100
batch_size = 1

dropout = 1

filt_1 = [20, 1000, 10]  #Configuration for conv1 in [num_filt,kern_size,pool_stride]
filt_2 = [20, 1000, 10]

num_fc_1 = 200
num_fc_2 = 200

learning_rate = 1e-2

checkpoint_step = 5
if LOG_Y:
    LOGDIR = "logdir_conv_logy"
    CHECKPOINTS = "checkpoints_conv_logy"
else:
    LOGDIR = "logdir_conv"
    CHECKPOINTS = "checkpoints_conv"

x = tf.placeholder("float", shape=[None, input_dims], name = 'Input_data')
y_ = tf.placeholder("float", shape=[None], name = 'height')

keep_prob = tf.placeholder("float", name = 'dropout_keep_prob')
bn_train = tf.placeholder(tf.bool)          #Boolean value to guide batchnorm


def weight_variable(shape, name):
    with tf.device("/cpu:0"):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name = name)

def bias_variable(shape, name):
    with tf.device("/cpu:0"):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name = name)

def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

with tf.name_scope("Reshaping_data") as scope:
  x_image = tf.reshape(x, [-1,input_dims,1,1])

"""
Build the graph
"""
with tf.name_scope("Conv1") as scope:
  W_conv1 = weight_variable([filt_1[1], 1, 1, filt_1[0]], 'Conv_Layer_1')
  b_conv1 = bias_variable([filt_1[0]], 'bias_for_Conv_Layer_1')
  a_conv1 = conv2d(x_image, W_conv1) + b_conv1
  h_conv1 = tf.nn.tanh(a_conv1)


with tf.name_scope('max_pool1') as scope:
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, filt_1[2], 1, 1],
                        strides=[1, filt_1[2], 1, 1], padding='VALID')
                        #width is now (128-4)/2+1
    width_pool1 = int(np.floor((input_dims-filt_1[2])/filt_1[2]))+1
    size1 = tf.shape(h_pool1)       #Debugging purposes


with tf.name_scope("Conv2") as scope:
  W_conv2 = weight_variable([filt_2[1], 1, filt_1[0], filt_2[0]], 'Conv_Layer_2')
  b_conv2 = bias_variable([filt_2[0]], 'bias_for_Conv_Layer_2')
  a_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
  h_conv2 = a_conv2
  h_conv2 = tf.nn.tanh(a_conv2)

with tf.name_scope('max_pool2') as scope:
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, filt_2[2], 1, 1],
                        strides=[1, filt_2[2], 1, 1], padding='VALID')
                        #width is now (128-4)/2+1
    width_pool2 = int(np.floor((width_pool1-filt_2[2])/filt_2[2]))+1
    size2 = tf.shape(h_pool2)       #Debugging purposes

with tf.name_scope('Batch_norm1') as scope:
    a_bn1 = batch_norm(h_pool2,filt_2[0],bn_train,'bn2')
    h_bn1 = tf.nn.tanh(a_bn1)


with tf.name_scope("Fully_Connected1") as scope:
# Now we proces the final information with a fully connected layer. We convert
# both activations over all channels into one 1D tensor per sample.
# We have "filt_2[0]" channels and "width_pool2" activations per channel.
# Hence we use "width_pool2*filt_2[0]" i this first line
  W_fc1 = weight_variable([width_pool2*filt_2[0], num_fc_1], 'Fully_Connected_layer_1')
  b_fc1 = bias_variable([num_fc_1], 'bias_for_Fully_Connected_Layer_1')
  h_flat = tf.reshape(h_bn1, [-1, width_pool2*filt_2[0]])
  h_flat = tf.nn.dropout(h_flat,keep_prob)
  h_fc1 = tf.nn.tanh(tf.matmul(h_flat, W_fc1) + b_fc1)

with tf.name_scope("Fully_Connected2") as scope:
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = tf.Variable(tf.truncated_normal([num_fc_1, num_fc_2], stddev=0.1),name = 'W_fc2')
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[num_fc_2]),name = 'b_fc2')
    h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    size3 = tf.shape(h_fc2)       #Debugging purposes

with tf.name_scope("Output_layer") as scope:
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    W_fc3 = tf.Variable(tf.truncated_normal([num_fc_2, batch_size], stddev=0.1),name = 'W_fc3')
    b_fc3 = tf.Variable(tf.constant(0.1, shape=[batch_size]),name = 'b_fc3')
    h_fc3 = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    size3 = tf.shape(h_fc3)       #Debugging purposes

with tf.name_scope("Loss"):
    cost = tf.reduce_sum(tf.pow(h_fc3-y_, 2))/(2*batch_size)

with tf.name_scope("train") as scope:
    # tvars = tf.trainable_variables()
    # grads = tf.gradients(cost, tvars)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cost)
    # gradients = zip(grads, tvars)

    # numel = tf.constant([[0]])
    # for gradient, variable in gradients:
    #     if isinstance(gradient, ops.IndexedSlices):
    #         grad_values = gradient.values
    #     else:
    #         grad_values = gradient
    #     numel +=tf.reduce_sum(tf.size(variable))
    #
    #     h1 = tf.histogram_summary(variable.name, variable)
    #     h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)
    #     h3 = tf.histogram_summary(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))

# Create a summary to monitor cost tensor
tf.scalar_summary("loss", cost)
merged_summary_op = tf.merge_all_summaries()

saver = tf.train.Saver()


with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    ckpt = tf.train.get_checkpoint_state(CHECKPOINTS)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
        i_iter = (epoch_n) * (train.num_examples/batch_size)
        print "Restored Epoch ", epoch_n
    else:
        if not os.path.exists(CHECKPOINTS):
            os.makedirs(CHECKPOINTS)

        epoch_n = 0
        sess.run(tf.initialize_all_variables())

    writer = tf.train.SummaryWriter(LOGDIR, sess.graph_def)

    # Training cycle
    for epoch in range(epoch_n, training_epochs):
        avg_cost = 0.
        total_batch = int(train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, summary, prediction = sess.run(   [train_op, cost, merged_summary_op, h_fc3],
                                        feed_dict={x: batch_x, y_: batch_y, bn_train: True, keep_prob: dropout}
                                    )
            writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
            if LOG_Y:
                print "Cost : ", c, "Logg-ed-Prediction : ", prediction, "Transformed Prediction : ", np.exp(prediction), " Actual : ", batch_y, "Transformed Actual : ", np.exp(batch_y)
            else:
                print "Cost : ", c, "Prediction : ", prediction, "Actual : ", batch_y


        print "epoch : ", epoch, "avg_cost : ", avg_cost
        # if epoch % checkpoint_step == 0 :
        #     print "Saving checkpoint...."
        #     saver.save(sess, CHECKPOINTS + '/model.ckpt', epoch)
        #     print "Checkpoint saved...."
