import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from bnf import *


def buildCNN(x, y_, input_dims, bn_train, keep_prob, batch_size, training_epochs, LOGDIR, CHECKPOINTS,):
    """
    Hyperparameters
    """
    filt_1 = [20, 1000, 10]  #Configuration for conv1 in [num_filt,kern_size,pool_stride]
    filt_2 = [20, 1000, 10]

    num_fc_1 = 200
    num_fc_2 = 200

    learning_rate = 1e-2

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
        pred = h_fc3

    with tf.name_scope("Loss"):
        cost = tf.reduce_sum(tf.pow(h_fc3-y_, 2))/(2*batch_size)

    with tf.name_scope("train") as scope:
        # tvars = tf.trainable_variables()
        # grads = tf.gradients(cost, tvars)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(cost)
        _realtime_learning_rate = optimizer._lr_t

    # Create a summary to monitor cost tensor
    tf.scalar_summary("loss", cost)
    # Create a summary to monitor learning rate
    tf.scalar_summary("Learning Rate", _realtime_learning_rate)

    return (pred, cost, train_op)
