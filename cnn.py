import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from bnf import *


def buildCNN(x, y_, input_dims, learning_rate, bn_train, keep_prob, batch_size, training_epochs, LOGDIR, CHECKPOINTS):
    """
    Hyperparameters
    """
    filters = [
        [20, 1000, 10], #Configuration for conv1 in [num_filt,kern_size,pool_stride]
        [20, 1000, 10],
        [5, 5000, 10]
    ]
    fc_layers = [200, 200, batch_size]

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
    _input = x_image
    _previous_filter_0 = 1
    _previous_widthpool = input_dims
    for _idx, _filter in enumerate(filters):
        with tf.name_scope("Conv_"+str(_idx+1)) as scope:
            W_conv = weight_variable([_filter[1], 1, _previous_filter_0, _filter[0]], 'Conv_Layer_'+str(_idx+1))
            b_conv = bias_variable([_filter[0]], 'bias_for_Conv_Layer_'+str(_idx+1))
            a_conv = conv2d(_input, W_conv) + b_conv
            h_conv = a_conv
            h_conv = tf.nn.tanh(a_conv)

        with tf.name_scope('max_pool'+str(_idx+1)) as scope:
            h_pool = tf.nn.max_pool(h_conv, ksize=[1, _filter[2], 1, 1],
                                strides=[1, _filter[2], 1, 1], padding='VALID')
                                #width is now (128-4)/2+1
            width_pool = int(np.floor((_previous_widthpool-_filter[2])/_filter[2]))+1
            size2 = tf.shape(h_pool)       #Debugging purposes

        _previous_widthpool = width_pool
        _previous_filter_0 = _filter[0]
        _input = h_pool
        _input_size = _previous_widthpool*_previous_filter_0

    with tf.name_scope('Batch_norm') as scope:
        a_bn1 = batch_norm(_input,_previous_filter_0,bn_train,'bn2')
        h_bn1 = tf.nn.tanh(a_bn1)
        _input = h_bn1

    for _idx, num_fc in enumerate(fc_layers):
        with tf.name_scope("Fully_Connected-"+str(_idx+1)) as scope:
            # Now we proces the final information with a fully connected layer. We convert
            # both activations over all channels into one 1D tensor per sample.
            # We have "filt_2[0]" channels and "width_pool2" activations per channel.
            # Hence we use "width_pool2*filt_2[0]" i this first line
            if _idx == 0:
                _input_drop = _input
            else:
                _input_drop = tf.nn.dropout(_input, keep_prob)

            W_fc = weight_variable([_input_size, num_fc], 'Fully_Connected_layer_'+str(_idx+1))
            b_fc = bias_variable([num_fc], 'bias_for_Fully_Connected_Layer_'+str(_idx+1))
            h_flat = tf.reshape(_input_drop, [-1, _input_size])
            h_flat = tf.nn.dropout(h_flat,keep_prob)

            h_fc = tf.matmul(h_flat, W_fc) + b_fc
            h_fc_tanh = tf.nn.tanh(h_fc)
            _input = h_fc_tanh
            _input_size = num_fc
            _input_unactivated = h_fc

    pred = _input_unactivated

    with tf.name_scope("Loss"):
        cost = tf.reduce_sum(tf.pow(pred-y_, 2))/(2*batch_size)

    with tf.name_scope("train") as scope:
        # tvars = tf.trainable_variables()
        # grads = tf.gradients(cost, tvars)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(cost)
        _realtime_learning_rate = optimizer._lr_t

    # Create a summary to monitor cost tensor
    tf.scalar_summary("loss", cost)
    # Create a summary to monitor learning rate
    tf.scalar_summary("Learning Rate", _realtime_learning_rate)

    return (pred, cost, train_op)
