#!/usr/bin/env python

import tensorflow as tf
import numpy as np

import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import cnn

import open_snp_data

RUN_NAME = "dropout-0.5"
DEBUG_MODE = True

LOG_Y = False
INCLUDE_METADATA = True

LOGDIR = "logdir"
CHECKPOINTS = "checkpoints"

LOGDIR += "/"+RUN_NAME
CHECKPOINTS += "/"+RUN_NAME

batch_size = 1
training_epochs = 100
checkpoint_step = False


train, test = open_snp_data.load_data("opensnp_data/", small=DEBUG_MODE, include_metadata=INCLUDE_METADATA, log_y=LOG_Y)

input_dims = len(train.snps[0])

x = tf.placeholder("float", shape=[None, input_dims], name = 'Input_data')
y_ = tf.placeholder("float", shape=[None], name = 'height')

"""
Build CNN
"""
dropout = 0.5
bn_train = tf.placeholder(tf.bool)          #Boolean value to guide batchnorm
keep_prob = tf.placeholder("float", name = 'dropout_keep_prob')
preds, cost, train_op = cnn.buildCNN(   x, y_,
                                        input_dims,
                                        bn_train,
                                        keep_prob,
                                        batch_size,
                                        training_epochs,
                                        LOGDIR,
                                        CHECKPOINTS
                                    )


merged_summary_op = tf.merge_all_summaries()
saver = tf.train.Saver()

config=tf.ConfigProto(  log_device_placement=True,
                        allow_soft_placement=True)
with tf.Session(config=config) as sess:
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
        train_actual = []
        train_predicted = []
        if INCLUDE_METADATA == True:
            train_gender = []

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, summary, prediction = sess.run(   [train_op, cost, merged_summary_op, preds],
                                        feed_dict={x: batch_x, y_: batch_y, bn_train: True, keep_prob: dropout}
                                    )
            writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
            if LOG_Y:
                print "Cost : ", c, "Logg-ed-Prediction : ", prediction, "Transformed Prediction : ", np.exp(prediction), " Actual : ", batch_y, "Transformed Actual : ", np.exp(batch_y)
                train_actual.append(np.exp(batch_y)[0])
                train_predicted.append(np.exp(prediction)[0][0])
            else:
                print "Cost : ", c, "Prediction : ", prediction, "Actual : ", batch_y
                train_actual.append(batch_y[0])
                train_predicted.append(prediction[0][0])

            if INCLUDE_METADATA == True:
                train_gender.append(batch_x[0][1]%2)

        ##Plot scatter plot for training set
        print "Plotting train-predictions scatter plot for Epoch : ", epoch
        plt.clf()
        if INCLUDE_METADATA == True:
            #Handle case of Gender coloring
            blue_actual = []
            blue_predicted = []
            red_actual = []
            red_predicted = []
            for idx, k in enumerate(train_gender):
                if train_gender[idx] == 0:
                    blue_actual.append(train_actual[idx])
                    blue_predicted.append(train_predicted[idx])
                else:
                    red_actual.append(train_actual[idx])
                    red_predicted.append(train_predicted[idx])

            plt.scatter(blue_actual, blue_predicted, color='b')
            plt.scatter(red_actual, red_predicted, color='r')
        else:
            plt.scatter(train_actual, train_predicted)

        plt.xlabel("Actual Heights")
        plt.ylabel("Predicted Heights")
        plt.savefig(LOGDIR+"/train-scatter-"+str(epoch)+".png")
        print "epoch : ", epoch, "avg_cost : ", avg_cost
        if checkpoint_step and (epoch % checkpoint_step == 0) :
            print "Saving checkpoint...."
            saver.save(sess, CHECKPOINTS + '/model.ckpt', epoch)
            print "Checkpoint saved...."
