#!/usr/bin/env python

import tensorflow as tf
import numpy as np

import os
import shutil
import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


import open_snp_data

LOGDIR = "logdir"
CHECKPOINTS = "checkpoints"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('-n','--name', help='Current Run Name', required=True)
parser.add_argument('-e','--epochs', help='Number of Epochs', type=int, default=100)
parser.add_argument('-m','--include_metadata', help='Include Metadata?', default=True, type=bool)
parser.add_argument('-l','--log_y', help="Move Y to Log Space", action='store_true')
parser.add_argument('-d','--debug', help="Debug Mode", action='store_true')
parser.add_argument('-g','--gwas', help="Train on GWAS Subset", action='store_true')
parser.add_argument('-r','--reset_logs', help="Reset LOGDIR and CHECKPOINTS directories", action='store_true')

args = vars(parser.parse_args())

RUN_NAME = args['name']
INCLUDE_METADATA = args['include_metadata']
LOG_Y = args['log_y']
DEBUG_MODE = args['debug']

LOGDIR += "/"+RUN_NAME
CHECKPOINTS += "/"+RUN_NAME

if args['reset_logs']:
    if os.path.exists(LOGDIR):
        shutil.rmtree(LOGDIR)
    if os.path.exists(CHECKPOINTS):
        shutil.rmtree(CHECKPOINTS)

batch_size = 8
training_epochs = args['epochs']
checkpoint_step = 10

if args['gwas']:
    import gwas_cnn as cnn
    train, test = open_snp_data.load_data("opensnp_data_gwas_subset/", small=DEBUG_MODE, include_metadata=INCLUDE_METADATA, log_y=LOG_Y)
    print "GWAS Subset Mode...."
else:
    import cnn
    train, test = open_snp_data.load_data("opensnp_data/", small=DEBUG_MODE, include_metadata=INCLUDE_METADATA, log_y=LOG_Y)


input_dims = len(train.snps[0])

x = tf.placeholder("float", shape=[None, input_dims], name = 'Input_data')
y_ = tf.placeholder("float", shape=[None], name = 'height')

"""
Build CNN
"""
dropout = 0.5
_learning_rate = 1e-1
decay = 0.2
learning_rate = tf.placeholder(tf.float32, shape=[])
bn_train = tf.placeholder(tf.bool)          #Boolean value to guide batchnorm
keep_prob = tf.placeholder("float", name = 'dropout_keep_prob')
preds, cost, train_op =     cnn.buildCNN(   x, y_,
                                        input_dims,
                                        learning_rate,
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

        e_learning_rate = _learning_rate * 1 / (1 + decay * ((epoch*1.0/training_epochs) * 100))
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, summary, prediction = sess.run(   [train_op, cost, merged_summary_op, preds],
                                        feed_dict={ x: batch_x, y_: batch_y,
                                                    bn_train: True,
                                                    keep_prob: dropout,
                                                    learning_rate: e_learning_rate}
                                    )
            writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
            if LOG_Y:
                print "Cost : ", c, "Logg-ed-Prediction : ", prediction, "Transformed Prediction : ", np.exp(prediction), " Actual : ", batch_y, "Transformed Actual : ", np.exp(batch_y)
                for b in range(batch_size):
                    train_actual.append(np.exp(batch_y)[b])
                    train_predicted.append(np.exp(prediction)[b][0])
            else:
                print "Cost : ", c, "Prediction : ", prediction, "Actual : ", batch_y
                for b in range(batch_size):
                    train_actual.append(batch_y[b])
                    train_predicted.append(prediction[b][0])

            if INCLUDE_METADATA == True:
                for b in range(batch_size):
                    train_gender.append(batch_x[b][1]%2)

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
        print "Learning Rate : ", e_learning_rate
        if checkpoint_step and (epoch % checkpoint_step == 0) :
            print "Saving checkpoint...."
            saver.save(sess, CHECKPOINTS + '/model.ckpt', epoch)
            print "Checkpoint saved...."
