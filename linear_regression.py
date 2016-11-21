#!/usr/bin/env python

import tensorflow as tf
import numpy as np

import open_snp_data

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns



train, test = open_snp_data.load_data("opensnp_data_gwas_subset/", small=False, include_metadata=True)

print test.snps[0]

# Parameters
learning_rate = 0.01
training_epochs = 100
batch_size = 10
display_step = 1


# tf Graph Input
x = tf.placeholder(tf.float32, [batch_size, len(train.snps[0])])
y = tf.placeholder(tf.float32, [batch_size])

# Set model weights
W = tf.Variable(tf.zeros([batch_size, len(train.snps[0])]))
b = tf.Variable(tf.zeros([batch_size]))

# Construct model
pred = tf.add(tf.matmul(x, tf.transpose(W)),  tf.transpose(b))

# Minimize error using Mean Squared Error
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
    #Predicting heights
    M_PREDS = []
    M_TRUE = []

    F_PREDS = []
    F_TRUE = []

    total_batch = int(test.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = test.next_batch(batch_size)
        predictions = sess.run([pred], feed_dict={x: batch_xs, y: batch_ys})

        gender = batch_xs[0:, 1]
        y_pred = predictions[0][0].tolist()
        y_true = batch_ys.tolist()

        for _idx, _female in enumerate(gender):
            if _female == True:
                F_PREDS.append(y_pred[_idx])
                F_TRUE.append(y_true[_idx])
            else:
                M_PREDS.append(y_pred[_idx])
                M_TRUE.append(y_true[_idx])

        #PREDS += y_pred
        #TRUE += y_true

plt.clf()
plt.scatter(M_PREDS, M_TRUE, color='b')
plt.scatter(F_PREDS, F_TRUE, color='r')

np.save("M_PREDS.npy", M_PREDS)
np.save("M_TRUE.npy", M_TRUE)
np.save("F_PREDS.npy", F_PREDS)
np.save("F_TRUE.npy", F_TRUE)


plt.ylabel("Actual Heights")
plt.xlabel("Predicted Heights")
plt.savefig("linear-train-scatter.png")
