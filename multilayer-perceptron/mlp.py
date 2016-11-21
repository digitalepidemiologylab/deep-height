#!/usr/bin/env python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
import open_snp_data
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

LOG_DIR = "./logdir-mlp"

train, test = open_snp_data.load_data("../opensnp_data_gwas_subset/", small=False, include_metadata=True)

input_dims = len(train.snps[0])

# Initialize the MLP
def initialize_nn(frame_size):
    model = Sequential() # The Keras Sequential model is a linear stack of layers.
    model.add(Dense(684, init='he_normal', activation='linear', input_dim=frame_size)) # Dense layer
    model.add(Dropout(0.01))
    model.add(Dense(684, init='he_normal', activation='linear')) # Another dense layer
    model.add(Dropout(0.01))
    model.add(Dense(1, init='he_normal', activation='linear')) # Last dense layer
    # model.compile(loss='mean_squared_error', optimizer="adam")
    model.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=["mean_absolute_error"])
    return model

train_X = train.snps
train_Y = train.heights

test_X = test.snps
test_Y = test.heights

total_X = np.array(train.snps.tolist() + test.snps.tolist())
total_Y = np.array(train.heights.tolist() + test.heights.tolist())

model = initialize_nn(input_dims)
model.fit(train_X, train_Y, nb_epoch=20, batch_size=16)
score = model.evaluate(test_X, test_Y, batch_size=16)

model.save('regression-data/regression_model.h5')  # creates a HDF5 file 'my_model.h5'

_predictions = model.predict(test_X)

_predictions_total = model.predict(total_X)

np.save("regression-data/PREDS.npy", _predictions)
np.save("regression-data/PREDS_total.npy", _predictions_total)
np.save("regression-data/TRUE.npy", test_Y)
np.save("regression-data/TRUE_total.npy", total_Y)


plt.clf()
plt.scatter(test_Y, _predictions)
plt.savefig("regression-data/scatter.png")
plt.clf()
plt.scatter(total_Y, _predictions_total)
plt.savefig("regression-data/scatter-total.png")
