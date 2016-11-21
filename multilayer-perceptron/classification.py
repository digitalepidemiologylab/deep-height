#!/usr/bin/env python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report
import pandas as pd
import numpy as np
import open_snp_data
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

LOG_DIR = "./logdir-mlp"

EPOCHS = 100

train, test = open_snp_data.load_data("../opensnp_data_gwas_subset/", small=False, include_metadata=True)

input_dims = len(train.snps[0])

# Initialize the MLP
def initialize_nn(frame_size, class_size):
    model = Sequential() # The Keras Sequential model is a linear stack of layers.
    model.add(Dense(10, init='he_normal', activation='tanh', input_dim=frame_size)) # Dense layer
    model.add(Dropout(0.01))
    model.add(Dense(10, init='he_normal', activation='tanh')) # Another dense layer
    model.add(Dropout(0.01))
    model.add(Dense(class_size, activation='softmax')) # Last dense layer with Softmax

    # model.compile(loss='mean_squared_error', optimizer="adam")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model

BINS = [0, 169, 177, 1000]
def compute_class_indices(Y, BINS):
    classIndices = []
    for _y in Y.tolist():
        result = False
        for _idx in range(len(BINS)-1):
            if _y > BINS[_idx] and _y <= BINS[_idx+1]:
                result = _idx
        _result = np.zeros(len(BINS)-1).tolist()
        _result[result] = 1
        classIndices.append(np.array(_result))
    return np.array(classIndices)

train_X = train.snps
train_Y = compute_class_indices(train.heights, BINS)


test_X = test.snps
test_Y = compute_class_indices(test.heights, BINS)



total_X = np.array(train.snps.tolist() + test.snps.tolist())
total_Y = compute_class_indices(np.array(train.heights.tolist() + test.heights.tolist()), BINS)

model = initialize_nn(input_dims, len(BINS)-1)
model.fit(train_X, train_Y, nb_epoch=EPOCHS, batch_size=16)
score = model.evaluate(test_X, test_Y, batch_size=16)

model.save('classification-data/regression_model.h5')  # creates a HDF5 file 'my_model.h5'

_predictions = model.predict(test_X)

_predictions_total = model.predict(total_X)

np.save("classification-data/PREDS.npy", _predictions)
np.save("classification-data/PREDS_total.npy", _predictions_total)
np.save("classification-data/TRUE.npy", test_Y)
np.save("classification-data/TRUE_total.npy", total_Y)

print ""
print classification_report(np.argmax(total_Y,axis=1), np.argmax(_predictions_total,axis=1), digits=4)


# plt.clf()
# sns.heatmap(np.array(np.argmax(test_Y, axis=1)), np.array(np.argmax(_predictions, axis=1)))
# plt.savefig("classification-data/scatter.png")
# plt.clf()
# sns.heatmap(np.array(np.argmax(total_Y, axis=1)),  np.array(np.argmax(_predictions_total, axis=1)))
# plt.savefig("classification-data/scatter-total.png")
