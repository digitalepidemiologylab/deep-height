#!/usr/bin/env python

import numpy as np
import open_snp_data
import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Input, Dropout

from keras.callbacks import ModelCheckpoint

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns




train, test = open_snp_data.load_data("opensnp_data_gwas_subset/", small=False, include_metadata=True)
EPOCHS=100

model = Sequential()
model.add(Dense(100, input_dim=train.snps.shape[1], activation='sigmoid'))
model.add(Dense(100, activation='sigmoid', init='normal'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mean_squared_error', 'mean_squared_logarithmic_error'])


filepath="keras_weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='mean_squared_error', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(train.snps, train.heights, nb_epoch=EPOCHS, batch_size=100)

y_pred = model.predict(np.concatenate((test.snps,train.snps), axis=0))
y_pred = y_pred.reshape((y_pred.shape[0],))
y_true = np.concatenate((test.heights,train.heights), axis=0)
gender = np.concatenate((test.snps[0:, 1].T,train.snps[0:, 1]), axis=0)

#Predicting heights
M_PREDS = []
M_TRUE = []

F_PREDS = []
F_TRUE = []


for _idx, _female in enumerate(gender):
        if _female == True:
            F_PREDS.append(y_pred[_idx])
            F_TRUE.append(y_true[_idx])
        else:
            M_PREDS.append(y_pred[_idx])
            M_TRUE.append(y_true[_idx])
plt.clf()
plt.scatter(M_TRUE, M_PREDS, color='b')
plt.scatter(F_TRUE, F_PREDS, color='r')
plt.xlabel("Actual Heights")
plt.ylabel("Predicted Heights")
plt.savefig("keras_mlp_scatter.png")
