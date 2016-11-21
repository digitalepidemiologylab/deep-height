#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


M_TRUE = np.load("M_TRUE.npy")
M_PREDS = np.load("M_PREDS.npy")
F_TRUE = np.load("F_TRUE.npy")
F_PREDS = np.load("F_PREDS.npy")
combined_TRUE = np.array(M_TRUE.tolist() + F_TRUE.tolist())
combined_PREDS = np.array(M_PREDS.tolist() + F_PREDS.tolist())

TRUE = [(M_TRUE, M_PREDS) , (F_TRUE, F_PREDS) , (combined_TRUE, combined_PREDS)]
LABELS = ["M", "F", "combined"]

_idx = 0
for _true, _preds in TRUE:
    THRESHOLDS = []
    ACCURACIES = []
    print "====================================="

    for threshold in np.arange(0, 20.5, 0.5):
        correct = np.abs(_true - _preds) < threshold
        _accuracy = np.sum(correct) * 1.0 / len(correct) * 100
        THRESHOLDS.append(threshold)
        ACCURACIES.append(_accuracy)
        print "TYPE : ",LABELS[_idx], ", Threshold : ", threshold, ", Accuracy : ", _accuracy, "%"

    plt.clf()
    plt.plot(THRESHOLDS, ACCURACIES)
    plt.savefig("plots/"+LABELS[_idx]+".png")
    _idx += 1
