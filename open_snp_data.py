#!/usr/bin/env python
import deepdish as dd
import numpy as np

def extract_data(filename, include_metadata = True):
    data = dd.io.load(filename)
    if include_metadata:
        X = data['X']/2
    else:
        X = data['X'][0:,3:]/2
    # Y = np.log(data['Y'])
    Y = data['Y']
    return (X, Y)

class DataSet:
	def __init__(self, filename, include_metadata=True):
		print "Loading data from : ", filename
		self._x, self._y = extract_data(filename, include_metadata)
		self._epochs_completed = 0
		self._index_in_epoch = 0

		assert len(self._x) == len(self._y)
		self._num_examples = len(self._x)

	@property
	def snps(self):
		return self._x

	@property
	def heights(self):
		return self._y

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size):
		start = self._index_in_epoch
		self._index_in_epoch += batch_size

		if self._index_in_epoch > self.num_examples:
			#Finished Epoch
			self._epochs_completed += 1

			#Shuffle the data
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._x = self._x[perm]
			self._y = self._y[perm]

			#Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples

		end = self._index_in_epoch
		return self._x[start:end], self._y[start:end]

	def save_small(self, target_filename, size):
		_d = {'X': self._x[:size], 'Y': self._y[:size]}
		dd.io.save(target_filename, _d)

def load_data(foldername, small=False, include_metadata=True):
	if not small:
		return (DataSet(foldername+"/train.h5", include_metadata), DataSet(foldername+"/test.h5", include_metadata))
	else:
		return (DataSet(foldername+"/train-small.h5", include_metadata), DataSet(foldername+"/test-small.h5", include_metadata))
