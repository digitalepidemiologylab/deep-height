#!/usr/bin/env python
import deepdish as dd
import numpy as np

def extract_data(filename):
	data = dd.io.load(filename)
	X = data['X']
	Y = data['Y']
	return (X, Y)

class DataSet:
	def __init__(self, filename):
		print "Loading data from : ", filename
		self._x, self._y = extract_data(filename)
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

def load_data(foldername):
	return (DataSet(foldername+"/train.h5"), DataSet(foldername+"/test.h5"))
