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
	def snps():
		return self._x

	def heights():
		return self._y
				

def load_data(foldername):
	return (DataSet(foldername+"/train.h5"), DataSet(foldername+"/test.h5"))
