#!/usr/bin/env python
import deepdish as dd
import numpy as np

def boolean_map(l):
	assert (l[0] & l[1]) != True

	if l[0] == False and l[1] == False:
		return 0
	elif l[0] == False and l[1] == True:
		return 1
	elif l[0] == True and l[1] == False:
		return 2
	else:
		return False
	
def unpack(d):
	unpacked = []
	_buffer = []
	for idx, k in enumerate(d):
		if idx % 4 == 0 and len(_buffer) == 4:
			unpacked += [boolean_map(_buffer[:2])]
			unpacked += [boolean_map(_buffer[2:])]
		else:
			_buffer.append(k)
	return unpacked

def extract_data(filename):
	data = dd.io.load(filename)
	X = []
	Y = []
	for _key in data.keys():
		_x = unpack(data[_key]['data'])
		_x = [	data[_key]['BORN'],
			data[_key]['SEX'] ] + _x
		_x = np.array(_x, dtype=np.uint8)
		X.append(_x)
		Y.append(np.uint8(data[_key]['HEIGHT']))

	X = np.array(X)
	Y = np.array(Y)

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
