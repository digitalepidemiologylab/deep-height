#!/usr/bin/env python
import glob
import numpy as np
import deepdish as dd
import re
import os
import random
import math
import multiprocessing
import time

"""
ARGUMENTS

TO-DO: use a proper argument parser for these 
"""
PATH_TO_INPUT_OPENSNP_DIRECTORY = "/mount/SDG/splitconcatdataset-uncompressed/Genomic_data"
PATH_TO_OUTPUT_DIRECTORY = "/mount/SDG/deep-height/opensnp_data"

TRAIN_PERCENT = 80

NUMBER_OF_PROCESSES = multiprocessing.cpu_count()
"""
Iterate over, and collect all the data
"""
data_d = {}

snp_files = glob.glob(PATH_TO_INPUT_OPENSNP_DIRECTORY+"/splitconcatfiles/chrconcat*")[:100]
def _process_file(_file):

	print "Processing : ", _file
	#return {'user-id-100': {'data' : [100]}}
	user_ids = re.findall("chrconcat([\d]+)_([\d]+)\.txt", _file.split("/")[-1])
	print _file, user_ids
	assert len(user_ids) == 1 and len(user_ids[0]) == 2 and user_ids[0][0] == user_ids[0][1]
	user_id = str(user_ids[0][1].strip())
	
	f = open(_file, "r")
	lines = f.readlines()
	"""
	TO-DO: Make the assertion for expected number of lines configurable
	"""
	assert len(lines) == 9520940
	data = []
	for _line in lines:
		_line = _line.strip().split("/")
		assert len(_line) == 2

		"""
		TO-DO:
			The values can technically be packed in just 2bits,
			so quite some room for improvement here in how to pack 
			the data more efficiently.
			Some thought could also go into how to better "organise" 
			the data (or exploit some domain specific correlations between them).


		Binary Mapping
		0 -> 00
		1 -> 01
		2 -> 10
		3 -> Does not exist (Yet ?)
		"""
		def encode_binary(k):
			k = int(k)
			if k == 0:
				return [False, False]
			elif k == 1:
				return [False, True]
			elif k == 2:
				return [True, False]
			else:
				return None

		data += encode_binary(_line[0]) #Store as Boolean values
		data += encode_binary(_line[1])

	f.close()
	"""
	Pack Data
	"""
	return {user_id: {'data': data}}

#Queue worker		
def _worker(files, done_queue):
	for _file in files:
		done_queue.put(_process_file(_file))
		print "Done processing : ", _file
		"""
		try:
			done_queue.put(_process_file(_file))
			print "Done processing : ", _file
		except Exception, e:
			print "ERROR processing : ", _file
			f = open("ERROR", "a")
			f.write(_file+"\n")
			f.write(str(e) + "\n")
			f.write("="*100+"\n")
			f.close()
			done_queue.put(False)
		"""
	return	

if __name__ == '__main__':
	#Queue handler
	done_queue = multiprocessing.Queue() 
	chunksize = int(math.ceil(len(snp_files)/float(NUMBER_OF_PROCESSES)))
	procs = []

	#Start worker processes
	for _process_no in range(NUMBER_OF_PROCESSES):
		p = multiprocessing.Process(
			target = _worker,
			args = (snp_files[chunksize * _process_no : chunksize * (_process_no + 1)],
				done_queue))
		print "Starting Process : ", p.name
		procs.append(p)
		p.start()


	"""
	This section needs some cleanup. A bit too hacky right now.
	"""	
	#
	# Wait for all worker processes to finish
	# for p in procs:
	#	p.join()
	#	print "Stopping Process : ", p.name

	# Collect all results into single results dict.
	for _file_no in range(len(snp_files)):
		_result = done_queue.get()
		if _result:
			data_d.update(_result)
			print "Obtained result from file no...", _file_no
		else:
			print "ERROR in a particular file"
	#
	# Ideally we shouldnt have manual termination of the processes
	# But the darn processes simply wont terminate for reason with 
	# p.join :'(
	#
	# TO-DO: Fix this
	for _process in procs:
		_process.terminate()
	
	while True:
		if any(p.is_alive() for p in procs):
			time.sleep(0.5)
		else:
			break


	#Split into train and test splits
	train_d = {}
	test_d = {}
	_keys = data_d.keys()
	for _key in _keys:
		if random.randint(0, 1000)/10 <= TRAIN_PERCENT:
			train_d[_key] = data_d[_key]
		else:
			test_d[_key] = data_d[_key]
			
	print "Number of Training Keys : ",len(train_d.keys())
	print "Number of testing keys : ", len(test_d.keys())

	"""
	Read other extra information from the height.tsv file
	"""
	print "Reading heights.tsv"
	f = open(PATH_TO_INPUT_OPENSNP_DIRECTORY+"/height.tsv", "r")
	lines = f.readlines()
	HEADERS = lines[0].strip().split()
	assert "ID" in HEADERS
	id_index = HEADERS.index("ID")
	all_users = data_d.keys()

	for _line in lines[1:]:
		_line = _line.strip().split()
		if len(_line) == len(HEADERS):
			#Make sure a corresponding data entry exists in the data dictionary
			if str(_line[id_index]) in all_users:
				for _idx, _head in enumerate(HEADERS):
					#A corresponding entry has to exist in either train_d or test_d
						
					try:
						train_d[str(_line[id_index])][_head] = int(_line[_idx])
					except:
						test_d[str(_line[id_index])][_head] = int(_line[_idx])

	"""
	Shuffle and write data to train/test files 
	"""
	print "Writing data to the output folder...."
	def write_to_file(data, filename):
		dd.io.save(filename, data)

	write_to_file(train_d, PATH_TO_OUTPUT_DIRECTORY+"/train.h5")
	write_to_file(test_d, PATH_TO_OUTPUT_DIRECTORY+"/test.h5")
		


	"""
	Shuffle and write data to train/test files 
	"""
	print "Writing data to the output folder...."
	def write_to_file(data, filename):
		dd.io.save(filename, data)

	write_to_file(train_d, PATH_TO_OUTPUT_DIRECTORY+"/train.h5")
	write_to_file(test_d, PATH_TO_OUTPUT_DIRECTORY+"/test.h5")
		

