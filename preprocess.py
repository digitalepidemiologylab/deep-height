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

snp_files = glob.glob(PATH_TO_INPUT_OPENSNP_DIRECTORY+"/splitconcatfiles/chrconcat*")

def _process_file(_file):

	print "Processing : ", _file
	#return {'user-id-100': {'data' : [100]}}
	user_ids = re.findall("chrconcat([\d]+)_([\d]+)", _file.split("/")[-1])
	assert len(user_ids) == 1 and len(user_ids[0]) == 2 and user_ids[0][0] == user_ids[0][1]
	user_id = int(user_ids[0][0])
	
	f = open(_file, "r")
	lines = f.readlines()
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
		"""
		data.append(np.fromstring(_line[0], dtype=np.uint8))
		data.append(np.fromstring(_line[1], dtype=np.uint8))

	f.close()
	data = np.array(data)
	return {user_id: data}

#Queue worker		
def _worker(files, done_queue):
	for _file in files:
		done_queue.put(_process_file(_file))
		print "Done processing : ", _file
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
	
	# Wait for all worker processes to finish
	#for p in procs:
	#	p.join()
	#	print "Stopping Process : ", p.name

	# Collect all results into single results dict.
	for _process_no in range(NUMBER_OF_PROCESSES):
		data_d.update(done_queue.get())
		print "Obtained result from _process_no...", _process_no

	for _process in procs:
		_process.terminate()
	
	while True:
		if any(p.is_alive() for p in procs):
			time.sleep(0.5)
		else:
			break


	train_d = dict(data_d.items()[int(len(opensnp_files)*1.0*TRAIN_PERCENTAGE/100):])
	test_d = dict(data_d.items()[:int(len(opensnp_files)*1.0*TRAIN_PERCENTAGE/100)])
	

	"""
	Read other extra information from the height.tsv file
	"""
	print "Reading heights.tsv"
	f = open(PATH_TO_INPUT_OPENSNP_DIRECTORY+"/height.tsv", "r")
	lines = f.readlines()
	HEADERS = lines[0].strip().split()
	assert "ID" in HEADERS
	id_index = HEADERS.index("ID")
	all_users = set(train_d.keys() + test_d.keys())

	for _line in lines[1:]:
		_line = _line.strip().split()
		assert len(_line) == len(HEADERS)
		#Make sure a corresponding data entry exists in the data dictionary
		assert _line[id_index] in all_users

		for _idx, _head in enumerate(HEADERS):
			#A corresponding entry has to exist in either train_d or test_d
			try:
				train_d[int(_line[id_index])][_head] = _line[_idx]
			except:
				test_d[int(_line[id_index])][_head] = _line[_idx]

	"""
	Shuffle and write data to train/test files 
	"""
	print "Writing data to the output folder...."
	def write_to_file(data, filename):
		dd.io.save(filename, data)

	write_to_file(train_d, PATH_TO_OUTPUT_DIRECTORY+"/train.h5")
	write_to_file(test_d, PATH_TO_OUTPUT_DIRECTORY+"/test.h5")
		

