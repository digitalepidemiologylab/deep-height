#!/usr/bin/env python
import glob
import numpy as np
import deepdish as dd
import re
import random

"""
ARGUMENTS
"""
PATH_TO_INPUT_OPENSNP_DIRECTORY = "/mount/SDG/splitconcatdataset-uncompressed/Genomic_data"
PATH_TO_OUTPUT_DIRECTORY = "/mount/SDG/deep-height/opensnp_data"

TRAIN_PERCENT = 80

"""
Iterate over, and collect all the data
"""
train_d = {}
test_d = {}
for _file in glob.glob(PATH_TO_INPUT_OPENSNP_DIRECTORY+"/splitconcatfiles/chrconcat*"):
	print "Processing : ", _file
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

	if random.randint(0,1000)/10 <= TRAIN_PERCENT:
		train_d[user_id] = {}
		train_d[user_id]['data'] = data
	else:
		test_d[user_id] = {}
		test_d[user_id]['data'] = data
		

"""
Read other extra information from the heights.tsv file
"""
print "Reading heights.tsv"
f = open(PATH_TO_INPUT_OPENSNP_DIRECTORY+"/heights.tsv", "r")
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
	

