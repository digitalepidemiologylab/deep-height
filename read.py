#!/usr/bin/env python


import open_snp_data

train, test = open_snp_data.load_data("opensnp_data")
print train, test
"""
train.save_small("opensnp_data/train-small.h5", 20)
test.save_small("opensnp_data/test-small.h5", 20)
print "Saved small..."
"""
