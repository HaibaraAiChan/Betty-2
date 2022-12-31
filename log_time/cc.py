import os
import numpy as np
import pandas as pd
from statistics import mean
import argparse
import sys


def data_collect(filename):
	items =[]
	group = []
	split = []
	
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("torch.sort() replace to OrderedDict time:") :
				items.append(float(line.split(' ')[-1]))

	ttemp = 1+32
	for i in range(0,len(items),ttemp):
		# print(i)
		group.append(items[i])
	print('grouping',len(group))
	print('grouping mean',mean(group))

	for j in range(1,ttemp):
		for i in range(j,len(items),ttemp):
			# print(i)
			split.append(items[i])
	print('split',len(split))
	print('splittingg mean',mean(split))


if __name__=='__main__':
	
	for filename in os.listdir('./'):
		if filename.endswith(".log"):
			file = os.path.join('./', filename)
			if '_split_32.' in file:
				print(file)
				data_collect(file)
				print()
	
	
		





