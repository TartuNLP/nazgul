#!/usr/bin/env python3

import sys

from collections import defaultdict
from operator import itemgetter
from datetime import datetime

def log(msg):
	sys.stderr.write("{0}: {1}\n".format(str(datetime.now()), msg))

def learnModel(lines):
	log("learning")
	rawmodel = defaultdict(lambda: defaultdict(int))
	
	logFreq = 100000
	i = 0
	for line in lines:
		words = line.strip().split()
		
		for word in words:
			rawmodel[word.lower()][word] += 1
		
		i += 1
		if not i % logFreq:
			log("read {0} lines".format(i))
	
	if i % logFreq:
		log("read {0} lines".format(i))
	return compressModel(rawmodel)

def compressModel(rawmodel):
	log("compressing")
	model = list()
	
	for key in rawmodel:
		sortedItems = sorted(rawmodel[key].items(), key=itemgetter(1), reverse=True)
		totFreq = sum(rawmodel[key].values())
		
		if totFreq > 1:
			if (len(sortedItems) > 1 and sortedItems[0][1] == sortedItems[1][1]):
				winner = max(sortedItems[0][0], sortedItems[1][0])
			else:
				winner = sortedItems[0][0]
			model.append(winner)
	
	return model

def saveModel(model, fh):
	log("saving")
	for w in model:
		fh.write(w)
		fh.write("\n")
	
if __name__ == '__main__':
	try:
		filename = sys.argv[1]
	except IndexError:
		filename = '-'
	
	if filename == '-':
		model = learnModel(sys.stdin)
	else:
		with open(filename, 'r') as fh:
			model = learnModel(fh)
	
	try:
		outputFile = sys.argv[2]
	except:
		outputFile = '-'
	
	if outputFile == '-':
		saveModel(model, sys.stdout)
	else:
		with open(outputFile, 'w') as ofh:
			saveModel(model, ofh)
