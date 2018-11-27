#!/usr/bin/env python3

import sys
import re

#from learntc import log

def log(msg):
	sys.stderr.write("{0}: {1}\n".format(str(datetime.now()), msg))

class DefUniqDict(dict):
	def __missing__(self, key):
		return key

def loadModel(filename):
	res = DefUniqDict()
	
	with open(filename, 'r') as filehandle:
		for w in filehandle:
			w = w.strip()
			
			res[w.lower()] = w
		
		return res

def isUpper(w):
	return re.search(r'[A-Z]', w) and not re.search(r'[a-z]', w)

def truecase(model, wordlist):
	return [model[w.lower()] if (i == 0 or isUpper(w) or wordlist[i-1] in ".:;") else w for i, w in enumerate(wordlist)]

def processLines(model, fh):
	logFreq = 100000
	i = 0
	for line in fh:
		words = line.strip().split()
		
		print(" ".join(truecase(model, words)))
		
		i += 1
		if not i % logFreq:
			log("processed {0} lines".format(i))
	
	if i % logFreq:
		log("processed {0} lines".format(i))

def processLine(model, line):
    words = line.strip().split()
    return " ".join(truecase(model, words))

if __name__ == '__main__':
	modelfile = sys.argv[1]
	
	model = loadModel(modelfile)
	
	try:
		filename = sys.argv[2]
	except IndexError:
		filename = '-'
	
	if filename == '-':
		processLines(model, sys.stdin)
	else:
		with open(filename, 'r') as fh:
			processLines(model, fh)
