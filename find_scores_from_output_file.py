import os
import re
import glob
import sys

def find(file):
	
	f = open(file, 'r')
	regex_unique = re.compile("/^new score: \d*\.?\d* curr_score: \d*\.?\d*$") 
	for line in f:
		match = re.search(regex_unique,line)
		if match:
			print(match.group(0))

if __name__ == "__main__":

  globals()[sys.argv[1]](sys.argv[2])