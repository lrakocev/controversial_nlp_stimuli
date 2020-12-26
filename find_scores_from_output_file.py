import os
import re
import glob
import sys

def find(file):
	
	f = open(file, 'r')
	data = f.read()
	for matches in re.findall("/^new score:\d*\.?\d*$/+", data):
		print(matches.group(0))

if __name__ == "__main__":

  globals()[sys.argv[1]](sys.argv[2])