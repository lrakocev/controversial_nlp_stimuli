import os
import re
import glob
import sys

def check_for_sentence(sent):
	
	file_list = glob.glob('6884_judge*.out')#sampling_logs_12_01_20/sample_judge*.out

	for file in file_list:
	    f = open(file)
	    x = f.read()
	    if " ".join(sent) in x:
	        print(file)


if __name__ == "__main__":

  
  globals()[sys.argv[1]](sys.argv[2:])
