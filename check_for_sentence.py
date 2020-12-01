import os
import re
import glob

def check_for_sentence(sent):
	
	file_list = glob.glob('sample_judge*.out')

	for file in file_list:
	    f = open(file)
	    x = f.read()
	    if sent in x:
	        print(file)
	        

if __name__ == "__main__":

  
  globals()[sys.argv[1]](sys.argv[2])
