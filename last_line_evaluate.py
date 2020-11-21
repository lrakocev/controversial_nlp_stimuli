import os

import glob
file_list = glob.glob('evaluate*.out')

score_dict = {}
for file in file_list:
	a_file = open(file, "r")
	lines = a_file.readlines()
	score = lines[-1]
	sentence = lines[-2]
	score_dict[sentence] = score

print(score_dict)