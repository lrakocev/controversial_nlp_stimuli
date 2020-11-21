import os

import glob
file_list = glob.glob('evaluate*.out')

score_dict = []
for file in file_list:
	a_file = open(file, "r")
	lines = a_file.readlines()
	score_dict[lines[-2]] = lines[-1]

print(score_dict)