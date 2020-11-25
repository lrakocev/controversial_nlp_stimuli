import os

import glob
file_list = glob.glob('albert_vs_gpt2/*.out')

score_dict = {}
for file in file_list:
	a_file = open(file, "r")
	lines = a_file.readlines()
	score = lines[-1].strip()
	sentence = lines[-2].strip()
	sentence = " ".join(sentence.split(" ")[1:])
	score_dict[sentence] = score
