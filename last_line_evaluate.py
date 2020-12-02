import os

import glob
file_list = glob.glob('6884_evaluate_output/6884*.out')

score_dict = {}
for file in file_list:
	a_file = open(file, "r")
	lines = a_file.readlines()
	score = lines[-1].strip()
	sentence = lines[-2].strip()
	sentence = " ".join(sentence.split(" ")[1:])
	score_dict[sentence] = score

vals = [int(v) for (k,v) in score_dict.items()]

avg = sum(vals)/len(vals)

print("Average score", avg)