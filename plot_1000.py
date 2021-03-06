import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def is_valid_decimal(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True

file_list = glob.glob('final_jsd_1000_*.out')
scores_list = []

for file in file_list:
	a_file = open(file, "r")
	lines = a_file.readlines()
	score = lines[-1].strip()

	if is_valid_decimal(score):
		
		scores = lines[-2].split(":")[2].strip()[1:-1].split(",")

		scores_list.append([float(x) for x in scores])


max_length = len(max(scores_list, key=len))

scores_list = [l + [np.nan]*(max_length - len(l))  if len(l) < max_length else l for l in scores_list]


avg_scores = np.array(np.nanmean(scores_list, axis = 0))
std_dev_scores = np.array(np.nanstd(scores_list, axis = 0))

avg_plus_std = [a + b for a, b in zip(avg_scores, std_dev_scores)]
avg_minus_std = [a - b for a, b in zip(avg_scores, std_dev_scores)]
x_marks = range(0, len(avg_scores))

plt.plot(x_marks, avg_scores, 'b', label = "average convergence line")

plt.errorbar(x_marks[0::50], avg_scores[0::50], std_dev_scores[0::50])

#plt.plot(x_marks, avg_plus_std, 'r', label = "average + std dev convergence line")
#plt.plot(x_marks, avg_minus_std, 'r',label = "average - std dev convergence line")

plt.legend()
plt.xlabel("iterations")
plt.ylabel("j-s divergence scores")
plt.title("Average Convergence Line with Std Deviations Shown")
plt.savefig("Convergence Graph")
plt.show()
plt.close()