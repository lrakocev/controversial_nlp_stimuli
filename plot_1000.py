import os
import glob
import numpy as np

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
		print("here")
		scores = lines[-2].strip().split(":")[2]
		scores_list.append(scores)


scores_list = np.array(scores_list).astype(np.float)
avg_scores = list(map(np.mean, scores_list))
std_dev_scores = list(map(np.std, scores_list))

avg_plus_std = [avg_scores[i] + std_dev_scores[i] for i in range(avg_scores)]
avg_minus_std = [avg_scores[i] - std_dev_scores[i] for i in range(avg_scores)]

x_marks = range(0, len(avg_scores))

plt.plot(x_marks, avg_scores, 'b', label = "average convergence line")

plt.plot(x_marks, avg_plus_std, 'r', label = "average + std dev convergence line")
plt.plot(x_marks, avg_minus_std, 'r',label = "average - std dev convergence line")

plt.legend()
plt.xlabel("iterations")
plt.ylabel("j-s divergence scores")
plt.title("Average Convergence Line with Std Deviations Shown")
plt.savefig("Convergence Graph")
plt.show()
plt.close()