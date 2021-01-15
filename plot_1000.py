import os
import glob
import numpy as np

file_list = glob.glob('final_jsd_1000_*.out')
scores_list = []

for file in file_list:
	a_file = open(file, "r")
	lines = a_file.readlines()
	score = lines[-1].strip()
	print(file, score)
	if score.isnumeric():
		scores = lines[-2].strip().split(":")[2]
		scores_list.append(scores)


avg_scores = list(map(np.mean, scores_list))
std_dev_scores = list(map(np.std, scores_list))

print(avg_scores)


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
plt.savefig("JSD 1000 Convergence Graph")
plt.show()
plt.close()