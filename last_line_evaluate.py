import os
import numpy as np
import matplotlib.pyplot as plot
from scipy import stats
import glob
import re

_float_regexp = re.compile(r"^[-+]?(?:\b[0-9]+(?:\.[0-9]*)?|\.[0-9]+\b)(?:[eE][-+]?[0-9]+\b)?$").match
def is_float_re(str):
    return True if _float_regexp(str) else False


file_list = glob.glob('6884_judge*') #6884_evaluate_output/6884*.out

after_score_dict = {}
for file in file_list:
	a_file = open(file, "r")
	lines = a_file.readlines()
	score = lines[-1].strip()
	sentence = lines[-2].strip()
	sentence = " ".join(sentence.split(" ")[1:])
	after_score_dict[sentence] = score


after_vals = [float(v) for (k,v) in after_score_dict.items() if is_float_re(v)]
after_std = np.std(after_vals)

sentences = [k for (k,v) in score_dict.items()]

after_avg = sum(after_vals)/len(after_vals)

print("Average score", after_avg)


file_list_2 = glob.glob('6884_evaluate_output/6884*.out') 

before_score_dict = {}
for file in file_list:
	a_file = open(file, "r")
	lines = a_file.readlines()
	score = lines[-1].strip()
	sentence = lines[-2].strip()
	sentence = " ".join(sentence.split(" ")[1:])
	before_score_dict[sentence] = score

before_vals = [float(v) for (k,v) in before_score_dict.items() if is_float_re(v)]

sentences = [k for (k,v) in score_dict.items()]

before_avg = sum(before_vals)/len(before_vals)
before_std = np.std(before_vals)

print("Average score", before_avg)

positions = ["Before", "After"]
x_pos = np.arange(len(positions))
error = [before_std, after_std]
avgs = [before_avg, after_avg]

fig, ax = plt.subplots()

ttest, pval = stats.ttest_rel(before_vals, after_vals)

ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Average Jensen Shannon Score')
ax.set_xticks(x_pos)
ax.set_xticklabels(materials)
ax.set_title('T-statistic ' + str(ttest) + " P value " + str(pval))
ax.yaxis.grid(True)

plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()




