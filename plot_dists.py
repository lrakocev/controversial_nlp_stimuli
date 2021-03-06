import matplotlib.pyplot as plt
import last_line_evaluate as lle
from scipy.stats import pearsonr
import regression_weights as rw
import scipy
from sklearn.metrics import r2_score
'''
import simple_jsd
import jsd_output

simple_jsd_scores = simple_jsd.final_jsd_scores
simple_jsd_scores = jsd_output.js
'''

ordered_cosine_distances = rw.cosine_distances
sentence_list = rw.sentence_list
cosine_dict = dict(zip(sentence_list, ordered_cosine_distances))

jsd_score_dict = lle.score_dict

# removing the error lines
jsd_score_dict = {k:v for (k,v) in jsd_score_dict.items() if k in sentence_list}

# getting rid of extra sentences
cosine_dict = {k:v for (k,v) in cosine_dict.items() if k in jsd_score_dict.keys()}

#getting vals in order of sorted keys
cosine_scores = [float(v) for (k,v) in sorted(cosine_dict.items(), key=lambda x: x[0], reverse=True)]
#jsd_scores = [float(v) for (k,v) in sorted(simple_jsd_scores.items(), key=lambda x: x[0], reverse=True)]
jsd_scores = [float(v) for (k,v) in sorted(jsd_score_dict.items(), key=lambda x: x[0], reverse=True)]

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(cosine_scores, jsd_scores)

abline_values = [slope * i + intercept for i in cosine_scores]

print("R2: ", r_value**2, " and p-val: ", p_value)

plt.plot(cosine_scores, abline_values, color='black', linewidth=3)
plt.scatter(cosine_scores, jsd_scores)
plt.xlabel("cosine distances")
plt.ylabel("j-s divergence scores")
plt.title("R2: " + str(r_value**2) + " and p-value: " + str(p_value))
plt.savefig("GPT2 vs Roberta JSD Scores")
plt.show()
plt.close()


