import regression_weights as rw
import matplotlib.pyplot as plt
import last_line_evaluate as lle
from scipy.stats import pearsonr

ordered_cosine_distances = rw.cosine_distances
sentence_list = rw.sentence_list
cosine_dict = dict(zip(sentence_list, ordered_cosine_distances))

print(cosine_dict)

jsd_score_dict = lle.score_dict

# removing the error lines
jsd_score_dict = {k:v for (k,v) in jsd_score_dict if k in sentence_list}

# getting rid of extra sentences
cosine_dict = {k:v for (k,v) in cosine_dict if k in jsd_score_list.keys()}

#getting vals in order of sorted keys

cosine_scores = [v for (k,v) in sorted(cosine_dict.items(), key=lambda x: x[0], reverse=True)]
jsd_scores = [v for (k,v) in sorted(jsd_score_dict.items(), key=lambda x: x[0], reverse=True)]

corr, _ = pearsonr(cosine_scores, jsd_scores)

print("corr", corr)

plt.scatter(cosine_scores, jsd_scores)
plt.show()
plt.savefig("JSD v Cosine")
plt.close()

