import pickle
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import torch
from transformers import RobertaTokenizer, RobertaForCausalLM
import numpy as np

score_name1 = '/Users/gt/.result_caching/neural_nlp.score/benchmark=Pereira2018-encoding-weights,model=roberta-base,subsample=None.pkl'

s = pd.read_pickle(score_name1)
d = s['data']

print(d.layer_weights[-1])

roberta_coeffs, roberta_intercept = d.layer_weights[-1]

def sample_sentences(file_name, n):

  with open(file_name) as f:
    head = [next(f).strip() for x in range(n)]

  return head 

new_model = LinearRegression()
new_model.intercept_ = roberta_coeffs
new_model.coef_ = roberta_intercept

sentences = sample_sentences("sentences4lara.txt", 100) 

for sent in sentences:

	tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
	model = RobertaForCausalLM.from_pretrained('roberta-base')

	inputs = tokenizer(sent, return_tensors="pt")
	outputs = model(**inputs)

	prediction_logits = outputs.logits[-1]
	new_model.predict(prediction_logits)
