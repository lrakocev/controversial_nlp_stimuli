import pickle
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import torch
from transformers import RobertaTokenizer, RobertaForCausalLM
import numpy as np
import pandas as pd
import xarray as xr

score_name1 = '/om2/user/gretatu/.result_caching/neural_nlp.score/benchmark=Pereira2018-encoding-weights,model=roberta-base,subsample=None.pkl'

s = pd.read_pickle(score_name1)
d = s['data']

roberta_coeffs = d.layer_weights[0][-1].values

roberta_intercept = d.layer_weights[0][-1].intercept.values

def sample_sentences(file_name, n):

  with open(file_name) as f:
    head = [next(f).strip() for x in range(n)]

  return head 

new_model = LinearRegression()
new_model.intercept_ = roberta_intercept
new_model.coef_ = roberta_coeffs

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForCausalLM.from_pretrained('roberta-base',  return_dict=True)


sentences = sample_sentences("sentences4lara.txt", 100) 

sent_dict = {}
for sent in sentences:


	inputs = tokenizer(sent,return_tensors="pt")
	outputs = model(**inputs, labels=inputs["input_ids"], output_hidden_states=True)

	hiddenStates = outputs.hidden_states 

	hiddenStatesLayer = hiddenStates[-1]

	lastWordState = hiddenStatesLayer[-1, :].detach().numpy()

	lastWordState = lastWordState[-1].reshape(1, -1)

	prediction = new_model.predict(lastWordState)
	print(prediction)
	sent_dict[sent] = prediction

print(sent_dict)


