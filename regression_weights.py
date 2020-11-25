import pickle
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import torch
from transformers import RobertaTokenizer, RobertaForCausalLM, GPT2LMHeadModel, GPT2Tokenizer, AlbertTokenizer, AlbertForMaskedLM
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import distance

def sample_sentences(file_name, n):

  with open(file_name) as f:
    head = [next(f).strip() for x in range(n)]

  return head 


def create_sent_to_score_dict(score_name, tokenizer, model, sentences):

	s = pd.read_pickle(score_name)
	d = s['data']

	coeffs = d.layer_weights[0][-1].values

	intercept = d.layer_weights[0][-1].intercept.values

	new_model = LinearRegression()
	new_model.intercept_ = intercept
	new_model.coef_ = coeffs

	sent_dict = {}
	for sent in sentences:

		inputs = tokenizer(sent,return_tensors="pt")
		outputs = model(**inputs, labels=inputs["input_ids"], output_hidden_states=True)

		hiddenStates = outputs.hidden_states 

		hiddenStatesLayer = hiddenStates[-1]

		lastWordState = hiddenStatesLayer[-1, :].detach().numpy()

		lastWordState = lastWordState[-1].reshape(1, -1)

		prediction = new_model.predict(lastWordState)
		
		sent_dict[sent] = prediction

	return sent_dict

sentences = sample_sentences("sentences4lara.txt", 500) 

r_score_name = '/om2/user/gretatu/.result_caching/neural_nlp.score/benchmark=Pereira2018-encoding-weights,model=roberta-base,subsample=None.pkl'
r_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
r_model = RobertaForCausalLM.from_pretrained('roberta-base',  return_dict=True)

g_score_name = '/om2/user/gretatu/.result_caching/neural_nlp.score/benchmark=Pereira2018-encoding-weights,model=gpt2,subsample=None.pkl'
g_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
g_model = GPT2LMHeadModel.from_pretrained('gpt2', return_dict =True)

a_score_name ='/om2/user/gretatu/.result_caching/neural_nlp.score/benchmark=Pereira2018-encoding-weights,model=albert-base-v2,subsample=None.pkl'
a_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
a_model = AlbertForMaskedLM.from_pretrained('albert-base-v2', return_dict=True)


r_score_dict = create_sent_to_score_dict(r_score_name, r_tokenizer, r_model, sentences)

g_score_dict = create_sent_to_score_dict(g_score_name, g_tokenizer, g_model, sentences)


r_scores = [v for (k,v) in sorted(r_score_dict.items(), key=lambda x: x[0], reverse=True)]
g_scores = [v for (k,v) in sorted(g_score_dict.items(), key=lambda x: x[0], reverse=True)]

cosine_distances = [distance.cosine(r_scores[i], g_scores[i]) for i in range(len(g_scores))]
sentence_list = [k for (k,v) in sorted(g_score_dict.items(), key=lambda x: x[0], reverse=True)]
