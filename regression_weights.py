import pickle
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import torch
from transformers import RobertaTokenizer, RobertaForCausalLM
import numpy as np
import pandas as pd

score_name1 = '/om2/user/gretatu/.result_caching/neural_nlp.score/benchmark=Pereira2018-encoding-weights,model=roberta-base,subsample=None.pkl'

s = pd.read_pickle(score_name1)
d = s['data']

print(d.layer_weights)

roberta_coeffs = d.layer_weights[0][0][-1].divider

roberta_intercept = d.layer_weights[0][0][-1].intercept

def sample_sentences(file_name, n):

  with open(file_name) as f:
    head = [next(f).strip() for x in range(n)]

  return head 

new_model = LinearRegression()
new_model.intercept_ = roberta_coeffs
new_model.coef_ = roberta_intercept

sentences = sample_sentences("sentences4lara.txt", 100) 

for sent in sentences:

	print(sent)

	tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
	model = RobertaForCausalLM.from_pretrained('roberta-base')

	inputs = torch.tensor([tokenizer.encode(sent)])

	outputs = model(inputs, output_hidden_states=True)
	hiddenStates = outputs[-1]  # number of layers + emb layer
	# print('Number of layers + embedding layer: ', np.shape(hiddenStates))
	hiddenStatesLayer = hiddenStates[-1]  # (batch_size, sequence_length, hidden_size)
	batchSize = np.shape(hiddenStatesLayer)[0]
	# print('Batch size: ', batchSize)
	# hiddenStatesLayer2 = hiddenStates[-1] # fetches last layer
	# np.shape(hiddenStatesLayer)
	lastWordState = hiddenStatesLayer[-1, :].detach().numpy()

	new_model.predict(lastWordState)
