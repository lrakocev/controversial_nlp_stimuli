from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from scipy.spatial import distance
from scipy.special import softmax
import numpy as np
import torch.nn as nn

def get_probabilities(nlp, sentence):

	sentence = sentence.split(" ")
	sentences = []
	for i in range(len(sentence)):
		cur_sentence = sentence.copy()
		cur_sentence[i] = nlp.tokenizer.mask_token
		cur_sentence = " ".join(cur_sentence)
		sentences.append(cur_sentence)
	
	scores = []
	for i in range(len(sentences)):
		sent = sentences[i]
		target = " " + sentence[i]
		score = nlp(sent, targets=[target])[0]['score']
		scores.append([score, 1-score])

	return scores

def get_probabilities_alternative(model, tokenizer, sentence):

	tokens = tokenizer.tokenize(sentence)
	ids = tokenizer.convert_tokens_to_ids(tokens)

	inputs = tokenizer(sentence, return_tensors='pt')
	outputs = model(**inputs)

	m = nn.LogSoftmax()

	predictions = m(outputs.logits, dim=2)

	scores = []
	for i in range(len(ids)):
		ind = ids[i]
		score = predictions[0][i][ind]
		print(score)
		scores.append([score, 1-score])

	return scores

def evaluate_sentence(scores1, scores2):

	js_scores = []
	for i in range(len(scores1)):
		js = distance.jensenshannon(scores1[i], scores2[i])
		js_scores.append(js)

	return sum(js_scores)/len(js_scores)


def sample_sentences(file_name, n):

  with open(file_name) as f:
    head = [next(f).strip() for x in range(n)]

  return head


sentences = sorted(sample_sentences("sentences4lara.txt", 100))

nlp_roberta = pipeline("fill-mask", model="roberta-base")
nlp_xlm = pipeline("fill-mask", model="xlm-mlm-xnli15-1024")
GPT2_model = GPT2LMHeadModel.from_pretrained('gpt2', return_dict =True)
GPT2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


final_jsd_scores = {}

for i in range(len(sentences)):
	sentence = sentences[i]
	scores1 = get_probabilities(nlp_roberta, sentence)
	#scores2 = get_probabilities(nlp_xlm, sentence)
	scores2 = get_probabilities_alternative(GPT2_model, GPT2_tokenizer, sentence)

	jsd = evaluate_sentence(scores1, scores2)
	final_jsd_scores[sentence] = jsd

print(final_jsd_scores)
