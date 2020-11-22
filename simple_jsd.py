from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer, RobertaTokenizer, RobertaForCausalLM
from scipy.spatial import distance
from scipy.special import softmax
import numpy as np
import torch.nn as nn
import math

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

	sentence_split = sentence.split(" ")
	word_to_ids = {}
	for i in range(len(sentence_split)):
		word = sentence_split[i]
		word_tokens = tokenizer.tokenize(word)
		word_ids = tokenizer.convert_tokens_to_ids(word_tokens)
		word_to_ids[word] = word_ids


	inputs = tokenizer(sentence, return_tensors='pt')
	outputs = model(**inputs)

	m = nn.LogSoftmax()

	predictions_total = outputs.logits

	scores = []
	for i in range(len(ids)):
		word = sentence_split[i]
		word_ids = word_to_ids[word]
		score = 0
		for ind in word_ids:
			predictions = m(predictions_total[0][i])
			score += float(predictions[ind])
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


sentences = sorted(sample_sentences("sentences4lara.txt", 10)) #4507

nlp_roberta = pipeline("fill-mask", model="roberta-base")
nlp_xlm = pipeline("fill-mask", model="xlm-mlm-xnli15-1024")
GPT2_model = GPT2LMHeadModel.from_pretrained('gpt2', return_dict =True)
GPT2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
Roberta_model = RobertaForCausalLM.from_pretrained('roberta-base',  return_dict=True)
Roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

final_jsd_scores = {}

for i in range(len(sentences)):
	sentence = sentences[i]
	#scores1 = get_probabilities(nlp_roberta, sentence)
	#scores2 = get_probabilities(nlp_xlm, sentence)
	scores1 = get_probabilities_alternative(Roberta_model, Roberta_tokenizer, sentence)
	scores2 = get_probabilities_alternative(GPT2_model, GPT2_tokenizer, sentence)

	#jsd = evaluate_sentence(scores1, scores2)
	#final_jsd_scores[sentence] = jsd

#print(final_jsd_scores)
