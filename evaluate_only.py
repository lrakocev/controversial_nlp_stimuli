import numpy as np
from scipy.stats import entropy as H
import tensorflow as tf
import torch
from transformers import  GPT2LMHeadModel, GPT2Tokenizer, TransfoXLLMHeadModel, TransfoXLTokenizer, T5Tokenizer, T5ForConditionalGeneration, T5Config, RobertaTokenizer, RobertaForCausalLM, RobertaConfig, AlbertTokenizer, AlbertForMaskedLM,XLMTokenizer, XLMWithLMHeadModel, BertTokenizer, BertForMaskedLM
import sys
from scipy.special import softmax
import torch
import random
import string
import copy
import random
import csv
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import math
from functools import reduce

class ModelInfo():

  def __init__(self, model, tokenizer, start_token_symbol, vocab, model_name):
    self.model = model.to("cuda")
    self.tokenizer = tokenizer
    self.start_token_symbol = start_token_symbol
    self.model_name = model_name
    self.word_token_dict = {word: self.tokenizer.tokenize(" " + str(word)) for word in vocab}

    all_tokens = list(self.word_token_dict.values())

    all_tokens = reduce(lambda x,y: x+y,all_tokens)

    self.id_token_dict = {token: self.tokenizer.convert_tokens_to_ids(token) for token in all_tokens}

    self.distr_dict_for_context = {}

def get_vocab(filename, length):

  data = pd.read_csv(filename, sep="\t")

  contractions = ['m', 't', 's', 're', 'd', 'll', 've']

  data = data[~data['Word'].isin(contractions)]

  vocab = data['Word'].head(length)

  vocab_list = vocab.values.tolist()

  return vocab_list

'''
def get_distribution(model_name, context, joint_vocab):
  tokenizer, model = model_name.tokenizer, model_name.model.to("cuda")
  inputs = tokenizer(context,return_tensors='pt').to("cuda")
  if model_name.model_name == "Albert":
    tokens = tokenizer.tokenize(context)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(ids).to('cuda') 

    x=1
    attention_mask= [1 for i in range(len(tokens)-x)] + [0 for i in range(x)]
    attention_mask = torch.tensor(attention_mask).to('cuda')

    outputs = model(input_ids, attention_mask=attention_mask)
  else:
    outputs = model(**inputs, labels=inputs["input_ids"])
  ids = range(0,tokenizer.vocab_size)
  vocab = tokenizer.convert_ids_to_tokens(ids)
  final_vocab = set(vocab) & joint_vocab if len(joint_vocab) != 0 else set(vocab)
  id_list = tokenizer.convert_tokens_to_ids(sorted(final_vocab))
  outputs_array = np.asarray(outputs.logits.cpu().detach()).flatten()

  final_outputs = [outputs_array[i] for i in id_list] 
  probabilities = softmax(final_outputs)
  distr_dict = dict(zip(final_vocab, probabilities))
  distr_dict = {k: v for k, v in sorted(distr_dict.items(), key=lambda item: item[1])}
  
  return distr_dict
'''

def get_distribution(model_name, context, vocab, n):

  print("context", context)

  tokenizer = model_name.tokenizer 
  model = model_name.model.to('cuda')
  model_word_token_dict = model_name.word_token_dict
  model_token_id_dict = model_name.id_token_dict
  tokenizer.pad_token = model_name.start_token_symbol

  if context in model_name.distr_dict_for_context.keys():
    print("seen it!")
    return model_name.distr_dict_for_context[context]

  context_tokens = tokenizer.tokenize(context)

  final_probabilities = {}

  vocab_splits = [vocab[i:i + n] for i in range(0, len(vocab), n)]

  for words in vocab_splits:

    batch_list = [context + str(word) for word in words]
    
    sub_word_token_groupings = [model_word_token_dict[word] for word in words]

    lengths_contexts = [len(sub_word_tokens) for sub_word_tokens in sub_word_token_groupings]

    max_length = max(lengths_contexts)

    id_nums = [[model_token_id_dict[token] for token in sub_word_tokens] for sub_word_tokens in sub_word_token_groupings ]

    for i in range(len(batch_list)):
      length = lengths_contexts[i]
      batch = batch_list[i]
      if length < max_length: 
        added_string = " ".join([model_name.start_token_symbol] * (max_length - length))
        batch = batch + " " + added_string

    inputs = tokenizer(batch_list, padding='longest', return_tensors="pt").to('cuda')

    if model_name.model_name == "Albert":
      attention_mask = []
      input_ids = []

      for i in range(len(batch_list)):
        length = lengths_contexts[i]
        tokens = tokenizer.tokenize(batch_list[i])
        if length < max_length:
          tokens += [tokenizer.eos_token]*(max_length-length)

        ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids.append(ids)
        x=1
        attention_mask.append([1 for i in range(len(tokens)-x)] + [0 for i in range(x)])

      attention_mask = torch.tensor(attention_mask).to('cuda')
      input_ids = torch.tensor(input_ids).to('cuda') 

      outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
    else:
      outputs = model(**inputs, labels=inputs["input_ids"])

    vectorize_log = np.vectorize(math.log)

    log_probabilities = [[vectorize_log(softmax(np.asarray(outputs.logits[j][i].cpu().detach()).flatten())) for i in range(max_length-1,max_length - lengths_contexts[j],-1)] for j in range(len(batch_list))]

    log_probabilities_per_tokens = [[log_probabilities[j][i][id_nums[j][i]] for i in range(len(id_nums[j])-1)] for j in range(len(batch_list))]

    probabilities = [sum(log_probabilities_per_tokens[i]) for i in range(len(log_probabilities_per_tokens))]

    final_probabilities.update({words[i]: probabilities[i] for i in range(len(words))})
  
  #normalizing
  final_probabilities_total = sum(final_probabilities.values())
  final_probabilities = {k: v / final_probabilities_total for k, v in final_probabilities.items()}

  model_name.distr_dict_for_context[context] = final_probabilities

  sorted_vals = [(k,v) for (k,v) in sorted(final_probabilities.items(), key = lambda x: x[1], reverse=True)][:20]

  final_probabilities_sorted = {k: v for k, v in sorted(final_probabilities.items(), key=lambda item: item[0])}
  

  return final_probabilities

def jsd(prob_distributions,logbase=math.e):

    n = len(prob_distributions)
    weights = np.empty(n)
    weights.fill(1/n)
    k = zip(weights, np.asarray(prob_distributions))
    wprobs = np.asarray([x*y for x,y in list(k)])
    mixture = wprobs.sum(axis=0)
    entropy_of_mixture = H(mixture, base=logbase)

    # right term: sum of entropies
    entropies = np.array([H(P_i, base=logbase) for P_i in prob_distributions])
    wentropies = weights * entropies
    sum_of_entropies = wentropies.sum()

    divergence = entropy_of_mixture - sum_of_entropies
    return(divergence)

def evaluate_sentence(model_list, sentence, vocab, n, js_dict):

  sentence_split = sentence.split(" ")
  len_sentence = len(sentence_split)

  curr_context = ""
  total_js = 0
  js_positions = []
  distrs = {}
  
  for i in range(0, len_sentence):
    curr_context += sentence_split[i] + " "

    if curr_context in js_dict.keys():
      curr_js = js_dict[curr_context]

    else:
      for model_name in model_list:
        next_word_distr = get_distribution(model_name, curr_context, vocab, n)
        distrs[model_name] = list(next_word_distr.values())

        top_5_distr = {key: next_word_distr[key] for key in sorted(next_word_distr, key=next_word_distr.get, reverse=True)[:5]}

        model_to_top_5[model_name] = top_5_distr
      curr_js = jsd(list(distrs.values()))
      js_dict[curr_context] = curr_js

    total_js += curr_js
    js_positions.append(curr_js)

  # now plot max_js_distr

  print(total_js/len_sentence)

  return total_js/len_sentence

def sample_sentences(file_name):

  file = open(file_name)
  reader = csv.reader(file)
  num_lines = len(list(reader))
  N = random.randint(0,num_lines-1)
  with open(file_name, 'r') as file:
      reader = csv.reader(file)
      line = next((x for i, x in enumerate(reader) if i == N), None)
      line = (" ".join(line)).translate(str.maketrans('', '', string.punctuation))

  return line

filename = "SUBTLEXus74286wordstextversion.txt"
vocab = get_vocab(filename, 3000)

GPT2 = ModelInfo(GPT2LMHeadModel.from_pretrained('gpt2', return_dict =True), GPT2Tokenizer.from_pretrained('gpt2'), "Ġ", vocab, "GTP2")

Roberta = ModelInfo(RobertaForCausalLM.from_pretrained('roberta-base',  return_dict=True), RobertaTokenizer.from_pretrained('roberta-base'), "_", vocab, "Roberta")


model_list = [GPT2, Roberta]
n = 100


if __name__ == "__main__":

  sentences = [sample_sentences("sentences4lara.txt") for i in range(2000)]

  sent_dict = dict(zip([str(x) for x in range(1,2000)], sentences))

  sentence = sent_dict[sys.argv[2]]

  globals()[sys.argv[1]](model_list, sentence, {})

