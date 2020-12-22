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
from scipy.spatial import distance
from itertools import combinations
import pickle
from sklearn.linear_model import LinearRegression

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

def get_pos_dict(filename):

  data = pd.read_csv(filename, sep="\t")

  data = data[['Word', 'Dom_PoS_SUBTLEX']]

  pos_dict = dict(zip(data.Word, data.Dom_PoS_SUBTLEX))

  return pos_dict


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

  print("model name is: ", model_name.model_name, " and it's final probabilties top 20 words are: ", sorted_vals)

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

def sorted_avg_dict_top_k(distrs, k):

  #have list of dicts, words: probabilities
  #first want to average them

  sorted_distrs = []
  for d in distrs:
    sorted_distrs.append([v for (k,v) in sorted(d.items(), key = lambda x: x[0])])
    vocab = [k for (k,v) in sorted(d.items(), key = lambda x: x[0])]

  df_probabilities = pd.DataFrame(sorted_distrs)

  df_probabilities_mean = df_probabilities.mean()

  final_avg_distr = dict(zip(vocab, df_probabilities_mean))

  sorted_dict_top_k = {key: final_avg_distr [key] for key in sorted(final_avg_distr , key=final_avg_distr .get, reverse=True)[:k]}

  return sorted_dict_top_k

def evaluate_sentence_jsd(model_list, sentence, vocab, n, js_dict):

  sentence_split = sentence.split(" ")
  len_sentence = len(sentence_split)

  curr_context = ""
  total_js = 0
  js_positions = []
  distrs = {}
  plotting_purposes = {}
  for model_name in model_list:
    plotting_purposes[model_name.model_name] = []

  for i in range(0, len_sentence):
    curr_context += sentence_split[i] + " "

    if curr_context in js_dict.keys():
      curr_js = js_dict[curr_context]

    else:
      for model_name in model_list:
        next_word_distr = get_distribution(model_name, curr_context, vocab, n)
        distrs[model_name] = list(next_word_distr.values())

        plotting_purposes[model_name.model_name].append(next_word_distr)
    
      curr_js = jsd(list(distrs.values()))
      js_dict[curr_context] = curr_js

    total_js += curr_js
    js_positions.append(curr_js)
    
  # plotting purposes
  top_avg_distr = {name: sorted_avg_dict_top_k(distrs, 5) for (name, distrs) in plotting_purposes.items()}

  # now overlap these

  for D in top_avg_distr.values():
    plt.bar(*zip(*D.items()), alpha=.1)

  name = sentence + " controversy graph.png"
  plt.savefig(name)
  plt.close()

  print(total_js/len_sentence)
  return total_js/len_sentence

filename = "SUBTLEXus74286wordstextversion.txt"
vocab = get_vocab(filename, 3000)

filename2 = "SUBTLEX-US frequency list with PoS information text version.txt"
pos_dict = get_pos_dict(filename2)

GPT2 = ModelInfo(GPT2LMHeadModel.from_pretrained('gpt2', return_dict =True), GPT2Tokenizer.from_pretrained('gpt2'), "Ġ", vocab, "GTP2")

Roberta = ModelInfo(RobertaForCausalLM.from_pretrained('roberta-base',  return_dict=True), RobertaTokenizer.from_pretrained('roberta-base'), "_", vocab, "Roberta")

XLM = ModelInfo(XLMWithLMHeadModel.from_pretrained('xlm-mlm-xnli15-1024', return_dict=True), XLMTokenizer.from_pretrained('xlm-mlm-xnli15-1024'), "_", vocab, "XLM")

T5 = ModelInfo(T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True), T5Tokenizer.from_pretrained("t5-base"), "_", vocab, "T5")

Albert = ModelInfo(AlbertForMaskedLM.from_pretrained('albert-base-v2', return_dict=True), AlbertTokenizer.from_pretrained('albert-base-v2'), "_", vocab, "Albert")

TXL = ModelInfo(TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103'),TransfoXLTokenizer.from_pretrained('transfo-xl-wt103'), "_", vocab, "TXL")

model_list = [GPT2, Roberta, XLM, T5] 
sentence = "word word word word word word word word word word"
evaluate_sentence_jsd(model_list, sentence, vocab, 100, {})

