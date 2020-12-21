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

def get_distribution(model_name, context, joint_vocab):
  tokenizer, model = model_name.tokenizer, model_name.model
  inputs = tokenizer(context,return_tensors='pt')
  if model_name.model_name == "Albert":
    tokens = tokenizer.tokenize(context)
    x=1
    attention_mask.append([1 for i in range(len(tokens)-x)] + [0 for i in range(x)])
    outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
  else:
    outputs = model(**inputs, labels=inputs["input_ids"])
  ids = range(0,tokenizer.vocab_size)
  vocab = tokenizer.convert_ids_to_tokens(ids)
  final_vocab = set(vocab) & joint_vocab if len(joint_vocab) != 0 else set(vocab)
  id_list = tokenizer.convert_tokens_to_ids(sorted(final_vocab))
  outputs_array = np.asarray(outputs[0]).flatten()
  final_outputs = [outputs_array[i] for i in id_list] 
  probabilities = softmax(final_outputs)
  distr_dict = dict(zip(final_vocab, probabilities))
  return distr_dict


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


def evaluate_sentence(model_list, sentence, joint_vocab):
  sentence_split = sentence.split(" ")
  len_sentence = len(sentence_split)
  curr_context = ""
  total_js = 0
  js_positions = []
  distrs = {}
  for i in range(0, len_sentence):
    curr_context += sentence_split[i] + " "
    
    for model_name in model_list:
      next_word_distr = get_distribution(model_name, curr_context, joint_vocab)
      distrs[model_name] = next_word_distr
    curr_js = jsd(distrs.values())
    total_js += curr_js
    
    js_positions.append(curr_js)
    
  return total_js/len_sentence, js_positions

def sample_sentences(file_name, n):

  with open(file_name) as f:
    head = [next(f).strip() for x in range(n)]

  return head 

filename = "SUBTLEXus74286wordstextversion.txt"
vocab = get_vocab(filename, 3000)

GPT2 = ModelInfo(GPT2LMHeadModel.from_pretrained('gpt2', return_dict =True), GPT2Tokenizer.from_pretrained('gpt2'), "Ġ", vocab, "GTP2")

Roberta = ModelInfo(RobertaForCausalLM.from_pretrained('roberta-base',  return_dict=True), RobertaTokenizer.from_pretrained('roberta-base'), "_", vocab, "Roberta")

Albert = ModelInfo(AlbertForMaskedLM.from_pretrained('albert-base-v2', return_dict=True), AlbertTokenizer.from_pretrained('albert-base-v2'), "_", vocab, "Albert")

XLM = ModelInfo(XLMWithLMHeadModel.from_pretrained('xlm-mlm-xnli15-1024', return_dict=True), XLMTokenizer.from_pretrained('xlm-mlm-xnli15-1024'), "_", vocab, "XLM")

T5 = ModelInfo(T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True), T5Tokenizer.from_pretrained("t5-base"), "_", vocab, "T5")

TXL = ModelInfo(TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103'),TransfoXLTokenizer.from_pretrained('transfo-xl-wt103'), "_", vocab, "TXL")

model_list = [GPT2, Albert, Roberta, XLM, T5] 
n = 100


if __name__ == "__main__":

  #entences = sorted(sample_sentences("sentences4lara.txt", 1000))

  #sent_dict = dict(zip([str(x) for x in range(1,1000)], sentences))

  sentence = "word word word word word word word word word word" #sent_dict[sys.argv[2]]

  globals()[sys.argv[1]](model_list, sentence, {})

