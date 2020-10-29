import numpy as np
from scipy.stats import entropy as H
import tensorflow as tf
import torch
from transformers import  GPT2LMHeadModel, GPT2Tokenizer, TransfoXLLMHeadModel, TransfoXLTokenizer, T5Tokenizer, T5ForConditionalGeneration, T5Config, RobertaTokenizer, RobertaForCausalLM, RobertaConfig, AlbertTokenizer, AlbertForMaskedLM,XLMTokenizer, XLMWithLMHeadModel
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


class ModelInfo():

  def __init__(self, model, tokenizer, start_token_symbol, vocab):
    self.model = model
    self.tokenizer = tokenizer
    self.start_token_symbol = start_token_symbol
    self.word_token_dict = {word: self.tokenizer.tokenize(" " + str(word)) for word in vocab}

  def create_word_to_token_dict(self, vocab):

    for word in vocab:
      word = " " + str(word)
      
      self.word_token_dict[word] = self.tokenizer.tokenize(word)


def get_vocab(filename):

  data = pd.read_csv(filename, sep="\t")

  contractions = ['m', 't', 's', 're', 'd', 'll', 've']

  data = data[~data['Word'].isin(contractions)]

  vocab = data['Word'].head(50000)

  vocab_list = vocab.values.tolist()

  return vocab_list


def get_distribution(model_name, context, next_word, vocab):

  tokenizer = model_name.tokenizer 
  model = model_name.model
  model_word_token_dict = model_name.word_token_dict

  inputs = tokenizer(context, return_tensors="pt")

  outputs = model(**inputs, labels=inputs["input_ids"])

  next_word_tokens = model_word_token_dict[str(next_word)]
  N = len(next_word_tokens)

  vectorize_log = np.vectorize(math.log)

  if N == 1: 
    probabilities = softmax(np.asarray(outputs.logits.detach()).flatten())
  else: 
    logits_size = list(outputs.logits.size())[1]

    log_probabilities = [vectorize_log(softmax(np.asarray(outputs.logits[0][i].detach()).flatten())) for i in range(logits_size)]

    
    log_probabilities = [l.tolist() for l in log_probabilities]

    probabilities = np.sum(log_probabilities, axis=0)

  distr_dict = dict(zip(vocab, probabilities))
  return probabilities, distr_dict


def jsd(prob_distributions, weights, logbase=math.e):
    # left term: entropy of misture
    
    types = [type(i) for i in prob_distributions]

    lens = [len(i) for i in prob_distributions]
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


def evaluate_sentence(model_list, sentence, vocab):

  sentence_split = sentence.split(" ")
  len_sentence = len(sentence_split)

  curr_context = ""
  total_js = 0
  js_positions = []
  distrs = {}

  for i in range(0, len_sentence-1):
    curr_context += sentence_split[i] + " "
    
    for model_name in model_list:
      tokenizer = model_name.tokenizer
      model = model_name.model
      probabilities, next_word_distr = get_distribution(model_name, curr_context, sentence_split[i+1], vocab)
      distrs[model_name] = list(next_word_distr.values())

    n = len(model_list)
    weights = np.empty(n)
    weights.fill(1/n)

    print("curr context", curr_context)

    total_js += jsd(list(distrs.values()), weights)
    curr_js = total_js/(i+1)
    js_positions.append(curr_js)
    
  return total_js/len_sentence, js_positions


def get_avg_distr(model_list, context, next_word, vocab):

    distrs = {}
    for model_name in model_list:
      tokenizer = model_name.tokenizer
      model = model_name.model

      probabilities, next_word_distr = get_distribution(model_name, context, next_word, vocab)
      distrs[model_name] = list(next_word_distr.values())

    df = pd.DataFrame(distrs.values())

    avg_distr = dict(df.mean())

    avg_distr_sorted_vals = [v for (k,v) in sorted(avg_distr.items(), key=lambda x: x[1], reverse=True)]

    avg_distr_vals = np.cumsum(np.array(avg_distr_sorted_vals))

    avg_distr_summed = dict(zip(vocab, avg_distr_vals))

    prob_list_sum = sum(avg_distr_vals)
    prob_list = [v/prob_list_sum for k, v in avg_distr_summed.items()]
    word_list = [k for k, v in avg_distr_summed.items()]



    return prob_list, word_list

def discounting(cur_ind, js_positions, gamma=0.9):

  total = 0
  for i in range(len(js_positions)-cur_ind):
    total += js_positions[cur_ind+i]*(gamma**i)

  return total/(len(js_positions)-cur_ind+1)


def change_sentence(model_list, sentence, vocab):

  original_score, original_js_positions = evaluate_sentence(model_list, sentence, vocab)
  print("Old sentence is: ", sentence, " with JS: ", original_score, " and positional JS scores: ", original_js_positions)
  scores = [original_score]
  js_positions = [original_js_positions]
  changes = []
  change = ""
  sentence_split = sentence.split(" ")
  len_sentence = len(sentence_split)
  final_modified_sentence = copy.deepcopy(sentence_split)

  curr_sentence_score, cur_js_positions = evaluate_sentence(model_list, ' '.join(sentence_split), vocab)

  modified_sentence_replacements = copy.deepcopy(sentence_split)
  modified_sentence_deletions = copy.deepcopy(sentence_split)
  modified_sentence_additions = copy.deepcopy(sentence_split)

  # deciding which position to change at 
  #exponentiated_scores = softmax(cur_js_positions)
  #n = list(np.random.multinomial(1,exponentiated_scores))
  #change_i = n.index(1)

  #print("change index", change_i)
  for change_i in range(len(cur_js_positions)):

    js_dict = {}

    # replacements 
    for j in range(0,10):
      cur_context = sentence_split[:change_i+1]

      cur_prob_list, cur_word_list = get_avg_distr(model_list, ' '.join(cur_context), sentence_split[change_i+1], vocab)

      n = list(np.random.multinomial(1,cur_prob_list))
      ind = n.index(1)
      new_word = cur_word_list[ind]
      modified_sentence_replacements[change_i] = str(new_word)

      print(modified_sentence_replacements)

      new_context = ' '.join(modified_sentence_replacements)
      js_dict[(new_word,"R")] = evaluate_sentence(model_list, new_context, vocab)
    

    #deletions
    modified_sentence_deletions.pop(change_i)
    if len(modified_sentence_deletions) > 0:
      js_dict[("", "D")] = evaluate_sentence(model_list, ' '.join(modified_sentence_deletions), vocab)
    else: 
      js_dict[("", "D")] = (0,[0])


    # additions
    for k in range(0,10):
      cur_context = sentence_split[:change_i+1]

      next_prob_list, next_word_list = get_avg_distr(model_list, ' '.join(cur_context), sentence_split[change_i+1], vocab)

      n = list(np.random.multinomial(1,next_prob_list))
      ind = n.index(1)
      new_word = next_word_list[ind]
      modified_sentence_additions.insert(change_i+1,str(new_word))
      new_context = ' '.join(modified_sentence_additions)
      js_dict[(new_word,"A")] = evaluate_sentence(model_list, new_context, vocab)
      modified_sentence_additions.pop(change_i+1)

    highest_js_word = sorted(js_dict.items(), key=lambda x: discounting(change_i,x[1][1]), reverse=True)[0]
    
    if highest_js_word[1] == "R":
      final_modified_sentence[change_i] = highest_js_word[0]
      change = "R"
    elif highest_js_word[1] == "A":
      final_modified_sentence.insert(change_i,highest_js_word[0])
      change = "A"
    else: 
      final_modified_sentence.pop(change_i)
      change = "D"

    new_sentence_score, new_js_positions = evaluate_sentence(model_list, ' '.join(final_modified_sentence), vocab)

    new_discounted_score = discounting(change_i, new_js_positions)
    curr_discounted_score = discounting(change_i, cur_js_positions)

    if change == "D":
      new_js_positions.insert(change_i, 0)

    if new_discounted_score > curr_discounted_score:
      scores.append(new_sentence_score)
      js_positions.append(new_js_positions)
      changes.append(change)
      sentence_split = final_modified_sentence
      print("Here is the new version of the sentence: ", ' '.join(sentence_split), " and the change made was ", change)

  print("New sentence is: ", ' '.join(sentence_split)," with total JS:", evaluate_sentence(model_list, ' '.join(sentence_split), vocab)[0])

  print("Scores", scores, "Changes", changes)
  return scores, js_positions, ' '.join(sentence_split)


def plot_scores(scores, sentence):

  plt.plot(range(len(scores)),scores)
  plt.show()
  name = sentence + ".png"
  plt.savefig(name)
  plt.close()


def sample_sentences(file_name):

  file = open(file_name)
  reader = csv.reader(file)
  num_lines = len(list(reader))
  N = random.randint(0,num_lines-1)

  with open(file_name, 'r') as file:
      reader = csv.reader(file)

      line = next((x for i, x in enumerate(reader) if i == N), None)

  return " ".join(line)

T5_PATH = "t5-base"
t5_config = T5Config.from_pretrained(T5_PATH, cache_dir='./pretrained_models')
roberta_config = RobertaConfig.from_pretrained("roberta-base")
roberta_config.is_decoder = True

filename = "SUBTLEXus74286wordstextversion.txt"
vocab = get_vocab(filename)

GPT2 = ModelInfo(GPT2LMHeadModel.from_pretrained('gpt2', return_dict =True), GPT2Tokenizer.from_pretrained('gpt2'), "Ġ", vocab)

TXL = ModelInfo(TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103'),TransfoXLTokenizer.from_pretrained('transfo-xl-wt103'), "_", vocab)

Roberta = ModelInfo(RobertaForCausalLM.from_pretrained('roberta-base', config=roberta_config), RobertaTokenizer.from_pretrained('roberta-base'), "_", vocab)

XLM = ModelInfo(XLMWithLMHeadModel.from_pretrained('xlm-mlm-xnli15-1024', return_dict=True), XLMTokenizer.from_pretrained('xlm-mlm-xnli15-1024'), "_", vocab)

T5 = ModelInfo(T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config, cache_dir='./pretrained_models'), T5Tokenizer.from_pretrained(T5_PATH, cache_dir='./pretrained_models'), "_", vocab)

Albert = (AlbertForMaskedLM.from_pretrained('albert-base-v2', return_dict=True), AlbertTokenizer.from_pretrained('albert-base-v2'), "_", vocab)


model_list = [GPT2, XLM]

for i in range(1):

 # sent = ' '.join(sample_sentences("sentences4lara.txt").split())
  sent  = "I am going to sleep"
  scores, js_positions, sentence = change_sentence(model_list, sent, vocab)
  #plot_scores(scores, sentence)
