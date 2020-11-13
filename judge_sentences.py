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
from functools import reduce


class ModelInfo():

  def __init__(self, model, tokenizer, start_token_symbol, vocab, model_name):
    self.model = model
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


def get_distribution(model_name, context, vocab, n):

  print("context", context)

  tokenizer = model_name.tokenizer 
  model = model_name.model
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

    inputs = tokenizer(batch_list, padding='longest', return_tensors="pt")

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

         
      attention_mask = torch.tensor(attention_mask) 
      input_ids = torch.tensor(input_ids) 

      outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
    else:
      outputs = model(**inputs, labels=inputs["input_ids"])

    vectorize_log = np.vectorize(math.log)

    log_probabilities = [[vectorize_log(softmax(np.asarray(outputs.logits[j][i].detach()).flatten())) for i in range(max_length-1,max_length - lengths_contexts[j],-1)] for j in range(len(batch_list))]

    log_probabilities_per_tokens = [[log_probabilities[j][i][id_nums[j][i]] for i in range(len(id_nums[j])-1)] for j in range(len(batch_list))]

    probabilities = [sum(log_probabilities_per_tokens[i]) for i in range(len(log_probabilities_per_tokens))]

    final_probabilities.update({words[i]: probabilities[i] for i in range(len(words))})

  
  #normalizing
  final_probabilities_total = sum(final_probabilities.values())
  final_probabilities = {k: v / total for k, v in final_probabilties.items()}

  model_name.distr_dict_for_context[context] = final_probabilities


  return final_probabilities


def jsd(prob_distributions, weights, logbase=math.e):

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


def evaluate_sentence(model_list, sentence, vocab, n):

  sentence_split = sentence.split(" ")
  len_sentence = len(sentence_split)

  curr_context = ""
  total_js = 0
  js_positions = []
  distrs = {}

  for i in range(0, len_sentence):
    curr_context += sentence_split[i] + " "
    
    for model_name in model_list:
      tokenizer = model_name.tokenizer
      model = model_name.model
      next_word_distr = get_distribution(model_name, curr_context, vocab, n)
      distrs[model_name] = list(next_word_distr.values())

    n = len(model_list)
    weights = np.empty(n)
    weights.fill(1/n)

    curr_js = jsd(list(distrs.values()), weights)
    #total_js += jsd(list(distrs.values()), weights)
    total_js += curr_js
    #curr_js = total_js/(i+1)
    js_positions.append(curr_js)
    
  return total_js/len_sentence, js_positions


def get_avg_distr(model_list, context, vocab, n):

    distrs = {}
    for model_name in model_list:
      tokenizer = model_name.tokenizer
      model = model_name.model

      next_word_distr = get_distribution(model_name, context, vocab, n)
      distrs[model_name] = [v for (k,v) in sorted(next_word_distr.items(), key = lambda x: x[0])]
    
    sorted_vocab = [k for (k,v) in sorted(distrs.values()[0].items(), key = lambda x: x[0])]

    df_probabilities = pd.DataFrame(distrs.values())

    df_probabilities_mean = df_probabilities.mean()

    avg_distr = dict(zip(sorted_vocab, df_probabilities_mean))

    #avg_distr = dict(df_probabilities.mean())
    #avg_distr_vals = [v for (k,v) in avg_distr.items()]
    #avg_distr_sorted_vals = [v for (k,v) in sorted(avg_distr.items(), key=lambda x: x[1], reverse=True)]
    #avg_distr_vals = np.cumsum(np.array(avg_distr_sorted_vals))
    #avg_distr_summed = dict(zip(vocab, avg_distr_vals))

    prob_list_sum = sum(df_probabilities_mean)
    prob_list = [v/prob_list_sum for (k, v) in avg_distr.items()]
    
    return prob_list, sorted_vocab

def discounting(cur_ind, js_positions, gamma=1):

  total = 0
  to_consider = len(js_positions) - cur_ind
  for i in range(to_consider):
    total += js_positions[cur_ind+i]*(gamma**i)

  length_js_pos = 1 if to_consider == 0 else to_consider
  return total/length_js_pos


def change_sentence(model_list, sentence, vocab, batch_size, num_changes):

  original_score, original_js_positions = evaluate_sentence(model_list, sentence, vocab, batch_size)
  print("Old sentence is: ", sentence, " with JS: ", original_score, " and positional JS scores: ", original_js_positions)
  scores = [original_score]
  js_positions = [original_js_positions]
  changes = []
  change = ""
  sentence_split = sentence.split(" ")
  len_sentence = len(sentence_split)

  for change_i in range(0,num_changes):

    exponentiated_scores = softmax(original_js_positions)
    n = list(np.random.multinomial(1,exponentiated_scores))
    change_i = n.index(1)

    #change_i = change_i-1 if change == "D" else change_i
    #change = ""

    print("current starting sentence", sentence_split)

    curr_score, curr_js_positions = evaluate_sentence(model_list, ' '.join(sentence_split), vocab, batch_size)

    final_modified_sentence = copy.deepcopy(sentence_split)
    modified_sentence_replacements = copy.deepcopy(sentence_split)
    modified_sentence_deletions = copy.deepcopy(sentence_split)
    modified_sentence_additions = copy.deepcopy(sentence_split)

    js_dict = {}

    # replacements 
    for j in range(0,5):
      cur_context = sentence_split[:change_i+1]

      cur_prob_list, cur_word_list = get_avg_distr(model_list, ' '.join(cur_context) + " ", vocab, batch_size)
     
      n = list(np.random.multinomial(1,cur_prob_list))
      ind = n.index(1)
      new_word = cur_word_list[ind]
      modified_sentence_replacements[change_i] = str(new_word)

      new_context = ' '.join(modified_sentence_replacements)

      print("replacement try", new_context)
      js_dict[(new_word,"R")] = evaluate_sentence(model_list, new_context, vocab, batch_size)
    

    #deletions
    modified_sentence_deletions.pop(change_i)
    if len(modified_sentence_deletions) > 0:

      print("deletion try", ' '.join(modified_sentence_deletions))
      js_dict[("", "D")] = evaluate_sentence(model_list, ' '.join(modified_sentence_deletions), vocab, batch_size)


    # additions
    for k in range(0,5):
      cur_context = sentence_split[:change_i+1]

      next_prob_list, next_word_list = get_avg_distr(model_list, ' '.join(cur_context) + " ", vocab, batch_size)

      n = list(np.random.multinomial(1,next_prob_list))
      ind = n.index(1)
      new_word = next_word_list[ind]
      modified_sentence_additions.insert(change_i+1,str(new_word))
      new_context = ' '.join(modified_sentence_additions)

      print("additions try", new_context)
      js_dict[(new_word,"A")] = evaluate_sentence(model_list, new_context, vocab, batch_size)
      modified_sentence_additions.pop(change_i+1)


    highest_js_word = sorted(js_dict.items(), key=lambda x: discounting(change_i,x[1][1]), reverse=True)[0]
    

    if highest_js_word[0][1] == "R":
      final_modified_sentence[change_i] = highest_js_word[0][0]
    elif highest_js_word[0][1] == "A":
      final_modified_sentence.insert(change_i,highest_js_word[0][0])
    else: 
      final_modified_sentence.pop(change_i)

    new_sentence_score, new_js_positions= evaluate_sentence(model_list, ' '.join(final_modified_sentence), vocab, batch_size)

    new_discounted_score = discounting(change_i, new_js_positions)
    curr_discounted_score = discounting(change_i, curr_js_positions)


    if new_discounted_score > curr_discounted_score:
      scores.append(new_sentence_score)
      js_positions.append(new_js_positions)
      change = highest_js_word[0][1]
      changes.append(change)
      sentence_split = final_modified_sentence
      print("new score", new_discounted_score, "curr_score", curr_discounted_score)
      print("Here is the new version of the sentence: ", ' '.join(sentence_split), " and the change made was ", change)

  print("New sentence is: ", ' '.join(sentence_split)," with total JS:", evaluate_sentence(model_list, ' '.join(sentence_split), vocab, batch_size)[0])

  print("Scores", scores, "Changes", changes)
  return scores, js_positions, ' '.join(sentence_split)


def plot_scores(scores, sentence):

  plt.plot(range(len(scores)),scores)
  plt.show()
  name = sentence + ".png"
  plt.savefig(name)
  plt.close()

def plot_positions(js_positions, sentence):


  for pos in js_positions:
    plt.plot(pos)
  ticks = sentence.split(" ")
  plt.xticks(np.arange(len(ticks)), ticks)
  plt.show()
  name = sentence + " positions.png"
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


filename = "SUBTLEXus74286wordstextversion.txt"
vocab = get_vocab(filename, 1000)

GPT2 = ModelInfo(GPT2LMHeadModel.from_pretrained('gpt2', return_dict =True), GPT2Tokenizer.from_pretrained('gpt2'), "Ġ", vocab, "GTP2")

TXL = ModelInfo(TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103'),TransfoXLTokenizer.from_pretrained('transfo-xl-wt103'), "_", vocab, "TXL")

Roberta = ModelInfo(RobertaForCausalLM.from_pretrained('roberta-base',  return_dict=True), RobertaTokenizer.from_pretrained('roberta-base'), "_", vocab, "Roberta")

XLM = ModelInfo(XLMWithLMHeadModel.from_pretrained('xlm-mlm-xnli15-1024', return_dict=True), XLMTokenizer.from_pretrained('xlm-mlm-xnli15-1024'), "_", vocab, "XLM")

T5 = ModelInfo(T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True), T5Tokenizer.from_pretrained("t5-base"), "_", vocab, "T5")

Albert = ModelInfo(AlbertForMaskedLM.from_pretrained('albert-base-v2', return_dict=True), AlbertTokenizer.from_pretrained('albert-base-v2'), "_", vocab, "Albert")


model_list = [Albert, GPT2] #, Roberta, XLM, T5] 

for i in range(1):

  sent = ' '.join(sample_sentences("sentences4lara.txt").split())
  scores, js_positions, sentence = change_sentence(model_list, sent, vocab, 100, 5)
  plot_scores(scores, sentence)
  plot_positions(js_positions, sentence)
