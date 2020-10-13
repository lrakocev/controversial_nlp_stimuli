import numpy as np
from scipy.stats import entropy
import tensorflow as tf
import torch
from transformers import  TFGPT2LMHeadModel, GPT2Tokenizer, TFTransfoXLLMHeadModel, TransfoXLTokenizer, T5Tokenizer, TFT5ForConditionalGeneration, T5Config, AlbertTokenizer, AlbertModel, RobertaTokenizer, RobertaModel,  XLMTokenizer, XLMModel
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

def get_distribution(model_info, model_name, context, joint_vocab):

  tokenizer, model = model_info[model_name]

  inputs = tokenizer(context)

  outputs = model(input_ids = inputs) if model_name == "t5-11b" else model(inputs)

  ids = range(0,tokenizer.vocab_size)
  vocab = tokenizer.convert_ids_to_tokens(ids)

  final_vocab = set(vocab) & joint_vocab if len(joint_vocab) != 0 else set(vocab)

  id_list = tokenizer.convert_tokens_to_ids(sorted(final_vocab))

  outputs_array = np.asarray(outputs[0]).flatten()

  final_outputs = [outputs_array[i] for i in id_list] 

  probabilities = softmax(final_outputs)

  distr_dict = dict(zip(final_vocab, probabilities))

  return distr_dict


def entropy(prob_dist, base=math.e):
        return -sum([p * math.log(p,base) for p in prob_dist if p != 0])

def jsd(prob_dists, base=math.e):
    weight = 1/len(prob_dists) #all same weight
    js_left = [0,0,0]
    js_right = 0    
    for pd in prob_dists:
        js_left[0] += pd[0]*weight
        js_left[1] += pd[1]*weight
        js_left[2] += pd[2]*weight
        js_right += weight*entropy(pd,base)
    return entropy(js_left)-js_right


def evaluate_sentence(model_info, sentence, joint_vocab):

  sentence_split = sentence.split(" ")
  len_sentence = len(sentence_split)

  curr_context = ""
  total_js = 0
  js_positions = []
  distrs = {}

  for i in range(0, len_sentence):
    curr_context += sentence_split[i] + " "
    
    for model_name in model_info.keys():
      tokenizer, model = model_info[model_name]

      next_word_distr = get_distribution(model_info, model_name, curr_context, joint_vocab)
      distrs[model_name] = next_word_distr

    total_js += jsd(distrs.values())
    curr_js = total_js/(i+1)
    js_positions.append(curr_js)
    
  return total_js/len_sentence, js_positions


def get_avg_distr(model_info, context, joint_vocab, top_p):

    distrs = {}
    for model_name in model_info.keys():
      tokenizer, model = model_info[model_name]

      next_word_distr = get_distribution(model_info, model_name, context, joint_vocab)
      distrs[model_name] = next_word_distr

    df = pd.DataFrame(distrs.values())
    avg_distr = dict(df.mean())

    avg_distr = {x: avg_distr[x] for x in joint_vocab}

    avg_distr_sorted_keys = [k for (k,v) in sorted(avg_distr.items(), key=lambda x: x[1], reverse=True)]
    avg_distr_sorted_vals = [v for (k,v) in sorted(avg_distr.items(), key=lambda x: x[1], reverse=True)]

    avg_distr_vals = np.cumsum(np.array(avg_distr_sorted_vals))

    avg_distr_summed = zip(avg_distr_sorted_keys, avg_distr_vals)

    avg_distr = {k: avg_distr[k] for (k, v) in avg_distr_summed if v <= top_p}

    prob_list = [v for k, v in sorted(avg_distr.items())]
    word_list = [k for k, v in sorted(avg_distr.items())]

    return prob_list, word_list

def discounting(cur_ind, js_positions, gamma=0.9):

  total = 0
  for i in range(len(js_positions)-cur_ind):
    total += js_positions[cur_ind+i]*(gamma**i)

  return total/(len(js_positions)-cur_ind+1)


def change_sentence(model_info, sentence, joint_vocab, top_p):

  original_score, original_js_positions = evaluate_sentence(model_info, sentence, joint_vocab)
  print("Old sentence is: ", sentence, " with JS: ", original_score, " and positional JS scores: ", original_js_positions)
  scores = [original_score]
  js_positions = [original_js_positions]
  changes = []
  change = ""
  sentence_split = sentence.split(" ")
  len_sentence = len(sentence_split)
  final_modified_sentence = copy.deepcopy(sentence_split)

  for i in range(0, num_changes):
    print("Round ", i, " and the sentence to be changed is ", ' '.join(sentence_split))
    curr_sentence_score, cur_js_positions = evaluate_sentence(model_info, ' '.join(sentence_split), joint_vocab)

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

        cur_prob_list, cur_word_list = get_avg_distr(model_info, ' '.join(cur_context), joint_vocab, top_p)

        n = list(np.random.multinomial(1,cur_prob_list))
        ind = n.index(1)
        new_word = cur_word_list[ind]
        modified_sentence_replacements[change_i] = new_word
        new_context = ' '.join(modified_sentence_replacements)
        js_dict[(new_word,"R")] = evaluate_sentence(model_info, new_context, joint_vocab)
      

      #deletions
      modified_sentence_deletions.pop(change_i)
      if len(modified_sentence_deletions) > 0:
        js_dict[("", "D")] = evaluate_sentence(model_info, ' '.join(modified_sentence_deletions), joint_vocab)
      else: 
        js_dict[("", "D")] = (0,[0])


      # additions
      for k in range(0,10):
        cur_context = sentence_split[:change_i+1]

        next_prob_list, next_word_list = get_avg_distr(model_info, ' '.join(cur_context), joint_vocab, top_p)

        n = list(np.random.multinomial(1,next_prob_list))
        ind = n.index(1)
        new_word = next_word_list[ind]
        modified_sentence_additions.insert(change_i+1,new_word)
        new_context = ' '.join(modified_sentence_additions)
        js_dict[(new_word,"A")] = evaluate_sentence(model_info, new_context, joint_vocab)
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

      new_sentence_score, new_js_positions = evaluate_sentence(model_info, ' '.join(final_modified_sentence), joint_vocab)


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

  print("New sentence is: ", ' '.join(sentence_split)," with total JS:", evaluate_sentence(model_info, ' '.join(sentence_split), joint_vocab)[0])

  print("Scores", scores, "Changes", changes)
  return scores, js_positions, ' '.join(sentence_split)


def plot_scores(scores, sentence):

  plt.plot(range(len(scores)),scores)
  plt.show()
  name = sentence + ".png"
  plt.savefig(name)
  plt.close()

'''
def plot_positions(js_positions, sentence):

  print("plot positions", js_positions)

  plt.plot(js_positions)
  plt.show()
  name = "positions of: " + sentence + ".png"
  plt.savefig(name)
  plt.close()
'''

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


model_info = {"xlm-mlm-xnli15-1024": (XLMTokenizer.from_pretrained('xlm-mlm-en-2048'), XLMModel.from_pretrained('xlm-mlm-en-2048'))
              "roberta-base": (RobertaTokenizer.from_pretrained('roberta-base'),RobertaModel.from_pretrained('roberta-base')),
              "albert-base-v2": (AlbertTokenizer.from_pretrained('albert-base-v2'),AlbertModel.from_pretrained('albert-base-v2'))}

'''
#(XLMTokenizer.from_pretrained("xlm-mlm-xnli15-1024"), XLMModel.from_pretrained(" xlm-mlm-xnli15-1024")),
"gpt2": (GPT2Tokenizer.from_pretrained('gpt2'), TFGPT2LMHeadModel.from_pretrained('gpt2')), 
"transfo-xl-wt103": (TransfoXLTokenizer.from_pretrained('transfo-xl-wt103'),TFTransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')),
"t5-11b": (T5Tokenizer.from_pretrained(T5_PATH, cache_dir='./pretrained_models'),TFT5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config, cache_dir='./pretrained_models'))
'''
curr_context = "I"
distrs = {}
for model_name in model_info.keys():
    tokenizer, model = model_info[model_name]

    next_word_distr = get_distribution(model_info, model_name, curr_context, {})
    distrs[model_name] = next_word_distr

joint_vocab = set(distrs["GPT2"].keys()).intersection(*distrs.values().keys())

for i in range(5):

  sent = ' '.join(sample_sentences("sentences4lara.txt").split())
  scores, js_positions, sentence = change_sentence(model_info, sent, joint_vocab, .9)
  plot_scores(scores, sentence)
  #plot_positions(js_positions,sentence)
