import numpy as np
from scipy.stats import entropy
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
  model_word_token_dict = model_name.create_word_to_token_dict(vocab)

  tokens = tokenizer.tokenize(context)
  tokens = [tokenizer.bos_token] + tokens + [tokenizer.eos_token]

  print(tokens)

  ids = tokenizer.convert_tokens_to_ids(tokens)

  x = 1
  attention_mask = [1 for i in range(len(ids)-x)] + [0 for i in range(x)]
  attention_mask = torch.tensor(attention_mask).unsqueeze(0)

  input_ids = torch.tensor(ids).unsqueeze(0)

  outputs = model(input_ids, attention_mask=attention_mask)

  print(next_word)
  next_word_tokens = model_word_token_dict[str(next_word)]

  probabilities = softmax(outputs)
  if len(next_word_tokens) == 1:
    
    distr_dict = dict(zip(joint_vocab, probabilities))
  else: 
    log_probabilities = math.log(probabilites)
    n = len(next_word_tokens)
    probabilities = sum(log_probabilities[-n:])

  distr_dict = dict(zip(joint_vocab, probabilities))

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


def evaluate_sentence(model_list, sentence, vocab):

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

      next_word_distr = get_distribution(model_name, curr_context, sentence_split[i+1], vocab)
      distrs[model_name] = next_word_distr

    total_js += jsd(distrs.values())
    curr_js = total_js/(i+1)
    js_positions.append(curr_js)
    
  return total_js/len_sentence, js_positions


def get_avg_distr(model_list, context, next_word, vocab, top_p):

    distrs = {}
    for model_name in model_list:
      tokenizer = model_name.tokenizer
      model = model_name.model

      next_word_distr = get_distribution(model_name, context, next_word, vocab)
      distrs[model_name] = next_word_distr

    df = pd.DataFrame(distrs.values())
    avg_distr = dict(df.mean())

    avg_distr = {x: avg_distr[x] for x in vocab}

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


def change_sentence(model_list, sentence, vocab, top_p):

  original_score, original_js_positions = evaluate_sentence(model_list, sentence, vocab)
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

        cur_prob_list, cur_word_list = get_avg_distr(model_list, context, sentence_split[change_i+1], vocab, top_p)

        n = list(np.random.multinomial(1,cur_prob_list))
        ind = n.index(1)
        new_word = cur_word_list[ind]
        modified_sentence_replacements[change_i] = new_word
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

        next_prob_list, next_word_list = get_avg_distr(model_list, ' '.join(cur_context), sentence_split[change_i+1], vocab, top_p)

        n = list(np.random.multinomial(1,next_prob_list))
        ind = n.index(1)
        new_word = next_word_list[ind]
        modified_sentence_additions.insert(change_i+1,new_word)
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

GPT2 = ModelInfo(GPT2LMHeadModel.from_pretrained('gpt2'), GPT2Tokenizer.from_pretrained('gpt2'), "Ġ", vocab)
TXL = ModelInfo(TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103'),TransfoXLTokenizer.from_pretrained('transfo-xl-wt103'), "_", vocab)

'''
              "t5-11b": (T5Tokenizer.from_pretrained(T5_PATH, cache_dir='./pretrained_models'),T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config, cache_dir='./pretrained_models')),
              "xlm-mlm-xnli15-1024": (XLMTokenizer.from_pretrained('xlm-mlm-xnli15-1024'), XLMWithLMHeadModel.from_pretrained('xlm-mlm-xnli15-1024', return_dict=True)),
              "roberta-base": (RobertaTokenizer.from_pretrained('roberta-base'), RobertaForCausalLM.from_pretrained('roberta-base', config=roberta_config)),
              "albert-base-v2": (AlbertTokenizer.from_pretrained('albert-base-v2'),AlbertForMaskedLM.from_pretrained('albert-base-v2', return_dict=True))}
'''




model_list = [GPT2, TXL]

for i in range(1):

  sent = ' '.join(sample_sentences("sentences4lara.txt").split())
  scores, js_positions, sentence = change_sentence(model_list, sent, vocab, .9)
  #plot_scores(scores, sentence)
