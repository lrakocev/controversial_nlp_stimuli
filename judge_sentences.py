import numpy as np
from scipy.stats import entropy
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, TransfoXLTokenizer, TFTransfoXLLMHeadModel
import sys
from scipy.special import softmax
import torch
import random
import string
import copy
import random
import csv
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def get_distribution(model_info, model_name, context, joint_vocab):

  model, tokenizer = model_info[model_name]

  input = tokenizer(context,return_tensors='tf')
  outputs = model(input)

  ids = range(0,tokenizer.vocab_size)
  vocab = tokenizer.convert_ids_to_tokens(ids)

  final_vocab = set(vocab) & joint_vocab if len(joint_vocab) != 0 else set(vocab)

  id_list = tokenizer.convert_tokens_to_ids(sorted(final_vocab))

  outputs_array = np.asarray(outputs[0]).flatten()

  final_outputs = [outputs_array[i] for i in id_list] 

  probabilities = softmax(final_outputs)

  distr_dict = dict(zip(final_vocab, probabilities))

  return distr_dict


def js(p, q):

  p = [v for k, v in sorted(p.items())]
  q = [v for k, v in sorted(q.items())]

  p = np.asarray(p)
  q = np.asarray(q)

  # normalize
  p /= p.sum()
  q /= q.sum()
  m = (p + q) / 2
  return (entropy(p, m) + entropy(q, m)) / 2

def evaluate_sentence(model_info, sentence, joint_vocab):

  sentence_split = sentence.split(" ")
  len_sentence = len(sentence_split)

  curr_context = ""
  total_js = 0
  js_positions = []
  for i in range(0, len_sentence):
    curr_context += sentence_split[i]
    p = get_distribution(model_info, 'GPT2', curr_context, joint_vocab)
    q = get_distribution(model_info,'TransformerXL', curr_context, joint_vocab)
    current_js = js(p,q)
    total_js += current_js
    js_positions.append(current_js)

  return total_js/len_sentence, js_positions

def get_avg_distr(model_info, context_split, joint_vocab, top_p):

    distrs = {}
    for model_name in ['GPT2','TransformerXL']:
      model, tokenizer = model_info[model_name]
      next_word_distr = get_distribution(model_info, model_name, ' '.join(context_split), joint_vocab)
      distrs[model_name] = next_word_distr

    A = distrs['GPT2']
    B = distrs['TransformerXL']
    avg_distr = {x: (A.get(x, 0) + B.get(x, 0))/2 for x in set(A).intersection(B)}

    avg_distr_sorted_keys = [k for (k,v) in sorted(avg_distr.items(), key=lambda x: x[1], reverse=True)]
    avg_distr_sorted_vals = [v for (k,v) in sorted(avg_distr.items(), key=lambda x: x[1], reverse=True)]

    avg_distr_vals = np.cumsum(np.array(avg_distr_sorted_vals))

    avg_distr_summed = zip(avg_distr_sorted_keys, avg_distr_vals)

    avg_distr = {k: avg_distr[k] for (k, v) in avg_distr_summed if v <= top_p}

    prob_list = [v for k, v in sorted(avg_distr.items())]
    word_list = [k for k, v in sorted(avg_distr.items())]

    return prob_list, word_list

def change_sentence(model_info, sentence, joint_vocab, num_changes, top_p):

  original_score, original_js_positions = evaluate_sentence(model_info, sentence, joint_vocab)
  print("Old sentence is: ", sentence, " with JS: ", original_score, " and positional JS scores: ", original_js_positions)
  scores = [original_score]
  js_positions = [original_js_positions]
  changes = []
  sentence_split = sentence.split(" ")
  modified_sentence_replacements = copy.deepcopy(sentence_split)
  modified_sentence_deletions = copy.copy(sentence_split)
  modified_sentence_additions = copy.deepcopy(sentence_split)
  final_modified_sentence = copy.deepcopy(sentence_split)
  len_sentence = len(sentence_split)

  for i in range(0, num_changes):
    curr_sentence_score = evaluate_sentence(model_info, ' '.join(sentence_split), joint_vocab)[0]

    # deciding which position to change at 
    exponentiated_scores = softmax(original_js_positions)
    n = list(np.random.multinomial(1,exponentiated_scores))
    change_i = n.index(1)

    # replacements 
    js_dict = {}
    for j in range(0,10):
      cur_context = sentence_split[:change_i-1]
      cur_prob_list, cur_word_list = get_avg_distr(model_info, cur_context, joint_vocab, top_p)

      n = list(np.random.multinomial(1,cur_prob_list))
      ind = n.index(1)
      new_word = cur_word_list[ind]
      modified_sentence_replacements[change_i] = new_word
      new_context = ' '.join(modified_sentence_replacements)
      js_dict[(new_word,"R")] = evaluate_sentence(model_info, new_context, joint_vocab)

    # deletions
    modified_sentence_deletions.pop(change_i)
    js_dict[("", "D")] = evaluate_sentence(model_info, ' '.join(modified_sentence_deletions), joint_vocab)

    # additions
    for k in range(0,10):
      cur_context = sentence_split[:change_i]

      print(cur_context)
      next_prob_list, next_word_list = get_avg_distr(model_info, cur_context, joint_vocab, top_p)

      n = list(np.random.multinomial(1,next_prob_list))
      ind = n.index(1)
      new_word = next_word_list[ind]
      modified_sentence_additions.insert(change_i+1,new_word)
      new_context = ' '.join(modified_sentence_additions)
      js_dict[(new_word,"A")] = evaluate_sentence(model_info, new_context, joint_vocab)
      modified_sentence_additions.pop(change_i+1)

    highest_js_word = sorted(js_dict.items(), key=lambda x: x[1][0], reverse=True)[0]
    
    if highest_js_word[1] == "R":
      final_modified_sentence[change_i] = highest_js_word[0][0]
      change = "R"
    if highest_js_word[1] == "A":
      final_modified_sentence.insert(change_i,highest_js_word[0])
      change = "A"
    else: 
      final_modified_sentence.pop(change_i)
      change = "D"

    new_sentence_score, new_js_positions = evaluate_sentence(model_info, ' '.join(final_modified_sentence), joint_vocab)

    if change == "D":
      new_js_positions.insert(change_i, 0)

    if new_sentence_score > curr_sentence_score:
      scores.append(new_sentence_score)
      js_positions.append(new_js_positions)
      changes.append(change)
      sentence_split = final_modified_sentence
      print(' '.join(sentence_split))

  print("New sentence is: ", ' '.join(sentence_split)," with total JS:", evaluate_sentence(model_info, ' '.join(sentence_split), joint_vocab)[0])

  print(len(scores), "Changes", changes)
  return scores, js_positions, ' '.join(sentence_split)


def plot_scores(scores, sentence):

  plt.plot(range(len(scores)),scores)
  plt.show()
  plt.savefig(sentence)
  plt.close()

def plot_positions(js_positions, sentence):

  plt.plot(js_positions)
  plt.show()
  plt.savefig("positions of: " + sentence)
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


model_info = {"GPT2": (TFGPT2LMHeadModel.from_pretrained("gpt2"),GPT2Tokenizer.from_pretrained("gpt2")), 
              "TransformerXL": (TFTransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103'),TransfoXLTokenizer.from_pretrained('transfo-xl-wt103'))}

curr_context = "I"
gpt2_dict = get_distribution(model_info, "GPT2", curr_context, {})
txl_dict = get_distribution(model_info, "TransformerXL", curr_context, {})

joint_vocab = gpt2_dict.keys() & txl_dict.keys()

#for i in range(5):

sent = sample_sentences("sentences4lara.txt")
scores, js_positions, sentence = change_sentence(model_info, sent, joint_vocab, 2, .8)
plot_scores(scores, sentence)
plot_positions(js_positions,sentence)
