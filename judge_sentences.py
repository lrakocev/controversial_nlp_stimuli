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
  for i in range(0, len_sentence):
    curr_context += sentence_split[i]
    p = get_distribution(model_info, 'GPT2', curr_context, joint_vocab)
    q = get_distribution(model_info,'TransformerXL', curr_context, joint_vocab)
    total_js += js(p,q)

  return total_js/len_sentence


def replace_words(model_info, sentence, joint_vocab, num_replacements, top_k):

  original_score = evaluate_sentence(model_info, sentence, joint_vocab)
  print("Old sentence is: ", sentence, " with JS: ", original_score)
  scores = [original_score]
  sentence_split = sentence.split(" ")
  modified_sentence = copy.copy(sentence_split)
  len_sentence = len(sentence_split)
  total_replacements = 0

  for i in range(0, num_replacements):
    replace_i = random.randint(0, len_sentence-1)
    distrs = {}
    for model_name in ['GPT2','TransformerXL']:
      model, tokenizer = model_info[model_name]
      next_word_distr = get_distribution(model_info, model_name, ' '.join(modified_sentence), joint_vocab)
      distrs[model_name] = next_word_distr

    A = distrs['GPT2']
    B = distrs['TransformerXL']
    avg_distr = {x: (A.get(x, 0) + B.get(x, 0))/2 for x in set(A).intersection(B)}

    avg_distr = {k: v for (k,v) in avg_distr.items() if v >= top_k}

    total = sum(avg_distr.values())

    avg_distr = {k: v/total for (k,v) in avg_distr.items()}
    
    prob_list = [v for k, v in sorted(avg_distr.items())]
    word_list = [k for k, v in sorted(avg_distr.items())]

    curr_sentence_score = evaluate_sentence(model_info, ' '.join(sentence_split), joint_vocab)
    js_dict = {}
    for j in range(0,10):
      n = list(np.random.multinomial(1,prob_list))
      ind = n.index(1)
      new_word = word_list[ind]
      modified_sentence[replace_i] = new_word
      new_context = ' '.join(modified_sentence)
      js_dict[new_word] = evaluate_sentence(model_info, new_context, joint_vocab)

    highest_js_word = sorted(js_dict.items(), key=lambda x: x[1], reverse=True)[0]
    modified_sentence[replace_i] = highest_js_word[0]
    new_sentence_score = evaluate_sentence(model_info, ' '.join(modified_sentence), joint_vocab)
    
    if new_sentence_score > curr_sentence_score:
      scores.append(new_sentence_score)
      total_replacements +=1
      print(modified_sentence)
      sentence_split = modified_sentence
    

  print("New sentence is: ", ' '.join(sentence_split)," with JS:", evaluate_sentence(model_info, ' '.join(sentence_split), joint_vocab))
  print(len(scores), total_replacements)
  return scores, ' '.join(sentence_split)

def plot_scores(scores, sentence):

  plt.plot(range(0,len(scores)),scores)
  plt.show()
  plt.savefig(sentence)


def sample_sentences(file_name, ):

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


for i in range(5):

  sent = sample_sentences("sentences4lara.txt")

  scores, sentence = replace_words(model_info, sent, joint_vocab, 10, .005)
  plot_scores(scores, sentence)


