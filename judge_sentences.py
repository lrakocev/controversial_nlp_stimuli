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

    #logsoftmax = torch.nn.LogSoftmax(dim=0)

    vectorize_log = np.vectorize(math.log)

    log_probabilities = [[vectorize_log(softmax(np.asarray(outputs.logits[j][i].cpu().detach()).flatten())) for i in range(max_length-1,max_length - lengths_contexts[j],-1)] for j in range(len(batch_list))]

    log_probabilities_per_tokens = [[log_probabilities[j][i][id_nums[j][i]] for i in range(len(id_nums[j])-1)] for j in range(len(batch_list))]

    probabilities = [sum(log_probabilities_per_tokens[i]) for i in range(len(log_probabilities_per_tokens))]

    final_probabilities.update({words[i]: probabilities[i] for i in range(len(words))})

  
  #normalizing
  final_probabilities_total = sum(final_probabilities.values())
  final_probabilities = {k: v / final_probabilities_total for k, v in final_probabilities.items()}

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

      n = len(model_list)
      weights = np.empty(n)
      weights.fill(1/n)

      curr_js = jsd(list(distrs.values()), weights)
      js_dict[curr_context] = curr_js

    total_js += curr_js
    js_positions.append(curr_js)
    
  return total_js/len_sentence, js_positions

def get_avg_distr(model_list, context, vocab, n):

    distrs = {}
    for model_name in model_list:
      tokenizer = model_name.tokenizer
      model = model_name.model

      next_word_distr = get_distribution(model_name, context, vocab, n)
      distrs[model_name] = [v for (k,v) in sorted(next_word_distr.items(), key = lambda x: x[0])]
    
      sorted_vocab = [k for (k,v) in sorted(next_word_distr.items(), key = lambda x: x[0])]

    df_probabilities = pd.DataFrame(distrs.values())

    df_probabilities_mean = df_probabilities.mean()

    avg_distr = dict(zip(sorted_vocab, df_probabilities_mean))

    prob_list_sum = sum(df_probabilities_mean)
    prob_list = [v/prob_list_sum for (k, v) in avg_distr.items()]

    return prob_list, sorted_vocab

def sample_bert(context, change_i, num_masks, top_k):

  new_context = copy.copy(context)
  new_context[change_i] = '[MASK]'
  if num_masks == 2:
    new_context.insert(change_i+1,'[MASK]')

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True)

  inputs = tokenizer(" ".join(new_context), return_tensors='pt')
  outputs = model(**inputs)
  predictions = outputs[0]

  predicted_indices = torch.topk(predictions[0, change_i], top_k).indices.to('cuda')
  predicted_tokens = tokenizer.convert_ids_to_tokens([predicted_indices[x] for x in range(top_k)])

  if num_masks == 2:
    predicted_indices_2 = torch.topk(predictions[0, change_i+1], top_k).indices.to('cuda')
    predicted_tokens_2 = tokenizer.convert_ids_to_tokens([predicted_indices_2[x] for x in range(top_k)])
    predicted_tokens = list(zip(predicted_tokens, predicted_tokens_2))

  print(predicted_tokens)

  return predicted_tokens

def discounting(cur_ind, js_positions, gamma=1):

  total = 0
  to_consider = len(js_positions) - cur_ind
  for i in range(to_consider):
    total += js_positions[cur_ind+i]*(gamma**i)

  length_js_pos = 1 if to_consider == 0 else to_consider
  return total/length_js_pos

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

def change_sentence(model_list, sentence, vocab, batch_size, num_changes, js_prev_dict):
  
  scores = []
  js_positions = []
  changes = []
  change = ""
  sentence_split = sentence.split(" ")
  len_sentence = len(sentence_split)

  for change_i in range(0,num_changes):

    curr_score, curr_js_positions = evaluate_sentence(model_list, ' '.join(sentence_split), vocab, batch_size, js_prev_dict)
    scores.append(curr_score)
    js_positions.append(curr_js_positions)
    exponentiated_scores = torch.tensor(softmax(curr_js_positions)).to('cuda')
    n = list(torch.multinomial(exponentiated_scores, 1))
    change_i = n[0]

    print("current starting sentence", sentence_split)

    modified_sentence_replacements = copy.copy(sentence_split)
    modified_sentence_deletions = copy.copy(sentence_split)
    modified_sentence_additions = copy.copy(sentence_split)

    js_dict = {}

    # replacements 
    num_masks = random.randint(1,2)

    new_word_list = sample_bert(sentence_split, change_i, num_masks, 3)

    for words in new_word_list: 
      if num_masks == 1:
         modified_sentence_replacements[change_i] = str(words)
      if num_masks == 2 and len(modified_sentence_replacements) > change_i + 1:
        modified_sentence_replacements[change_i] = str(words[0])
        modified_sentence_replacements[change_i+1] = str(words[1])
      elif num_masks == 2 and len(modified_sentence_replacements) <= change_i + 1:
        modified_sentence_replacements[change_i] = str(words[0])
        modified_sentence_replacements.insert(change_i+1,str(words[1]))

      new_context = ' '.join(modified_sentence_replacements)

      print("replacement try", new_context)
      js_dict[new_context] = evaluate_sentence(model_list, new_context, vocab, batch_size, js_prev_dict)
    

    #deletions
    modified_sentence_deletions.pop(change_i)
    if len(modified_sentence_deletions) > 0:

      print("deletion try", ' '.join(modified_sentence_deletions))
      js_dict[' '.join(modified_sentence_deletions)] = evaluate_sentence(model_list, ' '.join(modified_sentence_deletions), vocab, batch_size, js_prev_dict)


    # additions
    num_masks = random.randint(1,2)
    new_word_list = sample_bert(sentence_split, change_i, num_masks, 3)

    for words in new_word_list:
      print("words", words)
      if num_masks == 1:
        modified_sentence_additions.insert(change_i+1,str(words))
      if num_masks == 2:
        modified_sentence_additions.insert(change_i+1,str(words[0]))
        modified_sentence_additions.insert(change_i+2,str(words[1]))


      new_context = ' '.join(modified_sentence_additions)
      print("mod sentence additions", new_context)
      js_dict[new_context] = evaluate_sentence(model_list, new_context, vocab, batch_size, js_prev_dict)
      modified_sentence_additions = copy.copy(sentence_split)


    highest_js_word = sorted(js_dict.items(), key=lambda x: discounting(change_i,x[1][1]), reverse=True)[0]

    final_modified_sentence = highest_js_word[0]

    new_sentence_score, new_js_positions= evaluate_sentence(model_list, final_modified_sentence, vocab, batch_size, js_prev_dict)

    new_discounted_score = discounting(change_i, new_js_positions)
    curr_discounted_score = discounting(change_i, curr_js_positions)

    if new_discounted_score > curr_discounted_score:
      change = highest_js_word[0][1]
      changes.append(change)
      print("new score", new_discounted_score, "curr_score", curr_discounted_score)
      print("Here is the new version of the sentence: ", ' '.join(sentence_split), " and the change made was ", change)
      sentence_split = final_modified_sentence.split(" ")

  print("New sentence is: ", ' '.join(sentence_split)," with total scores: ", scores, " and js positions ", js_positions)

  plot_scores(scores, ' '.join(sentence_split))
  plot_positions(js_positions, ' '.join(sentence_split))

  return scores, js_positions, ' '.join(sentence_split)

def sample_sentences(file_name, n):

  with open(file_name) as f:
    head = [next(f).strip() for x in range(n)]

  return head 

filename = "SUBTLEXus74286wordstextversion.txt"
vocab = get_vocab(filename, 3000)

GPT2 = ModelInfo(GPT2LMHeadModel.from_pretrained('gpt2', return_dict =True), GPT2Tokenizer.from_pretrained('gpt2'), "Ġ", vocab, "GTP2")

Roberta = ModelInfo(RobertaForCausalLM.from_pretrained('roberta-base',  return_dict=True), RobertaTokenizer.from_pretrained('roberta-base'), "_", vocab, "Roberta")

XLM = ModelInfo(XLMWithLMHeadModel.from_pretrained('xlm-mlm-xnli15-1024', return_dict=True), XLMTokenizer.from_pretrained('xlm-mlm-xnli15-1024'), "_", vocab, "XLM")

T5 = ModelInfo(T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True), T5Tokenizer.from_pretrained("t5-base"), "_", vocab, "T5")

Albert = ModelInfo(AlbertForMaskedLM.from_pretrained('albert-base-v2', return_dict=True), AlbertTokenizer.from_pretrained('albert-base-v2'), "_", vocab, "Albert")

TXL = ModelInfo(TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103'),TransfoXLTokenizer.from_pretrained('transfo-xl-wt103'), "_", vocab, "TXL")

model_list = [GPT2, Roberta, Albert] #, XLM, T5, Albert]

sentences = sorted(sample_sentences("sentences4lara.txt", 100))

#print(sentences[0])

#change_sentence(model_list, sentences[0], vocab, 100, 3, {})

if __name__ == "__main__":

  sentences = sorted(sample_sentences("sentences4lara.txt", 100))

  sent_dict = dict(zip([str(x) for x in range(1,100)], sentences))

  sentence = sent_dict[sys.argv[2]]

  globals()[sys.argv[1]](model_list, sentence, vocab, 100, 5, {})
