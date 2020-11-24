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


class ModelInfo():

  def __init__(self, model, tokenizer, start_token_symbol, vocab, model_name, score_name):
    self.model = model #.to("cuda")
    self.tokenizer = tokenizer
    self.start_token_symbol = start_token_symbol
    self.model_name = model_name
    self.word_token_dict = {word: self.tokenizer.tokenize(" " + str(word)) for word in vocab}

    all_tokens = list(self.word_token_dict.values())

    all_tokens = reduce(lambda x,y: x+y,all_tokens)

    self.id_token_dict = {token: self.tokenizer.convert_tokens_to_ids(token) for token in all_tokens}

    self.distr_dict_for_context = {}
    self.score_name = score_name

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

def cosine_distance(prob_distributions, weights):

  cosine_list = []
  for i in combinations(prob_distributions, 2):
    distr1, distr2 = i
    cosine_list.append(distance.cosin(distr1, distr2))

  return sum(cosine_list)/len(cosine_list)


def get_prediction(score_name, tokenizer, model, sentence):

  s = pd.read_pickle(score_name)
  d = s['data']

  coeffs = d.layer_weights[0][-1].values

  intercept = d.layer_weights[0][-1].intercept.values

  new_model = LinearRegression()
  new_model.intercept_ = intercept
  new_model.coef_ = coeffs

  inputs = tokenizer(sentence, return_tensors="pt")
  outputs = model(**inputs, labels=inputs["input_ids"], output_hidden_states=True)

  hiddenStates = outputs.hidden_states 

  hiddenStatesLayer = hiddenStates[-1]

  lastWordState = hiddenStatesLayer[-1, :].detach().numpy()

  lastWordState = lastWordState[-1].reshape(1, -1)

  prediction = new_model.predict(lastWordState)
  
  return prediction


def evaluate_sentence(model_list, sentence, vocab, n, js_dict):

  sentence_split = sentence.split(" ")
  len_sentence = len(sentence_split)

  curr_context = ""
  distrs = {}

  for model_name in model_list:
    score_name = model_name.score_name
    model = model_name.model
    tokenizer = model_name.tokenizer
    prediction = get_prediction(score_name, tokenizer, model, sentence)
    distrs[model_name] = prediction

    n = len(model_list)
    weights = np.empty(n)
    weights.fill(1/n)

    curr_cosine = cosine_distance(list(distrs.values()), weights)

  return curr_cosine

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

  context[change_i] = '[MASK]'
  if num_masks == 2:
    context.insert(change_i+1,'[MASK]')

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True)

  inputs = tokenizer(" ".join(context), return_tensors='pt')
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
  changes = []
  change = ""
  sentence_split = sentence.split(" ")
  len_sentence = len(sentence_split)

  for change_i in range(0,num_changes):

    curr_score= evaluate_sentence(model_list, ' '.join(sentence_split), vocab, batch_size)
    
    print("Curr sentence is: ", sentence, " with JS: ", curr_score)

    scores.append(curr_score)

    #exponentiated_scores = torch.tensor(softmax(curr_js_positions))
    #n = list(torch.multinomial(exponentiated_scores, 1))
    #change_i = n[0]
    change_i = random.randint(1, len(sentence_split)-1)

    final_modified_sentence = copy.deepcopy(sentence_split)
    modified_sentence_replacements = copy.deepcopy(sentence_split)
    modified_sentence_deletions = copy.deepcopy(sentence_split)
    modified_sentence_additions = copy.deepcopy(sentence_split)

    new_sentence_list = []

    # replacements 
    num_masks = random.randint(1,2)

    new_word_list = sample_bert(sentence_split, change_i, num_masks, 3)

    for words in new_word_list: 
      modified_sentence_replacements[change_i] = str(words[0])
      if num_masks == 2 and len(modified_sentence_replacements) > change_i + 1:
        modified_sentence_replacements[change_i+1] = str(words[1])
      elif num_masks == 2 and len(modified_sentence_replacements) <= change_i + 1:
        modified_sentence_replacements.insert(change_i+1,str(words[1]))

      new_context = ' '.join(modified_sentence_replacements)
      new_sentence_list.append(new_context)
      

    #deletions
    modified_sentence_deletions.pop(change_i)
    if len(modified_sentence_deletions) > 0:
      print("deletion try", ' '.join(modified_sentence_deletions))
      new_sentence_list.append(' '.join(modified_sentence_deletions))

    # additions
    num_masks = random.randint(1,2)
    new_word_list = sample_bert(sentence_split, change_i, num_masks, 3)
    for words in new_word_list:
      print("mod sentence additions", modified_sentence_additions)
      modified_sentence_additions.insert(change_i+1,str(words[0]))
      if num_masks == 2:
        modified_sentence_additions.insert(change_i+2,str(words[1]))

      new_context = ' '.join(modified_sentence_additions)

      new_sentence_list.append(new_context)

    sampled_id = random.randint(0, len(new_sentence_list))
    final_modified_sentence = new_sentence_list[sampled_id]

    new_sentence_score= evaluate_sentence(model_list, final_modified_sentence, vocab, batch_size)

    if new_sentence_score > curr_score:
      change = highest_js_word[0][1]
      print("new score", new_sentence_score, "curr_score", curr_score)
      print("Here is the new version of the sentence: ", ' '.join(sentence_split), " and the change made was ", change)
      sentence_split = final_modified_sentence.split(" ")

  print("New sentence is: ", ' '.join(sentence_split)," with total scores: ", scores)

  plot_scores(scores, ' '.join(sentence_split))

  return scores, ' '.join(sentence_split)

def sample_sentences(file_name, n):

  with open(file_name) as f:
    head = [next(f).strip() for x in range(n)]

  return head 

filename = "SUBTLEXus74286wordstextversion.txt"
vocab = get_vocab(filename, 3000)

GPT2 = ModelInfo(GPT2LMHeadModel.from_pretrained('gpt2', return_dict =True), GPT2Tokenizer.from_pretrained('gpt2'), "Ġ", vocab, "GTP2", '/om2/user/gretatu/.result_caching/neural_nlp.score/benchmark=Pereira2018-encoding-weights,model=gpt2,subsample=None.pkl')

Roberta = ModelInfo(RobertaForCausalLM.from_pretrained('roberta-base',  return_dict=True), RobertaTokenizer.from_pretrained('roberta-base'), "_", vocab, "Roberta", '/om2/user/gretatu/.result_caching/neural_nlp.score/benchmark=Pereira2018-encoding-weights,model=roberta-base,subsample=None.pkl')

XLM = ModelInfo(XLMWithLMHeadModel.from_pretrained('xlm-mlm-xnli15-1024', return_dict=True), XLMTokenizer.from_pretrained('xlm-mlm-xnli15-1024'), "_", vocab, "XLM", '/om2/user/gretatu/.result_caching/neural_nlp.score/benchmark=benchmark=Pereira2018-encoding-weights,model=xlm-mlm-xnli15-1024,subsample=None.pkl')


model_list = [GPT2, Roberta] #, XLM, T5, Albert]

sentences = sorted(sample_sentences("sentences4lara.txt", 100))

if __name__ == "__main__":

  sentences = sorted(sample_sentences("sentences4lara.txt", 100))

  sent_dict = dict(zip([str(x) for x in range(1,100)], sentences))

  sentence = sent_dict[sys.argv[2]]

  globals()[sys.argv[1]](model_list, sentence, vocab, 100, 3, {})
