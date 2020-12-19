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

def evaluate_sentence_jsd(model_list, sentence, vocab, n, js_dict):

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

      curr_js = jsd(list(distrs.values()))
      js_dict[curr_context] = curr_js

    total_js += curr_js
    js_positions.append(curr_js)
    
  return total_js/len_sentence

def cosine_distance(prob_distributions):

  cosine_list = []
  for i in combinations(prob_distributions, 2):
    distr1, distr2 = i
    cosine_list.append(distance.cosine(distr1, distr2))

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


def evaluate_sentence_cosine(model_list, sentence, vocab, n, prev_dict={}):

  distrs = {}

  for model_name in model_list:
    score_name = model_name.score_name
    model = model_name.model
    tokenizer = model_name.tokenizer
    prediction = get_prediction(score_name, tokenizer, model, sentence)
    distrs[model_name] = prediction

  curr_cosine = cosine_distance(list(distrs.values()))

  return curr_cosine 

def sample_avg_distr(model_list, context, vocab, batch_size, top_k):

    distrs = {}
    for model_name in model_list:
      tokenizer = model_name.tokenizer
      model = model_name.model

      next_word_distr = get_distribution(model_name, " ".join(context), vocab, batch_size)
      distrs[model_name] = [v for (k,v) in sorted(next_word_distr.items(), key = lambda x: x[0])]
    
      sorted_vocab = [k for (k,v) in sorted(next_word_distr.items(), key = lambda x: x[0])]

    df_probabilities = pd.DataFrame(distrs.values())

    df_probabilities_mean = df_probabilities.mean()

    avg_distr = dict(zip(sorted_vocab, df_probabilities_mean))

    prob_list_sum = sum(df_probabilities_mean)
    prob_list = [v/prob_list_sum for (k, v) in avg_distr.items()]

    exponentiated_scores = torch.tensor(softmax(prob_list))
    n = list(torch.multinomial(exponentiated_scores, top_k))

    resulting_words = [sorted_vocab[i] for i in n]

    return resulting_words

def sample_random_words(vocab, top_k):

  N = len(vocab)
  resulting_words = [vocab[random.randint(1, N-1)] for x in range(top_k)]

  return resulting_words

def checking_tokens(context, predicted_tokens, want_prefix, prefix):

  final_tokens = []
  for token in predicted_tokens:
    if token not in string.punctuation and token not in context:
      if (want_prefix and token[0:2] == prefix) or (not want_prefix and token[0:2]!= prefix):
        final_tokens.append(token)
  return final_tokens    

def sample_bert(context, change_i, num_masks, top_k, replacement):

  new_context = copy.copy(context)
  if replacement:
    new_context[change_i] = '[MASK]'
    if num_masks == 2:
      new_context.insert(change_i+1,'[MASK]')
  else:
    new_context.insert(change_i+1,'[MASK]')
    if num_masks == 2:
      new_context.insert(change_i+2,'[MASK]')

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True)

  inputs = tokenizer(" ".join(new_context), return_tensors='pt')
  outputs = model(**inputs)
  predictions = outputs[0]

  predicted_indices = torch.topk(predictions[0, change_i], top_k).indices 
  predicted_tokens = tokenizer.convert_ids_to_tokens([predicted_indices[x] for x in range(top_k)])

  final_tokens = checking_tokens(context, predicted_tokens, False, "##")

  if num_masks == 2:
    predicted_indices_2 = torch.topk(predictions[0, change_i+1], top_k*10).indices 
    predicted_tokens_2 = tokenizer.convert_ids_to_tokens([predicted_indices_2[x] for x in range(top_k)])
    final_tokens_2 = checking_tokens(context, predicted_tokens_2, True, "##")

    if len(final_tokens_2) == 0:
      final_tokens = final_tokens
    elif len(final_tokens) > len(final_tokens_2) and len(final_tokens_2) != 0:
      final_tokens = final_tokens[0:len(final_tokens_2)]
      final_tokens = list(zip(final_tokens, final_tokens_2))
    elif len(final_tokens) < len(final_tokens_2):
      final_tokens_2 = final_tokens_2[0:len(final_tokens)]
      final_tokens = list(zip(final_tokens, final_tokens_2))

  print("final tokens", final_tokens)

  # making sure it doesn't include punctuation or repeats

  return final_tokens


def checking_tokens_pos(context, predicted_tokens, want_prefix, prefix, og_word, pos_dict):

  final_tokens = []
  for token in predicted_tokens:
    if token not in string.punctuation and token not in context:
      if (want_prefix and token[0:2] == prefix) or (not want_prefix and token[0:2]!= prefix):
        og_word_pos = pos_dict[og_word]
        token_pos = pos_dict[token]
        if token_pos == og_word_pos:
          final_tokens.append(token)
  return final_tokens   

def sample_bert_pos(context, change_i, num_masks, top_k, replacement, pos_dict):

  new_context = copy.copy(context)

  og_word = new_context[change_i]
  if replacement:
    new_context[change_i] = '[MASK]'
    if num_masks == 2:
      new_context.insert(change_i+1,'[MASK]')
  else:
    new_context.insert(change_i+1,'[MASK]')
    if num_masks == 2:
      new_context.insert(change_i+2,'[MASK]')

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True)

  inputs = tokenizer(" ".join(new_context), return_tensors='pt')
  outputs = model(**inputs)
  predictions = outputs[0]

  predicted_indices = torch.topk(predictions[0, change_i], top_k).indices 
  predicted_tokens = tokenizer.convert_ids_to_tokens([predicted_indices[x] for x in range(top_k)])

  final_tokens = checking_tokens_pos(context, predicted_tokens, False, "##")

  if num_masks == 2:
    predicted_indices_2 = torch.topk(predictions[0, change_i+1], top_k*10).indices 
    predicted_tokens_2 = tokenizer.convert_ids_to_tokens([predicted_indices_2[x] for x in range(top_k)])
    final_tokens_2 = checking_tokens_pos(context, predicted_tokens_2, True, "##")

    if len(final_tokens_2) == 0:
      final_tokens = final_tokens
    elif len(final_tokens) > len(final_tokens_2) and len(final_tokens_2) != 0:
      final_tokens = final_tokens[0:len(final_tokens_2)]
      final_tokens = list(zip(final_tokens, final_tokens_2))
    elif len(final_tokens) < len(final_tokens_2):
      final_tokens_2 = final_tokens_2[0:len(final_tokens)]
      final_tokens = list(zip(final_tokens, final_tokens_2))

  print("final tokens", final_tokens)

  # making sure it doesn't include punctuation or repeats or mismatched POS

  return final_tokens

def discounting(cur_ind, js_positions, gamma=1):

  total = 0
  to_consider = len(js_positions) - cur_ind
  for i in range(to_consider):
    total += js_positions[cur_ind+i]*(gamma**i)

  length_js_pos = 1 if to_consider == 0 else to_consider
  return total/length_js_pos

def plot_scores(scores, sentence):

  plt.plot(range(len(scores)),scores)
  plt.xlabel("Iterations")
  plt.ylabel("Jensen Shannon Scores")
  plt.title("GPT2-Roberta-Albert-XLM J-S Scores for " + sentence)
  plt.show()
  name = sentence + " sampling.png"
  plt.savefig(name)
  plt.close()

def plot_positions(js_positions, sentence):


  for pos in js_positions:
    plt.plot(pos)
  ticks = sentence.split(" ")
  plt.xticks(np.arange(len(ticks)), ticks)
  plt.xlabel("Words in Sentence")
  plt.ylabel("Jensen Shannon Scores")
  plt.title("GPT2-Roberta-Albert-XLM J-S Scores for")
  plt.show()
  name = sentence + " positions sampling.png"
  plt.savefig(name)
  plt.close()

def change_sentence(sentence, evaluate_sentence, sampler, **kwargs):

  changes = []
  # exclude final punctuation
  sentence_split = sentence.split(" ")[:-1]
  len_sentence = len(sentence_split)

  curr_score = evaluate_sentence(model_list, " ".join(sentence_split), vocab, batch_size, prev_dict)
    
  scores = [curr_score]

  print("OG sentence is: ", sentence, " with JS: ", curr_score)

  max_len = max(max_length, len_sentence + 3)

  while len(sentence_split) <= max_len:

    change_i = random.randint(0, len(sentence_split)-1)

    print("change i", change_i)

    modified_sentence_replacements = copy.deepcopy(sentence_split)
    modified_sentence_deletions = copy.deepcopy(sentence_split)
    modified_sentence_additions = copy.deepcopy(sentence_split)

    new_sentence_list = []

    # replacements 
    num_masks = random.randint(1,2)
    replacement = True
    context = sentence_split

    bert_args = (context,change_i, num_masks, top_k, replacement)
    bert_pos_args = (context,change_i, num_masks, top_k, replacement, pos_dict)
    rw_args = (vocab, top_k)
    ad_args = (model_list, context, vocab, batch_size, top_k)

    sampler_dict = {sample_bert: bert_args, sample_random_words: rw_args, sample_avg_distr: ad_args, sample_bert_pos: bert_pos_args}

    sampler_args = sampler_dict[sampler]

    new_word_list = sampler(*sampler_args)
    #sample_random_words(vocab, top_k)
    #sample_bert(sentence_split, change_i, num_masks, 50, True)
    #sample_avg_distr(model_list, sentence_split, vocab, batch_size, top_k)

    i = len(new_word_list)
    for words in new_word_list: 
      i -= 1
      if isinstance(words, tuple):
        modified_sentence_replacements[change_i] = str(words[0]) + str(words[1])[2:]
      else:
         modified_sentence_replacements[change_i] = str(words)

      new_context = ' '.join(modified_sentence_replacements)
      print("mod sentence replacement", new_context)
      new_sentence_list.append((i,new_context))
      modified_sentence_replacements = copy.copy(sentence_split)

    #deletions
    modified_sentence_deletions.pop(change_i)
    if len(modified_sentence_deletions) > 0:
      new_context = ' '.join(modified_sentence_deletions)
      print("deletion try", new_context)
      new_sentence_list.append((1,new_context))

    # additions
    if len(sentence_split) < max_len:
      num_masks = random.randint(1,2)
      replacement = False
      context = sentence_split

      sampler_args = sampler_dict[sampler]

      new_word_list = sampler(*sampler_args)
      #sample_random_words(vocab, top_k)
      #sample_bert(sentence_split, change_i, num_masks, 50, False)
      #sample_avg_distr(model_list, sentence_split, vocab, batch_size, top_k)

      i = len(new_word_list)
      for words in new_word_list:
        i -= 1
        print("words", words)
        if isinstance(words, tuple):
          modified_sentence_additions.insert(change_i+1,str(words[0]))
          modified_sentence_additions.insert(change_i+2,str(words[1]))
        else:
          modified_sentence_additions.insert(change_i+1,str(words))    

        new_context = ' '.join(modified_sentence_additions)
        print("mod sentence additions", new_context)
        new_sentence_list.append((i,new_context))
        modified_sentence_additions = copy.copy(sentence_split)

    exponentiated_scores = torch.tensor(softmax([i[0] for i in new_sentence_list]))
    n = list(torch.multinomial(exponentiated_scores, 1))
    sampled_id = n[0]
    final_modified_sentence = new_sentence_list[sampled_id][1]

    new_sentence_score = evaluate_sentence(model_list, final_modified_sentence, vocab, batch_size, prev_dict)

    if new_sentence_score > curr_score:
      print("new score", new_sentence_score, "curr_score", curr_score)
      print("Here is the new version of the sentence: ", ' '.join(sentence_split))
      sentence_split = final_modified_sentence.split(" ")
      curr_score = new_sentence_score
      changes.append((curr_score, final_modified_sentence))

    scores.append(curr_score)
    if len(scores) > convergence_criterion:
      last_N_scores = scores[-convergence_criterion:] 
      if len(set(last_N_scores)) == 1:

        print("New sentence is: ", ' '.join(sentence_split)," with total scores: ", scores, "and changes", changes)

        plot_scores(scores, ' '.join(sentence_split))

        print(curr_score)

        return scores, ' '.join(sentence_split)

def sample_sentences(file_name):

  file = open(file_name)
  reader = csv.reader(file)
  num_lines = len(list(reader))
  N = random.randint(0,num_lines-1)
  with open(file_name, 'r') as file:
      reader = csv.reader(file)
      line = next((x for i, x in enumerate(reader) if i == N), None)
      line = (" ".join(line)).translate(str.maketrans('', '', string.punctuation))

  return line


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

if __name__ == "__main__":

  sentences = [sample_sentences("sentences4lara.txt") for i in range(5)]

  sent_dict = dict(zip([str(x) for x in range(1,5)], sentences))

  sentence = sent_dict[sys.argv[2]]

  batch_size = 100
  convergence_criterion = int(sys.argv[4])
  model_list = [GPT2, Roberta, Albert, XLM, T5] 
  max_length = 8
  top_k = 50
  prev_dict = {}
  evaluate_sentence = evaluate_sentence_jsd
  sampler_dict = {"sample_bert": sample_bert, "sample_random_words": sample_random_words, "sample_bert_pos": sample_bert_pos, "sample_avg_distr": sample_avg_distr}

  sampler = sampler_dict[sys.argv[3]]

  kwargs = {"vocab": vocab, "pos_dict": pos_dict, "batch_size": batch_size, "convergence_criterion": convergence_criterion, "model_list": model_list, "prev_dict": prev_dict, "max_length": max_length, "top_k": top_k}

  globals()[sys.argv[1]](sentence, evaluate_sentence, sampler, **kwargs)
