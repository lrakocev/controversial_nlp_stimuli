import numpy as np
from scipy.stats import entropy
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, TransfoXLTokenizer, TFTransfoXLLMHeadModel
import sys
from scipy.special import softmax
import torch
import random
import string
from autoregressive import get_distribution, js


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

def replace_words(model_info, sentence, joint_vocab):

    print("Old sentence is: ", sentence, " with JS: ", evaluate_sentence(model_info, sentence, joint_vocab))

    sentence_split = sentence.split(" ")
    modified_sentence = copy.copy(sentence_split)
    len_sentence = len(sentence_split)

    for i in range(0, len_sentence):
      distrs = {}
      for model_name in ['GPT2','TransformerXL']:
          model, tokenizer = model_info[model_name]
          next_word_distr = get_distribution(model_info, model_name, ' '.join(modified_sentence), joint_vocab)
          distrs[model_name] = next_word_distr

      A = distrs['GPT2']
      B = distrs['TransformerXL']
      avg_distr = {x: (A.get(x, 0) + B.get(x, 0))/2 for x in set(A).intersection(B)}
      
      prev_sentence_score = evaluate_sentence(model_info, ' '.join(sentence_split), joint_vocab)
      js_dict = {}
      for j in range(0,5):
          n = random.randint(0,len(avg_distr))
          new_word = list(avg_distr.items())[n][0]
          print("NEW WORD", new_word)
          print("CURR PRE-NEW CONTEXT", curr_context)
          modified_sentence[i] = new_word
          new_context = ' '.join(modified_sentence)
          print("NEW CONTEXT", new_context)
          js_dict[new_word] = evaluate_sentence(model_info, new_context, joint_vocab)
    
      highest_js_word = sorted(js_dict.items(), key=lambda x: x[1], reverse=True)[0]
      modified_sentence[i] = highest_js_word[0]
      new_sentence_score = evaluate_sentence(model_info, ' '.join(modified_sentence), joint_vocab)
      
      if new_sentence_score > prev_sentence_score:
        sentence_split = modified_sentence
      

    print("New sentence is: ", ' '.join(sentence_split)," with JS:", new_sentence_score)

model_info = {"GPT2": (TFGPT2LMHeadModel.from_pretrained("gpt2"),GPT2Tokenizer.from_pretrained("gpt2")), 
              "TransformerXL": (TFTransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103'),TransfoXLTokenizer.from_pretrained('transfo-xl-wt103'))}
curr_context = "I"
gpt2_dict = get_distribution(model_info, "GPT2", curr_context, {})
txl_dict = get_distribution(model_info, "TransformerXL", curr_context, {})

joint_vocab = gpt2_dict.keys() & txl_dict.keys()

replace_words(model_info, "The cat sat in the hat", joint_vocab)
                 

