import numpy as np
from scipy.stats import entropy
import tensorflow as tf
from transformers import *
# TFGPT2LMHeadModel, GPT2Tokenizer, TransfoXLTokenizer, TFTransfoXLLMHeadModel
import sys
from scipy.special import softmax
import torch
import random
import string
import functools
import math

def get_distribution(model_info, model_name, context, joint_vocab):

        model, tokenizer = model_info[model_name]

        inputs = tokenizer(context,return_tensors='tf')
        outputs = model(inputs)
        
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

def get_avg_distr(model_info, context, joint_vocab, top_p):

    distrs = {}
    for model_name in model_info.keys():
      model, tokenizer = model_info[model_name]

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

def auto_regressive(model_info, curr_context, num_return_seqs, current_len, max_len, total_js, joint_vocab, top_k):

    if current_len == max_len:
        return total_js / len(curr_context.split(" "))

    distrs = {}
    highest = {}
    for model_name in ['GPT2','TransformerXL']:
        model, tokenizer = model_info[model_name]
        next_word_distr = get_distribution(model_info, model_name, curr_context, joint_vocab)
        distrs[model_name] = next_word_distr

    A = distrs['GPT2']
    B = distrs['TransformerXL']
    # average the two distributions
    avg_distr = {x: (A.get(x, 0) + B.get(x, 0))/2 for x in set(A).intersection(B)}
    
    avg_distr = {k: v for (k,v) in avg_distr.items() if v >= top_k}

    total = sum(avg_distr.values())

    avg_distr = {k: v/total for (k,v) in avg_distr.items()}
    
    prob_list = [v for k, v in sorted(avg_distr.items())]
    word_list = [k for k, v in sorted(avg_distr.items())]

    js_dict = {}
    for i in range(0,5):
        n = list(np.random.multinomial(1,prob_list))
        id = n.index(1)
        new_word = word_list[id]
        print("NEW WORD", new_word)
        print("CURR PRE-NEW CONTEXT", curr_context)
        new_context =  curr_context + " " + new_word
        print("NEW CONTEXT", new_context)
        distrs = {}
        for model_name in model_info.keys():
            model, tokenizer = model_info[model_name]

            next_word_distr = get_distribution(model_info, model_name, context, joint_vocab)
            distrs[model_name] = next_word_distr

        js_result = jsd(distrs.values())
        js_dict[new_word] = js_result


    highest_js_word = sorted(js_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
    print("highest JS word", highest_js_word)
    curr_context = curr_context + " " + highest_js_word

    distrs = {}
    for model_name in model_info.keys():
        model, tokenizer = model_info[model_name]

        next_word_distr = get_distribution(model_info, model_name, context, joint_vocab)
        distrs[model_name] = next_word_distr

    total_js += jsd(distrs.values())
    print("CURR CONTEXT", curr_context, "JS", total_js)
    # then autoregressive again on this new current context
    return auto_regressive(model_info, curr_context, num_return_seqs, current_len + 1, max_len , total_js, joint_vocab, top_k)


def auto_regressive_top_p(model_info, curr_context, num_return_seqs, current_len, max_len, total_js, joint_vocab, top_p):

    if current_len == max_len:
        return total_js / len(curr_context.split(" "))

    highest = {}
    
    prob_list, word_list = get_avg_distr(model_info, curr_context, joint_vocab, top_p)

    js_dict = {}
    for i in range(0,5):
        n = list(np.random.multinomial(1,prob_list))
        id = n.index(1)
        new_word = word_list[id]
        print("NEW WORD", new_word)
        print("CURR PRE-NEW CONTEXT", curr_context)
        new_context =  curr_context + " " + new_word
        print("NEW CONTEXT", new_context)

        distrs = {}
        for model_name in model_info.keys():
            model, tokenizer = model_info[model_name]

            next_word_distr = get_distribution(model_info, model_name, context, joint_vocab)
            distrs[model_name] = next_word_distr

        js_result = jsd(distrs.values())
        js_dict[new_word] = js_result


    highest_js_word = sorted(js_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
    print("highest JS word", highest_js_word)
    curr_context = curr_context + " " + highest_js_word

    
    distrs = {}
    for model_name in model_info.keys():
        model, tokenizer = model_info[model_name]

        next_word_distr = get_distribution(model_info, model_name, context, joint_vocab)
        distrs[model_name] = next_word_distr

    total_js += jsd(distrs.values())
    print("CURR CONTEXT", curr_context, "JS", total_js)
    # then autoregressive again on this new current context
    return auto_regressive_top_p(model_info, curr_context, num_return_seqs, current_len + 1, max_len , total_js, joint_vocab, top_p)

model_info = {"GPT2": (AutoModelWithLMHead.from_pretrained('gpt2-large'), AutoTokenizer.from_pretrained('gpt2-large')), 
              "TransformerXL": (TFTransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103'),TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')),
              "T5": (AutoModelWithLMHead.from_pretrained("t5-11b"), AutoTokenizer.from_pretrained("t5-11b")),
              "Roberta": (RobertaModel.from_pretrained('roberta-base'),RobertaTokenizer.from_pretrained('roberta-base')),
              "Albert": (AlbertModel.from_pretrained('albert-base-v2'), AlbertTokenizer.from_pretrained('albert-base-v2')),
              "XLM": ( XLMModel.from_pretrained('xlm-mlm-xnli15-1024'), XLMTokenizer.from_pretrained('xlm-mlm-xnli15-1024'))}

curr_context = sys.argv[1:]
curr_context = ' '.join(curr_context)
for model_name in model_info.keys():
    model, tokenizer = model_info[model_name]

    next_word_distr = get_distribution(model_info, model_name, curr_context, joint_vocab)
    distrs[model_name] = next_word_distr

joint_vocab = set(distrs["GPT2"].keys()).intersection(*distrs.values().keys())

auto_regressive_top_p(model_info, curr_context, 50, 1, 15, 0, joint_vocab, .7)
                 
