from nlp_adversarial_examples import *
import numpy as np
from scipy.stats import entropy
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, TransfoXLTokenizer, TFTransfoXLLMHeadModel
import sys

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

def auto_regressive(model_info, curr_context, num_return_seqs, current_len, max_len, total_js, joint_vocab):

    if current_len == max_len:
        return total_js / current_len

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
    
    js_dict = {}
    for i in range(0,5):
        n = random.randint(0,len(avg_distr))
        new_word = list(avg_distr.items())[n][0]
        print("NEW WORD", new_word)
        print("CURR PRE-NEW CONTEXT", curr_context)
        new_context =  curr_context + " " + new_word
        print("NEW CONTEXT", new_context)
        p = get_distribution(model_info, 'GPT2', new_context, joint_vocab)
        q = get_distribution(model_info,'TransformerXL', new_context, joint_vocab)
        print(p,q)
        js_result = js(p,q)
        js_dict[new_word] = js_result


    highest_js_word = sorted(js_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
    print("highest JS word", highest_js_word)
    curr_context = curr_context + " " + highest_js_word

    p = get_distribution(model_info,'GPT2', curr_context, joint_vocab)
    q = get_distribution(model_info,'TransformerXL', curr_context, joint_vocab)

    total_js += js(p,q)
    print("CURR CONTEXT", curr_context, "JS", total_js)
    # then autoregressive again on this new current context
    return auto_regressive(model_info, curr_context, num_return_seqs, current_len + 1, max_len , total_js, joint_vocab)


model_info = {"GPT2": (TFGPT2LMHeadModel.from_pretrained("gpt2"),GPT2Tokenizer.from_pretrained("gpt2")), 
              "TransformerXL": (TFTransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103'),TransfoXLTokenizer.from_pretrained('transfo-xl-wt103'))}

curr_context = "What"
gpt2_dict = get_distribution(model_info, "GPT2", curr_context, {})
txl_dict = get_distribution(model_info, "TransformerXL", curr_context, {})

joint_vocab = gpt2_dict.keys() & txl_dict.keys()

auto_regressive(model_info, curr_context, 50, 1, 15, 0, joint_vocab)
                 
