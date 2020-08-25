from nlp_adversarial_examples import *
import numpy as np
from scipy.stats import entropy
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, TransfoXLTokenizer, TFTransfoXLLMHeadModel
import sys

def get_distribution(model_info, model_name, context, num_return_seqs, current_len):

        model, tokenizer = model_info[model_name]
        # encode context the generation is conditioned on
        input_ids = tokenizer.encode(context, return_tensors='tf')

        num_tokens = input_ids.shape[1]
        #max_length=num_tokens + 1,
        sample_outputs = model.generate(
                input_ids,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                max_length = num_tokens + 1,
                no_repeat_ngram_size=2,
                num_return_sequences=num_return_seqs
        )

        distribution_dict = {}
        for i, sample_output in enumerate(sample_outputs):
                output = tokenizer.decode(sample_output, skip_special_tokens=True)
                key = output.replace(context,'')
                distribution_dict[key] = distribution_dict.get(key, 0) + 1
        distribution_dict = {k: v/num_return_seqs for k,v in distribution_dict.items()}

        return distribution_dict



def js(p, q):

    intersection = p.keys() & q.keys()
    p = {k:v for k,v in p.items() if k in intersection}
    q = {k:v for k,v in q.items() if k in intersection}

    p = [v for k, v in sorted(p.items())]
    q = [v for k, v in sorted(q.items())]

    p = np.asarray(p)
    q = np.asarray(q)
    # normalize
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2



def auto_regressive(model_info, curr_context, num_return_seqs, current_len, max_len, total_js):

    if current_len == max_len:
        return total_js / current_len

    distrs = {}
    highest = {}
    for model_name in ['GPT2','TransformerXL']:
        model, tokenizer = model_info[model_name]
        next_word_distr = get_distribution(model_info, model_name, curr_context, num_return_seqs,current_len)
        distrs[model_name] = next_word_distr

    A = distrs['GPT2']
    B = distrs['TransformerXL']
    # average the two distributions
    avg_distr = {x: (A.get(x, 0) + B.get(x, 0))/2 for x in set(A).intersection(B)}
    highest = sorted(avg_distr.items(), key=lambda x: x[1], reverse=True)
    print(highest)
    js_dict = {}
    for i in range(0,1):
        new_word = highest[i][0]
        print("NEW WORD", new_word)
        print("CURR PRE-NEW CONTEXT", curr_context)
        new_context =  curr_context + new_word
        print("NEW CONTEXT", new_context)
        p = get_distribution(model_info, 'GPT2', new_context, num_return_seqs, current_len)
        q = get_distribution(model_info,'TransformerXL', new_context, num_return_seqs, current_len)
        print(p,q)
        js_result = js(p,q)
        js_dict[new_word] = js_result


    #print(js_dict)
    #print("JS List", sorted(js_dict.items(), key=lambda x: x[1]))
    highest_js_word = sorted(js_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
print("highest JS word", highest_js_word)
    curr_context = curr_context + highest_js_word

    p = get_distribution(model_info,'GPT2', curr_context, num_return_seqs, current_len + 1)
    q = get_distribution(model_info,'TransformerXL', curr_context, num_return_seqs, current_len + 1)

    total_js += js(p,q)
    print("CURR CONTEXT", curr_context, "JS", total_js)
    # then autoregressive again on this new current context
    return auto_regressive(model_info, curr_context, num_return_seqs, current_len + 1, max_len , total_js)


model_info = {"GPT2": (TFGPT2LMHeadModel.from_pretrained("gpt2"),GPT2Tokenizer.from_pretrained("gpt2")), "TransformerXL": (TFTransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103'),TransfoXLTokenizer.from_pretrained('transfo-xl-wt103'))}
curr_context = sys.argv[1:]
curr_context = ' '.join(curr_context)
auto_regressive(model_info, curr_context, 50, 1, 15, 0)
                  
