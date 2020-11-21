import judge_sentences as j_s
import sys
import numpy as np

filename = "SUBTLEXus74286wordstextversion.txt"
vocab = j_s.get_vocab(filename, 10000)

model_list = [j_s.GPT2, j_s.Roberta] 
n = 100

def evaluate_sentence(sentence, n):

  sentence_split = sentence.split(" ")
  len_sentence = len(sentence_split)

  curr_context = ""
  total_js = 0
  js_positions = []
  distrs = {}

  for i in range(0, len_sentence):
    curr_context += sentence_split[i] + " "
    
    for model_name in model_list:
      next_word_distr = j_s.get_distribution(model_name, curr_context, vocab, n)
      distrs[model_name] = list(next_word_distr.values())

    n = len(model_list)
    weights = np.empty(n)
    weights.fill(1/n)

    curr_js = j_s.jsd(list(distrs.values()), weights)
    total_js += curr_js
    js_positions.append(curr_js)

  print(total_js/len_sentence)
  return total_js/len_sentence

if __name__ == "__main__":

  sentences = sorted(j_s.sample_sentences("sentences4lara.txt", 100))

  sent_dict = dict(zip([str(x) for x in range(1,100)], sentences))

  sentence = sent_dict[sys.argv[2]]

  globals()[sys.argv[1]](sentence, 100)

