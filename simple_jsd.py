from transformers import pipeline
from scipy.spatial import distance

def get_probabilities(nlp, sentence):

	sentence = sentence.split(" ")
	sentences = []
	for i in range(len(sentence)):
		cur_sentence = sentence.copy()
		cur_sentence[i] = nlp.tokenizer.mask_token
		cur_sentence = " ".join(cur_sentence)
		sentences.append(cur_sentence)

	scores = []
	for i in range(len(sentences)):
		sentence = sentences[i]
		target = " " + sentence[i]
		score = nlp(sentence, targets=[target]).score
		scores.append([score, 1-score])

	return scores

def alternative_get_probabilities(sentence, tokenizer, model):


    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss = model(tensor_input, labels=tensor_input)
    return -loss[0].item()


def evaluate_sentence(scores1, scores2):

	js_scores = []
	for i in range(len(scores1)):
		js = distance.jensenshannon(scores1[i], scores2[i])
		js_scores.append(js)

	return sum(js_scores)/len(js_scores)


def sample_sentences(file_name, n):

  with open(file_name) as f:
    head = [next(f).strip() for x in range(n)]

  return head


sentences = sorted(sample_sentences("sentences4lara.txt", 100))

nlp_roberta = pipeline("fill-mask", model="roberta-base")
nlp_xlm = pipeline("fill-mask", model="xlm-mlm-xnli15-1024")


final_jsd_scores = {}

for i in range(len(sentences)):
	sentence = sentences[i]
	scores1 = get_probabilities(nlp_roberta, sentence)
	scores2 = get_probabilities(nlp_xlm, sentence)

	jsd = evaluate_sentence(scores1, scores2)
	final_jsd_scores[sentence] = jsd

print(final_jsd_scores)