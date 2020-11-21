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
		sent = sentences[i]
		target = " " + sentence[i]
<<<<<<< HEAD
		
		score = nlp(sentence, targets=[target])[0]['score']
=======
		print(sentence)
		
		print(nlp(sent,targets=[target]))
		score = nlp(sent, targets=[target])[0]['score']
>>>>>>> 71fa2ec62ebadcc0f12a80d5852cc6b224f5d4db
		scores.append([score, 1-score])

	return scores

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
