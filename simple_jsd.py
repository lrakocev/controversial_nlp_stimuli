from transformers import pipeline
from scipy.spatial import distance
import judge_sentences as j_s

def get_probabilities(nlp, sentence):

	sentence = sentence.split(" ")
	sentences = []
	for i in range(len(sentence)):
		cur_sentence = sentence.copy()
		cur_sentence[i] = {nlp.tokenizer.mask_token}
		cur_sentence.join(" ")
		sentences.append(cur_sentence)

	scores = []
	for i in range(len(sentences)):
		sentence = sentences[i]
		target = " " + sentence[i]
		score = nlp(sentence, targets=[target]).score
		scores.append([score, 1-score])

	return scores


def evaluate_sentence(scores1, scores2):

	js_scores = []
	for i in range(len(scores1)):
		js = distance.jensenshannon(scores1[i], scores2[i])
		js_scores.append(js)

	return sum(js_scores)/len(js_scores)


sentences = sorted(j_s.sample_sentences("sentences4lara.txt", 100))

nlp_roberta = pipeline("fill-mask", model="roberta-base")
nlp_gpt2 = pipeline("fill-mask", model="gpt2")


final_jsd_scores = {}

for i in range(len(sentences)):
	sentence = sentences[i]
	scores1 = get_probabilities(nlp_roberta, sentence)
	scores2 = get_probabilities(nlp_gpt2, sentence)

	jsd = evaluate_sentence(scores1, scores2)
	final_jsd_scores[sentence] = jsd

print(final_jsd_scores)