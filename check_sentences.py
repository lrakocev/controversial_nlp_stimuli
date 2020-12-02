import nltk
nltk.download('punkt')
nltk.download('gutenberg')
from nltk.corpus import gutenberg

alice = gutenberg.sents('carroll-alice.txt')

sentences = [" ".join(sent) for sent in alice]

N = 500

for i in range(N):

	print(random.choice(sentences))