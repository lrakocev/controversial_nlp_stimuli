import nltk
nltk.download('punkt')
nltk.download('gutenberg')
from nltk.corpus import gutenberg

bible=gutenberg.sents('bible-kjv.txt')

sentences = [" ".join(sent) for sent in bible]

print(sentences)