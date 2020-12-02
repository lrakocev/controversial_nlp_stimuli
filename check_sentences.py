from nltk.corpus import gutenberg

def sentences_from_corpus(corpus, fileids = None):

    from nltk.corpus.reader.plaintext import read_blankline_block, concat

    def read_sent_block(stream):
        sents = []
        for para in corpus._para_block_reader(stream):
            sents.extend([s.replace('\n', ' ') for s in corpus._sent_tokenizer.tokenize(para)])
        return sents

    return concat([corpus.CorpusView(path, read_sent_block, encoding=enc)
                   for (path, enc, fileid)
                   in corpus.abspaths(fileids, True, True)])


bible=gutenberg.sents('bible-kjv.txt')

sentences = sentences_from_corpus(bible)

print(sentences)