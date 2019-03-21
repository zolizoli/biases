import codecs
import gensim
import logging


def chunkify(lst,n):
    return [lst[i::n] for i in range(n)]

corpus = codecs.open('data/processed/song_corpus/processed_songs.txt',
                     'r', 'utf-8').read().split()

#TODO: ha sok szövegen (fájl, vagy itt pl lista) kell trénelni, akkor jobb
#  ha csinálsz egy iterable osztályt amin a gensim dolgozhat, azaz csak
# akkor olvas be egy-egy elemet, amikor kell
class MySentences(object):
    """Makes an iterable object from the corpus"""

    def chunkify(self, lst, n):
        return [lst[i::n] for i in range(n)]

    def __init__(self, corpus):
        self.sents = self.chunkify(corpus, 100)

    def __iter__(self):
        for sent in self.sents:
            yield sent

sentences = MySentences(corpus)
model = gensim.models.Word2Vec(sentences, min_count=10, size=500, workers=4,
                               window=10, sample=1e-3)
model.save('models/song_corpus/song.model')
model.wv.save_word2vec_format('models/song_corpus/song.bin', binary=True)
