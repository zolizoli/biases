import numpy as np
from string import punctuation
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

gmodel = 'models/song_corpus/song.model'
bmodel = 'models/song_corpus/song.bin'

model =KeyedVectors.load_word2vec_format(bmodel, binary=True)

#TODO: printeljük ki magunknak az eredményt, mentsük ki egy fájlba
m_acc = model.accuracy('data/external/questions-words-hu.txt')

punctuation += ''.join([str(x) for x in range(10)])

#TODO: jobb, más filterezés?
def word_filter(w):
    return (w.lower() == w
            and len(w) < 20
            and np.all([x not in punctuation for x in w])
            )

best_words = [
    (x, model[x]/np.linalg.norm(model[x]))
    for x in model.vocab.keys()
    if model.vocab[x].index < 3000
    and word_filter(x)]

# ezek majd a gendered pairs részhez kellenek
best_words_array = [t[1] for t in best_words]
words = [t[0] for t in best_words]


## finding analogies
#TODO: találjunk ki saját analógiákat
analogy = model.most_similar(positive=['lány_NOUN', 'szép_ADJ'],
                             negative=['fiú_NOUN'])

a = model['fiú_NOUN']
b = model['lány_NOUN']
ab = a-b

# ez lassú, egy ebéd simán belefér, amíg kigyűjti
DELTA = 1
gendered_pairs = []
for i, x in enumerate(best_words_array):
    dist_xy = x - best_words_array[0:]
    norms = np.linalg.norm(dist_xy, axis=1)
    mask = norms > DELTA
    S = np.dot(dist_xy, ab)
    S = np.ma.masked_array(S, mask=mask)
    mx = S.max()
    if mx > 0:
        j = np.where(S == mx)[0][0]
        gendered_pairs.append((words[i], words[j], S[j]))

gendered_pairs.sort(key=lambda s: s[2], reverse=True)
