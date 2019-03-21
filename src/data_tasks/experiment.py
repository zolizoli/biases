import numpy as np
from string import punctuation
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

#############################################################################
#####          working with the pre-trained wikipedia dump              #####
#############################################################################
f = 'data/raw/hu.bin'

model = Word2Vec.load(f)
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
    if model.vocab[x].index < 50000
    and word_filter(x)]

# ezek majd a gendered pairs részhez kellenek
best_words_array = [t[1] for t in best_words]
words = [t[0] for t in best_words]


## finding analogies
#TODO: találjunk ki saját analógiákat
analogy = model.most_similar(positive=['anya', 'orvos'],
                             negative=['apa'])

a = model['férfi']
b = model['nő']
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
#TODO: elmenteni az eredményeket
#TODO: tegyük függyvénybe, hogy több párra is meg tudjuk hívni
#TODO: olvassa be egy txt fájlból a párokat, és mentse ki egy txt-be a
# rájuk kapott eredményeket
#############################################################################
#####                         working with hunembed                     #####
#############################################################################
# nyugodtan el lehet menni moziba amíg betölt, 10.1GB, ezért nem éri meg
# betölteni ha 15GB-nál kevesebb memóriád van
f = 'data/raw/word2vec-mnsz2-webcorp_600_w10_n5_i1_m10.w2v'
model = KeyedVectors.load_word2vec_format(f, binary=False)
