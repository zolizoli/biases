import math
import codecs
import numpy as np
import pandas as pd
from collections import Counter
from nltk.util import skipgrams

#############################################################################
#####              Building a PMI matrix from a corpus                  #####
#############################################################################
# Based on http://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/

#skipgrams(sent, n, k) where n- degree of ngrams, k - skip distance
txt = codecs.open('data/processed/song_corpus/processed_songs.txt', 'r',
                  'utf-8').read().split()

# unigrams
unigram_freqs = Counter(txt)
uni_total = sum(unigram_freqs.values())
unigram_probs = {}
for k in unigram_freqs.keys():
    unigram_probs[k] = unigram_freqs[k] / uni_total

# skipgrams
# this turns an iterable into a list, so don't call it on huge corpora
t = list(skipgrams(txt, 2, 10))
t_freqs = Counter(t)
skip_total = sum(t_freqs.values())
skip_probs = {}
for k in t_freqs.keys():
    skip_probs[k] = t_freqs[k] / skip_total

# normalized skipgrams
normalized_skipgram_probs = {}
for k in skip_probs.keys():
    a = k[0]
    b = k[1]
    pa = unigram_probs[a]
    pb = unigram_probs[b]
    pab = skip_probs[k]
    nsp = math.log2(pab / pa / pb)
    normalized_skipgram_probs[k] = nsp

# PMI matrix
of = codecs.open('models/pmi/song_matrix.tsv', 'w', 'utf-8')
column_names = sorted(unigram_probs.keys())
h = ""
for w in column_names:
    h += w + '\t'
h = h.strip() + '\n'
of.write(h)
row_names = sorted(unigram_probs.keys())
for word in column_names:
    o = ''
    for w2 in row_names:
        k = (word, w2)
        if k in skip_probs.keys():
            pmi = skip_probs[k]
        else:
            pmi = 0.0
        o += str(pmi) + '\t'
    o = o.strip() + '\n'
    of.write(o)

df = pd.read_csv('models/pmi/song_matrix.tsv',
                 encoding='utf-8', sep='\t')
pmi_matrix = df.as_matrix()
np.save('models/pmi/song_matrix.npy', pmi_matrix)
