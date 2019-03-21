import codecs
from hunlp import HuNlp
from os import listdir
from os.path import join, isfile

in_path = 'data/raw/song_corpus'
txts = [join(in_path, f) for f in listdir(in_path)
        if isfile (join(in_path,f))]

out_path = 'data/processed/song_corpus'
nlp = HuNlp()

for txt in txts:
    print(txt)
    lyrics = codecs.open(txt, 'r', 'utf-8').read()
    l = lyrics.strip()
    # correcting common character encoding mistakes
    l = l.replace('\n', '. ')
    l = l.replace('ä', 'a')
    l = l.replace('ô', 'ő')
    l = l.replace('õ', 'ö')
    l = l.replace('û', 'ű')
    l = l.replace('r.', '')
    lemmatized_text = []

    doc = nlp(l)
    for sent in doc:
        for tok in sent:
            lemma = tok.lemma
            pos = tok.tag
            #TODO: vannak-e még más pos tagek, amik ugyanolyan
            #  betűvel kezdődnek és meddig fedik egymást?
            if pos.startswith('A'):
                pos = pos[:3]
            else:
                pos = pos[0]
            if lemma.isalpha():
                tagged_word = lemma.strip().lower() + '_' + \
                              pos
                lemmatized_text.append(tagged_word)
    lemmatized_text = ' '.join(lemmatized_text)
    with codecs.open(join(out_path, txt.split('/')[-1]), 'w',
                     'utf-8') as out_file:
        out_file.write(lemmatized_text)
        print('Done with', txt)
