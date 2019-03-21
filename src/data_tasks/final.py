import luigi
import zipfile
import gzip
import codecs
import subprocess
import urllib.request
from os import mkdir, remove, listdir
from os.path import exists, join, isfile
from bs4 import BeautifulSoup
from langdetect import detect_langs


class MakeStructure(luigi.Task):
    """Makes data folder and downloads necessary data"""
    def requires(self):
        pass

    def output(self):
        pass

    def run(self):
        # make sure we have a data folder with the right structure
        if not exists('data/'):
            mkdir('data/')
            mkdir('data/external')
            mkdir('data/external/meta')
            mkdir('data/interim')
            mkdir('data/interim/processed_songs')
            mkdir('data/processed')
            mkdir('data/raw')
            mkdir('data/raw/song_corpus')
        if not exists('etc/'):
            mkdir('etc/')
        if not exists('etc/magyarlanc-3.0.jar'):
            url = 'http://rgai.inf.u-szeged.hu/project/nlp/research/magyarlanc/magyarlanc-3.0.jar'
            urllib.request.urlretrieve(url, 'etc/magyarlanc-3.0.jar')
        # check if we have all the necessary raw data
        # download pre-trained w2v models
        if not exists('data/raw/hu.bin'):
            url = "https://drive.google.com/uc?export=download&confirm=BI6L&id=0B0ZXk88koS2KX2xLamRlRDJ3N1U"
            urllib.request.urlretrieve(url, 'data/raw/hu.zip')
            with zipfile.ZipFile('data/raw/hu.zip', 'r') as zip_ref:
                zip_ref.extractall('data/raw')
        if not exists('data/raw/word2vec-mnsz2-webcorp_600_w10_n5_i1_m10.w2v'):
            url = 'http://corpus.nytud.hu/efnilex-vect/data/hunembed0.0'
            urllib.request.urlretrieve(url, 'data/raw/hunembed.gzip')
            with gzip.open('data/raw/hunembed.gzip', 'rb') as f:
                file_content = f.read()
            with open('data/raw/hunembed.w2v', 'w') as outfile:
                outfile.write(file_content)


class CollectLyricsData(luigi.Task):
    """Collects lyrics from zeneszoveg.hu"""
    c = 1

    def input(self):
        return 'data/external/meta/bands.csv'

    def meta(self):
        return luigi.LocalTarget('data/external/meta/meta.tsv')

    def run(self):
        songbase = 'http://www.zeneszoveg.hu/'
        with open(self.input(), 'r') as f:
            with self.meta().open('w') as metainfo:
                for l in f:
                    l = l.split(';')
                    if len(l) == 2:
                        band, url = l
                        band = band.strip().replace('\t', ' ')
                        try:
                            html = urllib.request.urlopen(url).read()
                            soup = BeautifulSoup(html, "lxml")
                            songlist = soup.find("div", "artistRelatedList")
                            songlist = ''.join(map(str, songlist.contents))
                            songsoup = BeautifulSoup(songlist, "lxml")
                            songs = songsoup.find_all('a')
                            for e in songs:
                                itemdict = e.attrs
                                if 'href' in itemdict and 'title' in itemdict:
                                    link = itemdict['href']
                                    if link.startswith('dalszoveg'):
                                        link = songbase + link
                                        title = itemdict['title']
                                        title = title.split(' - ')
                                        if len(title) == 1:
                                            title = title[0].strip().replace('\t',
                                                                             ' ')
                                        else:
                                            title = title[1].strip().replace('\t',
                                                                             ' ')
                                        with urllib.request.urlopen(
                                                link) as response:
                                            songhtml = response.read()
                                            lyricssoup = BeautifulSoup(songhtml,
                                                                       'lxml')
                                            lyrics = lyricssoup.find('div',
                                                                     'lyrics-plain-text')
                                            lyrics = ''.join(
                                                map(str, lyrics.contents))
                                            lyrics = lyrics.replace('<br />', '')
                                            lyrics = lyrics.replace('<br/>', '')
                                            if 'title="Még nincs megadva. Te tudod?' \
                                               ' Töltsd ki!">Keressük a dalszöveget!' \
                                                    not in lyrics:
                                                link = link.strip()
                                                # output
                                                with open(
                                                        'data/raw/song_corpus/' + str(self.c).zfill(4) + '.txt',
                                                        'w') as out_file:
                                                    out_file.write(lyrics)
                                                o = str(self.c).zfill(
                                                    4) + '\t' + title + '\t' + \
                                                    band + '\t' + link + '\n'
                                                metainfo.write(o)
                                                self.c += 1
                        except Exception as e:
                            print(str(e)*450)
                            continue


class FilterSongs(luigi.Task):
    """Deletes non-Hungarian lyrics
    """
    def input(self):
        in_path = 'data/raw/song_corpus'
        return [join(in_path, f) for f in listdir(in_path) if isfile(join(
            in_path, f))]

    def requires(self):
        if len(self.input()) == 0:
            return [CollectLyricsData()]
        else:
            pass

    def run(self):
        songs = self.input()
        for song in songs:
            txt = codecs.open(song, 'r', 'utf-8').read()
            if 'Még nincs megadva. Te tudod? Töltsd ki!' not in txt:
                try:
                    langs = detect_langs(txt)
                    langs = [e.lang for e in langs]
                    if 'hu' not in langs:
                        remove(song)
                except Exception as e:
                    continue


class PreprocessSongs(luigi.Task):
    """Make an all lowercase, lemma-only version from lyrics
       Start hunlp before you call this task
    """

    def input(self):
        in_path = 'data/raw/song_corpus'
        return [join(in_path, f) for f in listdir(in_path) if isfile(join(
            in_path, f))]

    def output(self):
        return luigi.LocalTarget(
            'data/interim/processed_songs/chained_songs.txt')

    def requires(self):
        if len(self.input()) == 0:
            return [FilterSongs()]
        else:
            pass

    def run(self):
        txts = self.input()
        songs = []
        for txt in txts:
            fname = txt.split('/')[-1]
            text = codecs.open(txt, 'r', 'utf-8').read()
            songs.append(text)
        songs = '\n'.join(songs)
        with self.output().open('w') as out_file:
            out_file.write(songs)


class ProcessSongs(luigi.Task):
    """Run magyarlanc on chained texts"""
    def input(self):
        return 'data/interim/processed_songs/chained_songs.txt'

    def outfile(self):
        return 'data/interim/processed_songs/processed_songs.out'

    def requires(self):
        if not exists('data/interim/processed_songs/chained_songs.txt'):
            return [PreprocessSongs()]
        else:
            pass

    def run(self):
        subprocess.call("java -Xmx1G -jar etc/magyarlanc-3.0.jar  -mode "
                        "morphparse -input " + self.input() + " -output "
                        + self.outfile(), stderr=subprocess.STDOUT,
                        shell=True)


class LyricsTask(luigi.Task):
    """Connects the parts of the lyrics pipeline
       Make a nice lemmatized version of the corpus
    """

    def input(self):
        return luigi.LocalTarget(
            'data/interim/processed_songs/processed_songs.out')

    def output(self):
        return luigi.LocalTarget(
            'data/processed/song_corpus/processed_songs.txt')

    def requires(self):
        if not exists('data/interim/processed_songs/processed_songs.out'):
            return [ProcessSongs()]
        else:
            pass

    def run(self):
        with self.input().open('r') as in_file:
            with self.output().open('w') as out_file:
                corp = []
                for l in in_file:
                    l = l.strip().split('\t')
                    l = [e.strip() for e in l]
                    if len(l) == 4:
                        wd, lemma, pos, desc = l[0], l[1], l[2], l[3]
                        if lemma.isalpha():
                            tagged_word = lemma.lower() + '_' + pos
                            corp.append(tagged_word)
                corp = ' '.join(corp)
                out_file.write(corp)


class FinalTask(luigi.Task):
    def requires(self):
        return [MakeStructure(), LyricsTask()]

    def output(self):
        pass

    def run(self):
        pass

