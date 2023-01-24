import pandas as pd
import random
from googletrans import Translator
import googletrans
translator = Translator()
languages = list(googletrans.LANGUAGES.keys())
inFile = "data/train_EN.tsv"
outFile = "data/train_cross_switch.p"

def data4transation(FNAME):

    data = pd.read_csv(FNAME, sep='\t', error_bad_lines=False).fillna(method="ffill")
    #data = data.sample(3) 

    sentences = []
    tags = []
    labels = []

    for index, row in data.iterrows():
        sentences.append(row['utterance'].split())
        tags.append(row['slot_labels'].split())
        labels.append(row['intent'])

    sentencess4test = [
        (sent, labs, slabs) for sent, labs, slabs in zip(sentences, tags, labels)
    ]

    return sentencess4test

def isSameLabel(label1, label2):
    return label1[1:] == label2[1:]

def translate(words):
    phrase = " ".join(words)
    tindex = random.randint(0, int(1.5*len(languages)))
    translated_phrase = phrase
    if tindex < len(languages):
        lang = languages[tindex]
        translated_phrase = translator.translate(phrase, dest=lang).text
    translated_phrase = translated_phrase.replace(',', ' ')
    return translated_phrase.split()

def translate_sentence(triple):
    words = triple[0]
    labels = triple[1]
    sentence_label = triple[2]

    translated_words = []
    translated_labels = []

    group = []
    start = 0
    current_label = labels[start]
    group.append(words[start])
    start +=1
    while start != len(words):
        if isSameLabel(current_label, labels[start]):
            group.append(words[start])
            start +=1
        else:
            tw = translate(group)
            if len(tw) > 0:
                translated_words.extend(tw)
                if 'O' in current_label:
                    translated_labels.extend(['O']*len(tw))
                else:
                    for i in range(len(tw)):
                        if i == 0:
                            translated_labels.append('B'+current_label[1:])
                        else:
                            translated_labels.append('I'+current_label[1:])
            group = []
            start +=1
            if start == len(words):
                break
            current_label = labels[start]
            group.append(words[start])

    return translated_words, translated_labels, sentence_label
from tqdm import tqdm
def translate_all(triple, k):
  triples = []
  for t in tqdm(triple):
    for index in range(k):
      triple = translate_sentence(t)
      if len(triple[0]) > 0:
        triples.append(triple)
  return triples

import pickle
x = data4transation(inFile)
y = translate_all(x, 5)
x.extend(y)
pickle.dump(x, open(outFile, 'wb'))
