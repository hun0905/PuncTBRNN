import string
import re
from os import listdir
from collections import Counter
from typing import Text
from nltk.corpus import stopwords

def load_doc(filename):
    file = open(filename,'r',encoding='utf-8',errors='ignore')
    text =file.read()
    file.close()
    return text

def clean_doc(doc):
    tokens = doc.split()
    re_punc = re.compile('[%s]'% re.escape(string.punctuation))
    token = [re_punc.sub('',w) for w in tokens] 
    stop_words = open('/home/yunghuan/NLP_Dataset/Chinese/stop_word2.txt',encoding='utf-8',errors='ignore').readlines()
    stop_words = set(stop_words)
    tokens =[w for w in tokens if not w in stop_words]
    tokens = [word for word in token if len(word) > 1]
    return tokens

def add_doc_to_vocab(filename,vocab):
    doc= load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)

def save_list(lines,filename):
    data = '\n'.join(lines)
    file = open(filename,'w')
    file.write(data)
    file.close()
vocab = Counter()
add_doc_to_vocab('/home/yunghuan/NLP_Dataset/Chinese/data_Ch/train.txt',vocab)
min_occurence = 5
tokens = [k for k,c in vocab.items() if c>= min_occurence]
save_list(tokens,'vocab.txt')