import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.metrics import accuracy_score
import json
import re
import nltk
from nltk.corpus import stopwords
# import pymorphy2
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from datetime import datetime
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np

metadata = pd.read_csv('../HackatonData/metadata.csv')

metadata.drop(metadata[(metadata.full_text_file.isnull()) | (
    metadata.pmcid.isnull()) | (metadata.url.isnull())].index, inplace=True)
metadata.publish_time = metadata.publish_time.apply(pd.to_datetime)
metadata = metadata[metadata.publish_time > datetime(2000, 1, 1)]


def deleted_symbol(text):
    pattern_end_html = r'</\w*>|<\w*>'
    pattern_start_html = r'<.*>|\n|\r|\r'
    pattern = r"[^а-яА-Яa-zA-Z]"
    text = re.sub(pattern_end_html, ' ', text)
    text = re.sub(pattern_start_html, '', text)

    text = re.sub("[^а-яА-Яa-zA-Z]", " ", text)

    return text


def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    text = deleted_symbol(text)
#     morph = pymorphy2.MorphAnalyzer()
    stop_words = set(stopwords.words('english')) | set(
        stopwords.words('russian'))
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if not token in stop_words]
    tokens = [lemmatizer.lemmatize(str(token)) for token in tokens]
    return tokens


def get_path(row):
    #     print(row)
    path = '../HackatonData/'
    path += str(row.full_text_file) + '/' + str(row.full_text_file) + \
        '/' 'pmc_json/' + str(row.pmcid) + '.xml.json'
    return path


def open_file(path):
    with open(path, 'r') as f:
        datastore = json.load(f)
    return datastore


def get_text_full(datastore):
    text_full = ''
    for text_value in datastore['body_text']:
        text_full += text_value['text']
    return text_full


def analyse_text(row):
    path = get_path(row)
    data = open_file(path)
    text_full = get_text_full(data)
    tokens = tokenize(text_full)
    return tokens


class TfIdfVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(w2v.values())))

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class MeanVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(next(iter(w2v.values())))

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


data_analyse = metadata.loc[0:1000]
texts_tokenizes = data_analyse.apply(analyse_text, axis=1)


model_w2v = word2vec.Word2Vec(texts_tokenizes, size=300, window=10, workers=4)
w2v = dict(zip(model_w2v.wv.index2word, model_w2v.wv.syn0))

tfidf = TfIdfVectorizer(w2v).fit(texts_tokenizes)
data_mean_tfidf = tfidf.transform(texts_tokenizes)

clustering = DBSCAN(eps=4, min_samples=2).fit(data_mean_tfidf)
labels = clustering.labels_
labels[labels != -1] = 1

y = labels
X = data_mean_tfidf

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
accuracy_score(y_test, y_pred)


def dummy_fun(doc):
    return doc


tfidf = TfidfVectorizer(analyzer='word', tokenizer=dummy_fun,
                        preprocessor=dummy_fun)
tfidf = tfidf.fit(texts_tokenizes)

model_class_name = './models/model_classification.sav'
model_w2v_name = './models/model_w2v.sav'
model_tfidf = './models/model_tfidf.sav'
joblib.dump(neigh, model_class_name)
joblib.dump(model_w2v, model_w2v_name)
joblib.dump(tfidf, model_tfidf)
print('eixt')
