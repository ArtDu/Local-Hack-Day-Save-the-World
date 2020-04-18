from collections import defaultdict
import numpy as np
import pandas as pd
import re
import pymorphy2
from nltk.corpus import stopwords
import nltk


def deleted_symbol(text):
    pattern_end_html = r'</\w*>|<\w*>'
    pattern_start_html = r'<.*>|\n|\r|\r'
    pattern = r"[^a-zA-Z]"
    text = re.sub(pattern_end_html, ' ', text)
    text = re.sub(pattern_start_html, '', text)

    text = re.sub("[^a-zA-Z]", " ", text)

    return text


def tokenize(text):
    text = deleted_symbol(text)
    morph = pymorphy2.MorphAnalyzer()
    stop_words = set(stopwords.words('english')) | set(
        stopwords.words('russian'))
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if not token in stop_words]
    tokens = [morph.parse(str(token))[0].normal_form for token in tokens]
    return tokens


class MeanVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TextProcessor:
    def __init__(self, model_w2v):
        self.w2v = dict(zip(model_w2v.wv.index2word, model_w2v.wv.syn0))

    def transform(self, text):
        text_tokens = tokenize(text)
        text_vec = MeanVectorizer(self.w2v).transform([text_tokens])
        return text_vec[0]


class Predition:
    def __init__(self, model):
        self.model = model

    def predict(self, text_vec):
        pred = self.model.predict([text_vec])[0]
        if pred == -1:
            return 'fake'
        else:
            return 'non fake'


if __name__ == '__main__':
    import numpy
    import joblib
    from gensim.models import Word2Vec
    model_w2v_name = './models/model_w2v.sav'

    model_w2v = joblib.load(model_w2v_name)

    text = 'Hello you luck guy'
    tp = TextProcessor(model_w2v)
    text_tokens = tp.transform(text)
    print(text_tokens)
    model_class_name = './models/model_classification.sav'
    model_classification = joblib.load(model_class_name)
    prediction = Predition(model_classification)
    print(prediction.predict(text_tokens))
