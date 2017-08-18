from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.utils import shuffle
import multiprocessing
import numpy as np
import nltk
import os
import logging
logging.basicConfig(filename='model.log', level=logging.INFO)


class DocIter(object):

    def __init__(self, training_set):
        self.training_set = training_set

    def __iter__(self):
        # want to intersperse the collections semi randomly
        for i, (s, b, is_spam) in enumerate(self.training_set):
            # the collections could be different sizes
            sub_words = nltk.word_tokenize(s)
            for _ in range(5):
                yield TaggedDocument(sub_words, [i])
            for sent in nltk.sent_tokenize(b):
                words = nltk.word_tokenize(sent)
                yield TaggedDocument(words, [i])
            for _ in range(15):
                if is_spam:
                    yield TaggedDocument('<spam>', [i])
                else:
                    yield TaggedDocument('<ham>', [i])


def train_d2v_model(training_set):

    doc_iter = DocIter(training_set)

    model = Doc2Vec(documents=doc_iter,
                    iter=20,
                    size=300,
                    min_count=1,
                    sample=1e-5,
                    window=5,
                    hs=0,
                    negative=10,
                    alpha=0.1,
                    min_alpha=0.01,
                    dm=0,
                    workers=multiprocessing.cpu_count())

    if not os.path.isdir('models'):
            os.mkdir('models')
    model.save('models/d2v.model')
    return model


def txt_2_vectors(txt_set, model, is_training):
    txt_vectors = []
    txt_ys = []

    for i, (s, b, is_spam) in enumerate(txt_set):
        if is_training:
            v = model.docvecs[i]
        else:
            words = nltk.word_tokenize(s * 5 + b)
            v = model.infer_vector(words)

        v = np.append(v, [len(s), len(b)])

        txt_vectors.append(v)
        txt_ys.append(int(is_spam))

    txt_vectors = np.array(txt_vectors)
    txt_ys = np.reshape(np.array(txt_ys), (len(txt_ys), 1))

    txt_vectors, txt_ys = shuffle(txt_vectors, txt_ys)

    # print txt_vectors.shape, txt_ys.shape

    return txt_vectors, txt_ys
