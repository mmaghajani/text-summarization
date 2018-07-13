import math
import os
import pprint
import random
import re
import scipy

import numpy as np
from numpy import linalg as LA
import copy
import pandas as pd
from scipy.spatial import distance

DOCS = dict()
WORD_DATA = dict()


def read_document(doc_name):
    file_path = "data/Single/Source/DUC/" + doc_name

    with open(file_path) as fp:
        doc = fp.readlines()
        content = ""
        for line in doc:
            content += line

        fp.close()
    sentences = re.split("\.|\?|\!", content)
    while sentences.__contains__(''):
        sentences.remove('')

    words = set(map(lambda x: x.strip(), content.replace("?", " ").replace("!", " ").replace(".", " ").
                replace("؟", " ").replace("!", " ").replace("،", " ").split(" ")))
    if words.__contains__(''):
        words.remove('')
    return sentences, words


def make_term_frequency(sentences, words):
    term_frequency = dict()
    for sentence in sentences:
        vector = list()
        for i in range(0, len(words)):
            word = words[i]
            vector.append(sentence.count(word))
        term_frequency[sentence] = vector
    return term_frequency


def sparse_coding(S, summary, lam):
    rotated_S = np.rot90(S)
    rotated_summary = np.rot90(summary)
    n = rotated_S.shape[1]
    k = rotated_summary.shape[1]
    A_current = np.full((k, n), 1/k)
    lam1 = np.full((k, n), lam)
    t = 0
    while t < 100:
        A_new = np.divide(np.multiply(A_current, np.matmul(rotated_summary.transpose(), rotated_S)),
                          np.add(np.matmul(
                              np.matmul(rotated_summary.transpose(), rotated_summary), A_current), lam1))
        if LA.norm(A_new - A_current) < 0.01:
            break
        t += 1
        A_current = A_new
    return A_current


def J(S, A, summary_current, lam):
    loss1 = 0
    loss2 = 0
    for i in range(len(S)):
        prediction = 0
        for j in range(len(summary_current)):
            prediction += A[j][i] * np.array(S[j])

        loss1 += LA.norm(np.array(S[i])-prediction)      # L2-norm
        loss2 += LA.norm(np.array(A[:, i]), ord=1)  # L1-norm of a:i

    return loss1+(lam*loss2)


def update_summary(s, T, summary_new, S):
    # TODO : T coming in play!
    min_dist = 9999999  # max
    min_index = 0
    for index in range(len(S)):
        temp_dist = scipy.spatial.distance.euclidean(s,S[index])
        if temp_dist < min_dist and index not in summary_new.keys():
            min_dist = temp_dist
            min_index = index
    return S[min_index], min_index


def Accept(s_index, tmp_index, summary_current, T, S, A, lam):
    summary_temp = copy.deepcopy(summary_current)
    summary_temp.pop(s_index)   # remove a sentence
    summary_temp[tmp_index] = S[tmp_index]  # replace a new sentence
    current_J = J(S, A, summary_current, lam)
    next_J = J(S, A, summary_temp, lam)
    if next_J < current_J:
        return True
    else:
        deltaE = current_J - next_J
        if T is 0:
            return False
        p = math.e**(deltaE/T)
        if random.random() < p:
            return True
        else:
            return False


def update_T(step):
    return math.e**(-1*(step/5-5))


def MDS_sparse(S, k, lam, Tstop, MaxConseRej):
    """
    :param S:       This is candidate set. A n*d matrice
    :param k:       The number of sentence in summary set(basis functions)
    :param lam:     The lambda parameter
    :param Tstop:   Stop temperature
    :param MaxConseRej:
    :return:        Index of sentences in candidate set that be included in summary set
    """
    n = S.shape[0]
    summary_current = dict()    # initialized randomly matrice k*d
    for index in random.sample(range(0, n), k):
        summary_current[index] = S[index]
    rej = 0
    Jopti = 9999999     # max
    summary_opti = copy.deepcopy(summary_current)
    step = 0
    T = update_T(step)    # arbitrary
    while T > Tstop:
        A = sparse_coding(S, summary_current, lam)  # k*n matrice
        current_J = J(S, A, summary_current, lam)
        if current_J < Jopti:
            Jopti = current_J
            summary_opti = copy.deepcopy(summary_current)
        else:
            rej += 1
            if rej >= MaxConseRej:
                return summary_opti.keys()
        summary_new = dict()
        for index in summary_current.keys():
            s = summary_current[index]
            tmp, tmp_index = update_summary(s, T, summary_new, S)
            if Accept(index, tmp_index, summary_current, T, S, A, lam):
                summary_new[tmp_index] = tmp
            else:
                summary_new[index] = s
        summary_current = copy.deepcopy(summary_new)
        step += 1
        T = update_T(step)
    return summary_opti.keys()


sentences, words = read_document('ALF.CU.13910117.019.txt')
listed_word = list(words)
print(listed_word)
term_frequency = make_term_frequency(sentences, listed_word)
