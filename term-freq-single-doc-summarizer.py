import math
import os
import pprint
import re
import numpy as np
from numpy import linalg as LA
import pandas as pd

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
    n = S.shape[1]
    k = summary.shape[1]
    A_current = np.full((k, n), 1/k)
    lam1 = np.full((k, n), lam)
    t = 0
    while t < 100:
        A_new = np.divide(np.multiply(A_current, np.matmul(summary.transpose(), S)),
                          np.add(np.matmul(np.matmul(summary.transpose(), summary), A_current), lam1))
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


def update_summary(s, T):
    return ''


def Accept(s, tmp, summary_current, T):
    return False


def update_T(step):
    return ''


def MDS_sparse(S, k, lam, Tstop, MaxConseRej):
    summary_current = pd.DataFrame()    # initialized randomly
    T = 0
    rej = 0
    Jopti = 9999999 #max
    summary_opti = pd.DataFrame()
    step = 0
    while T > Tstop:
        A = sparse_coding(S, summary_current, lam)
        current_J = J(S, A, T, lam)
        if current_J < Jopti:
            Jopti = current_J
            summary_opti = summary_current
        else:
            rej += 1
            if rej >= MaxConseRej:
                return summary_opti
        summary_new = pd.DataFrame()
        for s in summary_current:
            tmp = update_summary(s, T)
            if Accept(s, tmp, summary_current, T):
                summary_new.add(tmp)
            else:
                summary_new.add(s)
        summary_current = summary_new
        T = update_T(step)
        step += 1
    return summary_opti


sentences, words = read_document('ALF.CU.13910117.019.txt')
listed_word = list(words)
print(listed_word)
term_frequency = make_term_frequency(sentences, listed_word)
