import numpy as np
from numpy import linalg as LA
from scipy.spatial import distance
import scipy
import copy
import math
import random


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
        A_current = copy.deepcopy(A_new)
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


def is_in_numpy(sentence, matrice):
    for s in matrice:
        if (sentence==s).all():
            return True
    return False


def update_summary(s, T, summary_new, S):
    # TODO : T coming in play!
    min_dist = 9999999  # max
    min_sentence = 0
    for sentence in S:
        temp_dist = scipy.spatial.distance.euclidean(s, sentence)
        if temp_dist < min_dist and not is_in_numpy(sentence, summary_new):
            min_dist = temp_dist
            min_sentence = sentence
    return min_sentence


def Accept(s, tmp, summary_current, T, S, A, lam):
    summary_temp = list()
    for sentence in summary_current:
        if (s == sentence).all():
            summary_temp.append(tmp)
        else:
            summary_temp.append(sentence)
    summary_temp = np.array(summary_temp)
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
    summary_current = list()    # initialized randomly matrice k*d
    for index in random.sample(range(0, n), k):
        summary_current.append(S[index])
    summary_current = np.array(summary_current)
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
                return summary_opti
        summary_new = list()
        for s in summary_current:
            tmp = update_summary(s, T, summary_new, S)
            if Accept(s, tmp, summary_current, T, S, A, lam):
                summary_new.append(tmp)
            else:
                summary_new.append(s)
        summary_current = copy.deepcopy(np.array(summary_new))
        step += 1
        T = update_T(step)
    return summary_opti
