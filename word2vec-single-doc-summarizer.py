
import numpy as np
import pandas as pd
import re
import os
import fnmatch
import sparse_coding as sc
from evaluation import Level
import evaluation as eval
import scipy


NUMBER_SUMMARY_SET_ELEMENT = 5
LAMBDA = 3
TSTOP = 0.0001
MAX_CONSE_REJ = 100

DATA_PATH = "data/Single/Source/DUC/"
SUMMARY_PATH = "data/Single/Summ/Extractive/"


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

    return sentences


w2v = dict()
print("waiting to load word2vec model...")
with open('twitt_wiki_ham_blog.fa.text.100.vec', 'r', encoding='utf-8') as infile:
    first_line = True
    for line in infile:
        if first_line:
            first_line = False
            continue
        tokens = line.split()
        w2v[tokens[0]] = [float(el) for el in tokens[1:]]
        if len(w2v[tokens[0]]) != 100:
            print('Bad line!')
print("model loaded")


def AvgSent2vec(words, model):
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.mean(axis=0)
    return v / np.sqrt((v ** 2).sum())


def represent(data, model):
    word2vec = dict()  # final dictionary containing sentence as the key and its representation as value
    DocMatix = np.zeros((len(data), 100))

    for i in range(len(data)):

        words = list(map(lambda x: x.strip(), data[i].replace("?", " ").replace("!", " ").replace(".", " ").
                    replace("؟", " ").replace("!", " ").replace("،", " ").split(" ")))
        if words.__contains__(''):
            words.remove('')

        result = AvgSent2vec(words, model)
        if not( np.isnan(result).any()):
            DocMatix[i] = result
            word2vec[data[i]] = DocMatix[i]
    print("features calculated")
    # print(word2vec)
    train_df = pd.DataFrame(DocMatix)

    train_df.to_csv('AvgSent2vec.csv', index=False)

    return word2vec


def summary_vector_to_text_as_list(summary_set, term_frequency):
    summary_text = list()
    for sen_vec in summary_set:
        min_dist = 9999999   # max
        min_sentence = ''
        for sentence in term_frequency.keys():
            temp_dist = scipy.spatial.distance.euclidean(sen_vec, term_frequency[sentence])
            if temp_dist < min_dist:
                min_dist = temp_dist
                min_sentence = sentence
        summary_text.append(min_sentence)
    return summary_text


def read_ref_summaries(filename):
    directory = os.fsencode(SUMMARY_PATH)
    summaries = list()
    for file in os.listdir(directory):
        name = os.fsdecode(file)
        if fnmatch.fnmatch(name, filename + '*'):
            with open(SUMMARY_PATH + name) as summ_file:
                lines = summ_file.readlines()
                content = ''
                for line in lines:
                    content += line
                sentences = re.split("\.|\?|\!", content)
                summaries.append(sentences)
                summ_file.close()
    return summaries

# result = represent(read_document('ALF.CU.13910117.019.txt'), w2v)
# print(result)
# print(result.shape)

directory = os.fsencode(DATA_PATH)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    # sentences = read_document(filename)
    reference_summaries = read_ref_summaries(filename[:-4])

    word2vec = represent(read_document(filename), w2v)

    candidate_set = np.array(list([*v] for k, v in word2vec.items()))
    summary_set = sc.MDS_sparse(candidate_set, NUMBER_SUMMARY_SET_ELEMENT, LAMBDA, TSTOP, MAX_CONSE_REJ)
    summary_text = summary_vector_to_text_as_list(summary_set, word2vec)
    rouge_1_fscores = 0
    rouge_2_fscores = 0
    for summary_ref in reference_summaries:
        rouge_1_fscore = eval.RougeFScore(summary_text,  summary_ref, Level.Rouge_1)
        rouge_1_fscores += rouge_1_fscore

        rouge_2_fscore = eval.RougeFScore(summary_text,  summary_ref, Level.Rouge_2)
        rouge_2_fscores += rouge_2_fscore

    print("Rouge-1 Fscore : ", rouge_1_fscores/5)
    print("Rouge-2 Fscore : ", rouge_2_fscores/5)

