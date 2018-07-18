import math

from lxml import etree
from scipy.spatial import distance
import scipy
import re
import fnmatch
import numpy as np
import pandas as pd
import os
from numpy import dot
from numpy.linalg import norm
import sparse_coding as sc
import evaluation as eval
from evaluation import Level

DOCS = dict()

DATA_PATH = "data/Single/Source/DUC/"
SUMMARY_PATH = "data/Single/Summ/Extractive/"


def __content_processing(content):
    content = content.replace("\n", " ").replace("(", " ").replace(")", " "). \
        replace("  ", " ").replace("  ", " ")
    sentences = re.split("\.|\?|\!", content)
    while sentences.__contains__(''):
        sentences.remove('')

    while sentences.__contains__(' \n'):
        sentences.remove(' \n')

    for sentence in sentences:
        words = sentence.split(" ")
        while words.__contains__(''):
            words.remove('')
        if len(words) < 2:
            sentences.remove(sentence)

    words = list(set(map(lambda x: x.strip(), content.replace("?", " ").replace("!", " ").replace(".", " ").
                         replace("؟", " ").replace("!", " ").replace("،", " ").split(" "))))
    if words.__contains__(''):
        words.remove('')
    return sentences, words


def read_document(doc_name):
    file_path = DATA_PATH + doc_name

    with open(file_path) as fp:
        doc = fp.readlines()
        content = ""
        for line in doc:
            content += line

        fp.close()
    return __content_processing(content)


def read_documents(directory_path):
    # directory_path = "data/Multi/Track1/Source/D91A01/"
    directory = os.fsencode(directory_path)
    contents = ""
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        content = etree.parse(directory_path + filename)
        memoryElem = content.find('TEXT')
        DOCS[filename] = memoryElem.text
        contents += memoryElem.text

    return __content_processing(contents)


def make_term_frequency(sentences, words):
    term_frequency = dict()
    for sentence in sentences:
        vector = list()
        for i in range(0, len(words)):
            word = words[i]
            vector.append(sentence.count(word))
        if norm(vector) != 0:
            term_frequency[sentence] = vector
            # term_frequency[sentence] = vector / norm(vector, ord=1)
    return term_frequency


def __avg_sent_2_vec(words, model):
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.mean(axis=0)
    return v / np.sqrt((v ** 2).sum())


def read_word2vec_model():
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
    return w2v


def make_word_2_vec(data, model):
    word2vec = dict()  # final dictionary containing sentence as the key and its representation as value
    DocMatix = np.zeros((len(data), 100))

    for i in range(len(data)):

        words = list(map(lambda x: x.strip(), data[i].replace("?", " ").replace("!", " ").replace(".", " ").
                         replace("؟", " ").replace("!", " ").replace("،", " ").split(" ")))
        if words.__contains__(''):
            words.remove('')

        result = __avg_sent_2_vec(words, model)
        if not (np.isnan(result).any()):
            DocMatix[i] = result
            word2vec[data[i]] = DocMatix[i]
    print("features calculated")
    # print(word2vec)
    train_df = pd.DataFrame(DocMatix)

    # train_df.to_csv('AvgSent2vec.csv', index=False)

    return word2vec


def __summary_vector_to_text_as_list_cosine(summary_set, representation):
    summary_text = list()
    for sen_vec in summary_set:
        min_dist = 9999999  # max
        min_sentence = ''
        if norm(sen_vec) == 0:
            print("sen vec zero")
        for sentence in representation.keys():
            if norm(representation[sentence]) == 0:
                print("rep vec zero")
            temp_dist = dot(sen_vec, representation[sentence]) / (norm(sen_vec) * norm(representation[sentence]))
            if temp_dist < min_dist:
                min_dist = temp_dist
                min_sentence = sentence
        summary_text.append(min_sentence)
    return summary_text


def __summary_vector_to_text_as_list_euclidean(summary_set, representation):
    summary_text = list()
    for sen_vec in summary_set:
        min_dist = 9999999  # max
        min_sentence = ''
        for sentence in representation.keys():
            temp_dist = scipy.spatial.distance.euclidean(sen_vec, representation[sentence])
            if temp_dist < min_dist:
                min_dist = temp_dist
                min_sentence = sentence
        summary_text.append(min_sentence)
    return summary_text


def __find_in_subdirectory(filename, subdirectory="data/Multi"):
    if subdirectory:
        path = subdirectory
    else:
        path = os.getcwd()
    for root, dirs, names in os.walk(path):
        if filename in names:
            return os.path.join(root, filename)
    return 'File not found'


def read_multi_ref_summaries(dir):
    summaries = []
    rootdir1 = "data/Multi"
    summary_len = 0
    for i in range(1, 9):
        subDir = os.path.join(rootdir1, "Track" + str(i) + "/Summ/")
        for root, dirs, files in os.walk(subDir):
            for file in files:
                if dir + '.E' in file:
                    with open(__find_in_subdirectory(file)) as fp:
                        doc = fp.readlines()
                        content = ""
                        for line in doc:
                            content += line
                        fp.close()
                    sentences = re.split("\.|\?|\!", content)
                    summary_len += len(sentences)
                    summaries.append(sentences)
    return summaries, math.ceil(summary_len/5)


def read_single_ref_summaries(filename):
    directory = os.fsencode(SUMMARY_PATH)
    summaries = list()
    summary_len = 0
    for file in os.listdir(directory):
        name = os.fsdecode(file)
        if fnmatch.fnmatch(name, filename + '*'):
            with open(SUMMARY_PATH + name) as summ_file:
                lines = summ_file.readlines()
                content = ''
                for line in lines:
                    content += line
                sentences = re.split("\.|\?|\!", content)
                summary_len += len(sentences)
                summaries.append(sentences)
                summ_file.close()
    return summaries, math.ceil(summary_len / 5)


def evaluate(representation, K, LAMBDA, t_stop, max_conse_rej, reference_summaries):
    candidate_set = np.array(list([*v] for k, v in representation.items()))
    summary_set = sc.MDS_sparse(candidate_set, K, LAMBDA, t_stop, max_conse_rej)
    summary_text = __summary_vector_to_text_as_list_euclidean(summary_set, representation)
    rouge_1_fscores = 0
    rouge_2_fscores = 0
    rouge_1_precisions = 0
    rouge_2_precisions = 0
    rouge_1_recalls = 0
    rouge_2_recalls = 0
    summ_len = 0
    for summary_ref in reference_summaries:
        summ_len += len(summary_ref)
        rouge_1_fscore = eval.rouge_Fscore(summary_text, summary_ref, Level.Rouge_1)
        rouge_1_fscores += rouge_1_fscore
        rouge_1_precision = eval.rouge_precision(summary_text, summary_ref, Level.Rouge_1)
        rouge_1_precisions += rouge_1_precision
        rouge_1_recall = eval.rouge_recall(summary_text, summary_ref, Level.Rouge_1)
        rouge_1_recalls += rouge_1_recall

        rouge_2_fscore = eval.rouge_Fscore(summary_text, summary_ref, Level.Rouge_2)
        rouge_2_fscores += rouge_2_fscore
        rouge_2_precision = eval.rouge_precision(summary_text, summary_ref, Level.Rouge_2)
        rouge_2_precisions += rouge_2_precision
        rouge_2_recall = eval.rouge_recall(summary_text, summary_ref, Level.Rouge_2)
        rouge_2_recalls += rouge_2_recall
    print("Rouge-1 Fscore : ", rouge_1_fscores / 5)
    print("Rouge-2 Fscore : ", rouge_2_fscores / 5)
    print("------------------------------------")
    return rouge_1_fscores / 5, rouge_2_fscores / 5, rouge_1_precisions / 5, rouge_2_precisions / 5, \
        rouge_1_recalls / 5, rouge_2_recalls / 5
