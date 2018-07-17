from lxml import etree
from scipy.spatial import distance
import scipy
import re
import fnmatch
import numpy as np
import pandas as pd
import os

DOCS = dict()

DATA_PATH = "data/Single/Source/DUC/"
SUMMARY_PATH = "data/Single/Summ/Extractive/"


def read_document(doc_name):
    file_path = DATA_PATH + doc_name

    with open(file_path) as fp:
        doc = fp.readlines()
        content = ""
        for line in doc:
            content += line

        fp.close()
    sentences = re.split("\.|\?|\!", content)
    while sentences.__contains__(''):
        sentences.remove('')

    words = list(set(map(lambda x: x.strip(), content.replace("?", " ").replace("!", " ").replace(".", " ").
                replace("؟", " ").replace("!", " ").replace("،", " ").split(" "))))
    if words.__contains__(''):
        words.remove('')
    return sentences, words


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

    sentences = re.split("\.|\?|\!", contents)
    while sentences.__contains__(''):
        sentences.remove('')

    words = list(set(map(lambda x: x.strip(), contents.replace("?", " ").replace("!", " ").replace(".", " ").
                         replace("؟", " ").replace("!", " ").replace("،", " ").split(" "))))
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
    for i in range(1, 9):
        subDir = os.path.join(rootdir1, "Track" + str(i) + "/Summ/")
        for root, dirs, files in os.walk(subDir):
            for file in files:
                if dir+'.E' in file:
                    with open(__find_in_subdirectory(file)) as fp:
                        doc = fp.readlines()
                        content = ""
                        for line in doc:
                            content += line
                        fp.close()
                    sentences = re.split("\.|\?|\!", content)
                    summaries.append(sentences)
    return summaries


def read_single_ref_summaries(filename):
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