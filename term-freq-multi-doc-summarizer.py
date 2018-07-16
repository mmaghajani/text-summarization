import os
import re
from lxml import etree
import sparse_coding as sc
import evaluation as eval
import numpy as np
from evaluation import Level
import fnmatch

DOCS = dict()
WORD_DATA = dict()

NUMBER_SUMMARY_SET_ELEMENT = 5
LAMBDA = 3
TSTOP = 0.0001
MAX_CONSE_REJ = 100


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


def make_summary_as_text(summary_set, term_frequency):
    summary_text = list()
    for sentence in term_frequency.keys():
        if term_frequency[sentence] in summary_set:
            summary_text.append(sentence)
    return summary_text


def findInSubdirectory(filename, subdirectory="data/Multi"):
    if subdirectory:
        path = subdirectory
    else:
        path = os.getcwd()
    for root, dirs, names in os.walk(path):
        if filename in names:
            return os.path.join(root, filename)
    return 'File not found'


def readSummaries(dir):
    summaries = []
    rootdir1 = "data/Multi"
    for i in range(1, 9):
        subDir = os.path.join(rootdir1, "Track" + str(i) + "/Summ/")
        for root, dirs, files in os.walk(subDir):
            for file in files:
                if dir+'.E' in file:
                    with open(findInSubdirectory(file)) as fp:
                        doc = fp.readlines()
                        content = ""
                        for line in doc:
                            content += line
                        fp.close()
                    summaries.append(content)

    return summaries


rootdir = "data/Multi"
for i in range(1,9):
    subDir = os.path.join(rootdir, "Track"+str(i)+"/Source/")
    for root, dirs, files in os.walk(subDir):
        if dirs:
            for dir in dirs:
                print(dir)
                print(subDir+dir+"/")
                sentences, words = read_documents(subDir+dir+"/")
                print(sentences)
                reference_summaries = readSummaries(dir)
                term_frequency = make_term_frequency(sentences, words)
                candidate_set = np.array(list([*v] for k, v in term_frequency.items()))
                summary_set = sc.MDS_sparse(candidate_set, NUMBER_SUMMARY_SET_ELEMENT, LAMBDA, TSTOP, MAX_CONSE_REJ)
                summary_text = make_summary_as_text(summary_set, term_frequency)
                rouge_1_fscores = 0
                rouge_2_fscores = 0
                for summary_ref in reference_summaries:
                    rouge_1_fscore = eval.RougeFScore(summary_text, summary_ref, Level.Rouge_1)
                    rouge_1_fscores += rouge_1_fscore

                    rouge_2_fscore = eval.RougeFScore(summary_text, summary_ref, Level.Rouge_2)
                    rouge_2_fscores += rouge_2_fscore

                print("Rouge-1 Fscore : ", rouge_1_fscores / 5)
                print("Rouge-2 Fscore : ", rouge_2_fscores / 5)
                # print(result)
