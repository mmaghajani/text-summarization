import os
import re
import sparse_coding as sc
import evaluation as eval
import pandas as pd
from evaluation import Level
import fnmatch


DOCS = dict()
WORD_DATA = dict()

NUMBER_SUMMARY_SET_ELEMENT = 10
LAMBDA = 3
TSTOP = 0.0001
MAX_CONSE_REJ = 100

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


def make_summary_text(summary_set, term_frequency):
    summary_text = ''
    for sentence in term_frequency.keys():
        if term_frequency[sentence] in summary_set.values():
            summary_text += sentence
    return summary_text


def read_ref_summaries(filename):
    directory = os.fsencode(SUMMARY_PATH)
    summaries = list()
    for file in os.listdir(directory):
        name = os.fsdecode(file)
        if fnmatch.fnmatch(name, filename + '*'):
            with open(SUMMARY_PATH + name) as summ_file:
                content = summ_file.readlines()
                summaries.append(content)
                summ_file.close()
    return summaries


directory = os.fsencode(DATA_PATH)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    sentences, words = read_document(filename)
    reference_summaries = read_ref_summaries(filename[:-4])

    listed_word = list(words)
    term_frequency = make_term_frequency(sentences, listed_word)
    candidate_set = pd.DataFrame([*v] for k, v in term_frequency.items())
    summary_set = sc.MDS_sparse(candidate_set, NUMBER_SUMMARY_SET_ELEMENT, LAMBDA, TSTOP, MAX_CONSE_REJ)
    summary_text = make_summary_text(summary_set, term_frequency)

    rouge_1_fscores = 0
    rouge_2_fscores = 0
    for summary_ref in reference_summaries:
        rouge_1_fscore = eval.RougeFScore(summary_text,  summary_ref, Level.Rouge_1)
        rouge_1_fscores += rouge_1_fscore

        rouge_2_fscore = eval.RougeFScore(summary_text,  summary_ref, Level.Rouge_2)
        rouge_2_fscores += rouge_2_fscore

    print("Rouge-1 Fscore : ", rouge_1_fscores/5)
    print("Rouge-2 Fscore : ", rouge_2_fscores/5)
