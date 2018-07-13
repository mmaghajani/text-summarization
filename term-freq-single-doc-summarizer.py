import re
import sparse_coding as sc
import evaluation as eval
import pandas as pd

DOCS = dict()
WORD_DATA = dict()

NUMBER_SUMMARY_SET_ELEMENT = 10
LAMBDA = 3
TSTOP = 0.0001
MAX_CONSE_REJ = 100


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


def make_summary_text(summary_set, term_frequency):
    summary_text = ''
    for sentence in term_frequency.keys():
        if term_frequency[sentence] in summary_set.values():
            summary_text += sentence
    return summary_text


sentences, words = read_document('ALF.CU.13910117.019.txt')
reference_summary = read_document('path to ref summary')
listed_word = list(words)
term_frequency = make_term_frequency(sentences, listed_word)
candidate_set = pd.DataFrame([*v] for k, v in term_frequency.items())
summary_set = sc.MDS_sparse(candidate_set, NUMBER_SUMMARY_SET_ELEMENT, LAMBDA, TSTOP, MAX_CONSE_REJ)
summary_text = make_summary_text(summary_set, term_frequency)
rouge_fscore = eval.RougeFScore(summary_text,  reference_summary, NUMBER_SUMMARY_SET_ELEMENT)
print("Rouge FSCORE : ", rouge_fscore)
