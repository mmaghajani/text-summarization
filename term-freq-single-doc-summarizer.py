import re
import sparse_coding as sc
import evaluation as eval


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


sentences, words = read_document('ALF.CU.13910117.019.txt')
reference_summary = read_document('path to ref summary')
listed_word = list(words)
term_frequency = make_term_frequency(sentences, listed_word)
summary_set = sc.MDS_sparse(term_frequency, 10, 3, 0.0001, 100)
eval.RougeFScore(summary_set,  reference_summary, 10)
