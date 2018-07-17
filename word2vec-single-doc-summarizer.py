import numpy as np
import pandas as pd
import os
import sparse_coding as sc
from evaluation import Level
import evaluation as eval
import utils as util


NUMBER_SUMMARY_SET_ELEMENT = 5
LAMBDA = 3
TSTOP = 0.0001
MAX_CONSE_REJ = 100


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


directory = os.fsencode(util.DATA_PATH)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    # sentences = read_document(filename)
    reference_summaries = util.read_single_ref_summaries(filename[:-4])
    sentence, _ = util.read_document(filename)
    word2vec = represent(sentence, w2v)

    candidate_set = np.array(list([*v] for k, v in word2vec.items()))
    try:
        summary_set = sc.MDS_sparse(candidate_set, NUMBER_SUMMARY_SET_ELEMENT, LAMBDA, TSTOP, MAX_CONSE_REJ)
        summary_text = util.summary_vector_to_text_as_list(summary_set, word2vec)
        rouge_1_fscores = 0
        rouge_2_fscores = 0
        for summary_ref in reference_summaries:
            rouge_1_fscore = eval.RougeFScore(summary_text,  summary_ref, Level.Rouge_1)
            rouge_1_fscores += rouge_1_fscore

            rouge_2_fscore = eval.RougeFScore(summary_text,  summary_ref, Level.Rouge_2)
            rouge_2_fscores += rouge_2_fscore

        print("Rouge-1 Fscore : ", rouge_1_fscores/5)
        print("Rouge-2 Fscore : ", rouge_2_fscores/5)
    except ValueError:
        print("Sample larger than population")

