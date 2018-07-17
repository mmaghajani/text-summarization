import numpy as np
import pandas as pd
import os
from lxml import etree
import re
import sparse_coding as sc
from evaluation import Level
import evaluation as eval
import tqdm

DOCS = dict()

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
        if not(np.isnan(result).any()):
            DocMatix[i] = result
            word2vec[data[i]] = DocMatix[i]

    print("features calculated")
    train_df = pd.DataFrame(DocMatix)

    train_df.to_csv('AvgSent2vec.csv', index=False)

    return word2vec


def summary_vector_to_text_as_list(summary_set, term_frequency):
    summary_text = list()
    for sen_vec in summary_set:
        for sentence in term_frequency.keys():
            if (term_frequency[sentence] == sen_vec).all():
                summary_text.append(sentence)
                break
    return summary_text


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
                    sentences = re.split("\.|\?|\!", content)
                    summaries.append(sentences)
    return summaries


def findInSubdirectory(filename, subdirectory="data/Multi"):
    if subdirectory:
        path = subdirectory
    else:
        path = os.getcwd()
    for root, dirs, names in os.walk(path):
        if filename in names:
            return os.path.join(root, filename)
    return 'File not found'


rootdir = "data/Multi"
for i in range(1,9):
    subDir = os.path.join(rootdir, "Track"+str(i)+"/Source/")
    for root, dirs, files in os.walk(subDir):
        if dirs:
            for dir in dirs:
                # print(dir)
                # print(subDir+dir+"/")
                # result = represent(read_documents(subDir+dir+"/"), w2v)
                # summaries = readSummaries(dir)
                reference_summaries = readSummaries(dir)
                term_frequency = represent(read_documents(subDir+dir+"/"), w2v)
                candidate_set = np.array(list([*v] for k, v in term_frequency.items()))
                summary_set = sc.MDS_sparse(candidate_set, NUMBER_SUMMARY_SET_ELEMENT, LAMBDA, TSTOP, MAX_CONSE_REJ)
                summary_text = summary_vector_to_text_as_list(summary_set, term_frequency)
                print(len(summary_text))
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
                # print(result.shape)

# a = readSummaries('D91A50')
