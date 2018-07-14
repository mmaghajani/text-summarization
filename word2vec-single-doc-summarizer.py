
import numpy as np
import pandas as pd
import re


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
        DocMatix[i] = AvgSent2vec(words, model)
        word2vec[data[i]] = DocMatix[i]
    print("features calculated")
    # print(word2vec)
    train_df = pd.DataFrame(DocMatix)

    train_df.to_csv('AvgSent2vec.csv', index=False)

    return train_df

result = represent(read_document('ALF.CU.13910117.019.txt'), w2v)
print(result)
print(result.shape)



