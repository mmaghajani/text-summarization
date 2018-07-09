import math
import os
import pprint
import re

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


def tokenize():
    for filename in DOCS.keys():
        content = DOCS[filename]
        words = content.replace("\n", " ").replace(":", " ").replace("»", " ").replace("'", " ").\
            replace(".", " ").replace("،", " ").replace("«", " "). \
            replace("!", " ").replace(";", " ").replace(")", " ").replace("(", " "). \
            split(" ")
        for word in set(words):
            if word not in WORD_DATA.keys():
                WORD_DATA[word] = {filename: words.count(word)}
            else:
                WORD_DATA[word].update({filename: words.count(word)})

    for word in WORD_DATA.keys():
        count = 0
        for num in WORD_DATA[word].values():
            count += num
        WORD_DATA[word].update({"all": count})


def mutual_info():
    MIs = dict()
    N = 100
    score = dict()
    for word in WORD_DATA.keys():
        Nw = WORD_DATA.get(word)["all"]
        Nwbar = N - Nw
        for doc in DOCS.keys():
            Ni = len(DOCS.get(doc))
            Niw = WORD_DATA.get(word)[doc]
            Niwbar = Ni - Niw
            Nibar = N - Ni
            Nibarw = Nw - Niw
            Nibarwbar = Nibar - Nibarw
            a = 0.0000000000
            b = 0.0000000000
            c = 0.0000000000
            d = 0.0000000000
            try:
                if Niw is not 0:
                    a = (Niw / N) * math.log2((N * Niw) / (Nw * Ni))
                if Niwbar is not 0:
                    b = (Niwbar / N) * math.log2((N * Niwbar) / (Nwbar * Ni))
                if Nibarw is not 0:
                    c = (Nibarw / N) * math.log2((N * Nibarw) / (Nw * Nibar))
                if Nibarwbar is not 0:
                    d = (Nibarwbar / N) * math.log2((N * Nibarwbar) / (Nwbar * Nibar))
            except ValueError as e:
                print(N , Niw , Ni , Niwbar , Nwbar, Nibarwbar, Nibar)
            MI = a + b + c + d
            if word not in MIs.keys():
                MIs[word] = {doc: MI}
            else:
                MIs.get(word).update({doc: MI})
        s = 0
        for doc in DOCS.keys():
            Ni = len(DOCS.get(doc))
            Pci = Ni / N
            s += (MIs.get(word)[doc] * Pci)
        score[word] = s


sentences, words = read_document('ALF.CU.13910117.019.txt')
print(sentences, "\n", words)
# tokenize()