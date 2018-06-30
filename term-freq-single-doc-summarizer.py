import os
import pprint

DOCS = dict()
WORD_DATA = dict()


def read_documents():
    directory_path = "data/Single/Source/DUC/"
    directory = os.fsencode(directory_path)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        with open(directory_path + filename) as fp:
            doc = fp.readlines()
            content = ""
            for line in doc:
                content += line

            fp.close()
        DOCS[filename] = content


def tokenize():
    for filename in DOCS.keys():
        content = DOCS[filename]
        words = content.replace("\n", " ").replace(".", " ").replace("،", " "). \
            replace("!", " ").replace(";", " ").replace(")", " ").replace("(", " "). \
            split(" ")
        for word in set(words):
            if filename not in WORD_DATA.keys():
                WORD_DATA[filename] = {word: words.count(word)}
            else:
                WORD_DATA[filename].update({word: words.count(word)})


read_documents()
tokenize()
pprint.pprint(WORD_DATA)