import os
import pprint

DOCS = dict()


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


read_documents()
