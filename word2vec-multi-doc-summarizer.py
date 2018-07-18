import numpy as np
import os
import sparse_coding as sc
from evaluation import Level
import evaluation as eval
import utils as util


NUMBER_SUMMARY_SET_ELEMENT = 5
LAMBDA = 0.5
TSTOP = 0.0001
MAX_CONSE_REJ = 100

rouge1_fscores_list = list()
rouge2_fscores_list = list()
rouge1_precisions_list = list()
rouge2_precisions_list = list()
rouge1_recalls_list = list()
rouge2_recalls_list = list()


w2v = util.read_word2vec_model()
rootdir = "data/Multi"
for i in range(1,9):
    subDir = os.path.join(rootdir, "Track"+str(i)+"/Source/")
    for root, dirs, files in os.walk(subDir):
        if dirs:
            for dir in dirs:
                reference_summaries, K = util.read_multi_ref_summaries(dir)
                sentences, _ = util.read_documents(subDir+dir+"/")
                word_2_vec = util.make_word_2_vec(sentences, w2v)
                try:
                    rouge_1_fscore, rouge_2_fscore, rouge_1_precision, rouge_2_precision, \
                    rouge_1_recall, rouge_2_recall = \
                        util.evaluate(word_2_vec, NUMBER_SUMMARY_SET_ELEMENT, LAMBDA, TSTOP, MAX_CONSE_REJ, reference_summaries)
                    rouge1_fscores_list.append(rouge_1_fscore)
                    rouge2_fscores_list.append(rouge_2_fscore)
                    rouge1_precisions_list.append(rouge_1_precision)
                    rouge2_precisions_list.append(rouge_2_precision)
                    rouge1_recalls_list.append(rouge_1_recall)
                    rouge2_recalls_list.append(rouge_2_recall)
                except ValueError:
                    print("Sample larger than population")

print("Rouge-1 FScore Avg       : ", sum(rouge1_fscores_list)/len(rouge1_fscores_list))
print("Rouge-2 FScore Avg       : ", sum(rouge2_fscores_list)/len(rouge2_fscores_list))
print("Rouge-1 Precision Avg    : ", sum(rouge1_precisions_list)/len(rouge1_precisions_list))
print("Rouge-2 Precision Avg    : ", sum(rouge2_precisions_list)/len(rouge2_precisions_list))
print("Rouge-1 Recall Avg       : ", sum(rouge1_recalls_list)/len(rouge1_recalls_list))
print("Rouge-2 Recall Avg       : ", sum(rouge2_recalls_list)/len(rouge2_recalls_list))