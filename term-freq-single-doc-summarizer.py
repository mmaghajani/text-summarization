import os
import sparse_coding as sc
import evaluation as eval
import numpy as np
from evaluation import Level
import utils as util


NUMBER_SUMMARY_SET_ELEMENT = 5
LAMBDA = 0.1
TSTOP = 0.0001
MAX_CONSE_REJ = 100


directory = os.fsencode(util.DATA_PATH)
rouge1_fscores_list = list()
rouge2_fscores_list = list()
rouge1_precisions_list = list()
rouge2_precisions_list = list()
rouge1_recalls_list = list()
rouge2_recalls_list = list()
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    sentences, words = util.read_document(filename)
    reference_summaries = util.read_single_ref_summaries(filename[:-4])

    term_frequency = util.make_term_frequency(sentences, words)
    candidate_set = np.array(list([*v] for k, v in term_frequency.items()))
    try:
        summary_set = sc.MDS_sparse(candidate_set, NUMBER_SUMMARY_SET_ELEMENT, LAMBDA, TSTOP, MAX_CONSE_REJ)
        summary_text = util.summary_vector_to_text_as_list(summary_set, term_frequency)
        rouge_1_fscores = 0
        rouge_2_fscores = 0
        rouge_1_precisions = 0
        rouge_2_precisions = 0
        rouge_1_recalls = 0
        rouge_2_recalls = 0
        for summary_ref in reference_summaries:
            rouge_1_fscore = eval.rouge_Fscore(summary_text,  summary_ref, Level.Rouge_1)
            rouge_1_fscores += rouge_1_fscore
            rouge_1_precision = eval.rouge_precision(summary_text,  summary_ref, Level.Rouge_1)
            rouge_1_precisions += rouge_1_precision
            rouge_1_recall = eval.rouge_recall(summary_text,  summary_ref, Level.Rouge_1)
            rouge_1_recalls += rouge_1_recall

            rouge_2_fscore = eval.rouge_Fscore(summary_text,  summary_ref, Level.Rouge_2)
            rouge_2_fscores += rouge_2_fscore
            rouge_2_precision = eval.rouge_precision(summary_text,  summary_ref, Level.Rouge_2)
            rouge_2_precisions += rouge_2_precision
            rouge_2_recall = eval.rouge_recall(summary_text,  summary_ref, Level.Rouge_2)
            rouge_2_recalls += rouge_2_recall

        print("Rouge-1 Fscore : ", rouge_1_fscores/5)
        print("Rouge-2 Fscore : ", rouge_2_fscores/5)
        print("------------------------------------")
        rouge1_fscores_list.append(rouge_1_fscores/5)
        rouge2_fscores_list.append(rouge_2_fscores/5)
        rouge1_precisions_list.append(rouge_1_precisions/5)
        rouge2_precisions_list.append(rouge_2_precisions/5)
        rouge1_recalls_list.append(rouge_1_recalls/5)
        rouge2_recalls_list.append(rouge_2_recalls/5)
    except ValueError:
        print("Sample larger than population")

print("Rouge-1 FScore Avg       : ", sum(rouge1_fscores_list)/len(rouge1_fscores_list))
print("Rouge-2 FScore Avg       : ", sum(rouge2_fscores_list)/len(rouge2_fscores_list))
print("Rouge-1 Precision Avg    : ", sum(rouge1_precisions_list)/len(rouge1_precisions_list))
print("Rouge-2 Precision Avg    : ", sum(rouge2_precisions_list)/len(rouge2_precisions_list))
print("Rouge-1 Recall Avg       : ", sum(rouge1_recalls_list)/len(rouge1_recalls_list))
print("Rouge-2 Recall Avg       : ", sum(rouge2_recalls_list)/len(rouge2_recalls_list))
