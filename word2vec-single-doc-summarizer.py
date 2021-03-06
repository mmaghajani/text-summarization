import numpy as np
import os
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
directory = os.fsencode(util.DATA_PATH)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    # sentences = read_document(filename)
    reference_summaries, K = util.read_single_ref_summaries(filename[:-4])

    sentence, _ = util.read_document(filename)
    word2vec = util.make_word_2_vec(sentence, w2v)

    candidate_set = np.array(list([*v] for k, v in word2vec.items()))
    try:
        rouge_1_fscore, rouge_2_fscore , rouge_1_precision, rouge_2_precision,\
            rouge_1_recall, rouge_2_recall = \
            util.evaluate(word2vec, NUMBER_SUMMARY_SET_ELEMENT, LAMBDA, TSTOP, MAX_CONSE_REJ, reference_summaries)
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