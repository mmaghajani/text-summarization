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


w2v = util.read_word2vec_model()
rootdir = "data/Multi"
for i in range(1,9):
    subDir = os.path.join(rootdir, "Track"+str(i)+"/Source/")
    for root, dirs, files in os.walk(subDir):
        if dirs:
            for dir in dirs:
                reference_summaries = util.read_multi_ref_summaries(dir)
                sentences, _ = util.read_documents(subDir+dir+"/")
                term_frequency = util.make_word_2_vec(sentences, w2v)
                candidate_set = np.array(list([*v] for k, v in term_frequency.items()))
                try:
                    summary_set = sc.MDS_sparse(candidate_set, NUMBER_SUMMARY_SET_ELEMENT,
                                                LAMBDA, TSTOP, MAX_CONSE_REJ)
                    summary_text = util.summary_vector_to_text_as_list(summary_set, term_frequency)
                    rouge_1_fscores = 0
                    rouge_2_fscores = 0
                    for summary_ref in reference_summaries:
                        rouge_1_fscore = eval.RougeFScore(summary_text, summary_ref, Level.Rouge_1)
                        rouge_1_fscores += rouge_1_fscore

                        rouge_2_fscore = eval.RougeFScore(summary_text, summary_ref, Level.Rouge_2)
                        rouge_2_fscores += rouge_2_fscore

                    print("Rouge-1 Fscore : ", rouge_1_fscores / 5)
                    print("Rouge-2 Fscore : ", rouge_2_fscores / 5)
                except ValueError:
                    print("Sample larger than population")

