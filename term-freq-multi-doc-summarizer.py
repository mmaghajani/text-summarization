import os
import sparse_coding as sc
import evaluation as eval
import numpy as np
from evaluation import Level
import utils as util


NUMBER_SUMMARY_SET_ELEMENT = 10
LAMBDA = 3
TSTOP = 0.001
MAX_CONSE_REJ = 100


rootdir = "data/Multi"
for i in range(1,9):
    subDir = os.path.join(rootdir, "Track"+str(i)+"/Source/")
    for root, dirs, files in os.walk(subDir):
        if dirs:
            for dir in dirs:
                sentences, words = util.read_documents(subDir+dir+"/")
                reference_summaries = util.read_multi_ref_summaries(dir)
                term_frequency = util.make_term_frequency(sentences, words)
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
