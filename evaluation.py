from enum import Enum


class Level(Enum):
    Rouge_1 = 1
    Rouge_2 = 2


def collect_pairs(lines):
    token = []
    for line in lines:
        words = line.split()
        for i in range(len(words) - 1):
            token.append(words[i] + " " + words[i + 1])

    return set(token)


def Rouge(candidate_summary, reference_summary, level, mode):  # mode = precision or recall
    coOccurrings = []
    if level == Level.Rouge_2:
        bigrams = collect_pairs(reference_summary)
        for summary in candidate_summary:
            for bigram in bigrams:
                if bigram in summary:
                    coOccurrings.append(bigram)

        if mode == "precision":
            return len(set(coOccurrings)) / len(collect_pairs(candidate_summary))
        elif mode == "recall":
            return len(set(coOccurrings)) / len(bigrams)

    elif level == Level.Rouge_1:
        splited = []
        for summary in reference_summary:
            splited += summary.split()
        unigrams = set(splited)
        for summary in candidate_summary:
            for unigram in unigrams:
                if unigram in summary:
                    coOccurrings.append(unigram)

        if mode == "precision":
            tmp = []
            for summary in candidate_summary:
                tmp += summary.split()
            return len(set(coOccurrings)) / len(set(tmp))
        elif mode == "recall":
            return len(set(coOccurrings)) / len(unigrams)


def RougeFScore(candidate_summary, reference_summary, n):
    Precision = Rouge(candidate_summary, reference_summary, n, "precision")
    Recall = Rouge(candidate_summary, reference_summary, n, "recall")

    if not (Precision + Recall == 0):
        return 2 * (Precision * Recall) / (Precision + Recall)
    else:
        print("No CoOccurrings")
        return 0
