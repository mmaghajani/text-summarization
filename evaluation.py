def collect_pairs(lines):
    token = []
    for line in lines:
        words = line.split()
        for i in range(len(words) - 1):
            token.append(words[i] + " " + words[i + 1])

    return set(token)


def Rouge(candidate_summary, reference_summary, n, mode):  # mode = precision or recall
    coOccurrings = 0
    if n == 2:
        bigrams = collect_pairs(reference_summary)
        for summary in candidate_summary:
            for bigram in bigrams:
                coOccurrings += int(summary.count(bigram))
        if mode == "precision":
            return coOccurrings / len(collect_pairs(candidate_summary))
        elif mode == "recall":
            return coOccurrings / len(bigrams)

    elif n == 1:
        splited = []
        for summary in reference_summary:
            splited += summary.split()
        unigrams = set(splited)
        for summary in candidate_summary:
            for unigram in unigrams:
                coOccurrings += int(summary.count(unigram))
        if mode == "precision":
            tmp = []
            for summary in candidate_summary:
                tmp += summary.split()
            return coOccurrings / len(set(tmp))
        elif mode == "recall":
            return coOccurrings / len(unigrams)


def RougeFScore(candidate_summary, reference_summary, n):
    Precision = Rouge(candidate_summary, reference_summary, n, "precision")
    Recall = Rouge(candidate_summary, reference_summary, n, "recall")

    return 2 * (Precision * Recall) / (Precision + Recall)


# candidate_summary = ['او نیامده خواهد آمد', 'من به زودی میروم']
# reference_summary = ['او نیامده', 'او زودی نیامده']
# print(RougeFScore(candidate_summary, reference_summary, 2))
