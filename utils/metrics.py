def compute_precision(ypred, ytrue):
    nsamples = len(ypred)
    assert nsamples == len(ytrue)

    tp = sum([1 for i in range(nsamples) if ytrue[i] == 1 and ypred[i] == 1])
    fp = sum([1 for i in range(nsamples) if ytrue[i] == 0 and ypred[i] == 1])
    return tp / (tp + fp)


def compute_recall(ypred, ytrue):
    nsamples = len(ypred)
    assert nsamples == len(ytrue)

    tp = sum([1 for i in range(nsamples) if ytrue[i] == 1 and ypred[i] == 1])
    fn = sum([1 for i in range(nsamples) if ytrue[i] == 1 and ypred[i] == 0])
    return tp / (tp + fn)


def f1_score(ypred, ytrue):
    precision = compute_precision(ypred, ytrue)
    recall = compute_recall(ypred, ytrue)
    return 2 * (precision * recall) / (precision + recall)


def all_scores(ypred, ytrue):
    precision = compute_precision(ypred, ytrue)
    recall = compute_recall(ypred, ytrue)
    return precision, recall, 2 * (precision * recall) / (precision + recall)
