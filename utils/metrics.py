def compute_precision(ypred, ytrue, epsilon=1e-6):
    nsamples = len(ypred)
    assert nsamples == len(ytrue)

    tp = sum([1 for i in range(nsamples) if ytrue[i] == 1 and ypred[i] == 1])
    fp = sum([1 for i in range(nsamples) if ytrue[i] == 0 and ypred[i] == 1])
    return tp / (tp + fp + epsilon)


def compute_recall(ypred, ytrue, epsilon=1e-6):
    nsamples = len(ypred)
    assert nsamples == len(ytrue)

    tp = sum([1 for i in range(nsamples) if ytrue[i] == 1 and ypred[i] == 1])
    fn = sum([1 for i in range(nsamples) if ytrue[i] == 1 and ypred[i] == 0])
    return tp / (tp + fn + epsilon)


def f1_score(ypred, ytrue, epsilon=1e-6):
    precision = compute_precision(ypred, ytrue)
    recall = compute_recall(ypred, ytrue)
    return 2 * (precision * recall) / (precision + recall + epsilon)


def all_scores(ypred, ytrue, epsilon=1e-6):
    precision = compute_precision(ypred, ytrue)
    recall = compute_recall(ypred, ytrue)
    return precision, recall, 2 * (precision * recall) / (precision + recall + epsilon)
