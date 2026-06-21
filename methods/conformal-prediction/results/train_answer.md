The problem is how to turn a predictor that ranks candidates well into a prediction that carries a finite-sample reliability guarantee. A classifier, regressor, or nearest-neighbor rule may order labels plausibly, but its raw score is a modeling claim, not a calibrated probability. If the likelihood, posterior, variance model, or asymptotic approximation is wrong, the stated uncertainty can be overconfident. Classical confidence intervals are exact only under strong parametric assumptions, and validation quantiles do not by themselves justify coverage for a future example unless the future error is exchangeable with past errors. We need a wrapper that separates the predictive heuristic from the coverage proof.

The key observation is that exchangeability is enough. If the observed examples and the next example are exchangeable, then conditional on the unordered bag of examples no position is special. A symmetric score that measures how strange each labeled example looks relative to the others can therefore be used to test candidate labels without knowing the true data-generating distribution.

The method is conformal prediction. In its full form, choose a permutation-symmetric nonconformity measure that assigns a larger value to a labeled example that fits the rest of the bag worse. For a new input and a candidate label, provisionally complete the sample, compute the nonconformity score of every example in the completed bag, and compute the candidate-label p-value as the fraction of completed-sample scores at least as large as the candidate's score. At significance level eps, include the candidate label in the prediction set when its p-value is larger than eps. If the completed examples are exchangeable, the probability that the true label is excluded is at most eps.

In practice the full form can be expensive, because testing many candidate labels may require recomputing scores many times. The split form is the usual compromise. First train or choose a score function using data that will not be used for calibration. Then compute scores on a held-out calibration set. Take the calibrated threshold as the kth smallest element of the multiset of calibration scores augmented with positive infinity, where k = ceil((n+1)(1-alpha)). For a new input, return every label whose score is no larger than that threshold. Under exchangeability of the calibration and future examples, after the score is fixed, the prediction set covers the true label with probability at least 1-alpha.

The proof is a permutation-rank argument. The true future score is exchangeable with the calibration scores, so it can land in an extreme rank only as often as symmetry permits. The score affects efficiency, not validity: a good score yields small, informative sets, while a poor score yields large or unhelpful sets under the same coverage guarantee. The guarantee is marginal over the exchangeable draw, not conditional on a fixed input, and it does not by itself ensure calibration within subgroups.

```python
import numpy as np
from typing import Callable

def split_conformal(
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    alpha: float = 0.1,
    randomized: bool = False,
    allow_empty: bool = False,
):
    """
    Fit a split-conformal predictor.

    score_fn(x, y) -> nonconformity scores, lower is more conforming.
    X_cal, Y_cal: calibration data.
    alpha: miscoverage level (target error rate).
    randomized: if True, randomize at the threshold for exact size control.
    allow_empty: if False, force a prediction by setting the threshold to max.
    """
    n = len(Y_cal)
    cal_scores = score_fn(X_cal, Y_cal)
    k = int(np.ceil((n + 1) * (1 - alpha)))

    sorted_scores = np.sort(cal_scores)
    if k <= n:
        qhat = sorted_scores[k - 1]
    else:
        qhat = np.inf

    if not allow_empty and qhat == np.inf:
        qhat = np.max(cal_scores)

    predictor = {
        "score_fn": score_fn,
        "qhat": qhat,
        "randomized": randomized,
        "eta": None,
    }

    if randomized:
        if k <= n:
            count_at_threshold = np.sum(cal_scores == qhat)
            eta = (
                (n + 1) * alpha - n + np.sum(cal_scores < qhat)
            ) / count_at_threshold
            eta = np.clip(eta, 0.0, 1.0)
        else:
            eta = 0.0
        predictor["eta"] = eta

    return predictor


def predict_set(
    predictor: dict,
    X: np.ndarray,
    candidate_labels: np.ndarray,
):
    """
    Return the conformal prediction set for each input in X.

    candidate_labels: array of shape (num_candidates,) with all possible labels.
    Returns a list of arrays, one prediction set per input.
    """
    score_fn = predictor["score_fn"]
    qhat = predictor["qhat"]
    randomized = predictor["randomized"]
    eta = predictor["eta"]

    sets = []
    for x in X:
        x_batch = np.tile(x, (len(candidate_labels), 1))
        scores = score_fn(x_batch, candidate_labels)
        included = scores <= qhat
        if randomized and eta is not None and np.any(scores == qhat):
            tie_mask = scores == qhat
            tie_idx = np.where(tie_mask)[0]
            np.random.shuffle(tie_idx)
            keep = int(np.floor(eta * len(tie_idx)))
            included[tie_idx[keep:]] = False
        sets.append(candidate_labels[included])
    return sets


# Example: conformalized nearest-neighbor classification.
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestNeighbors

    X, y = make_classification(n_samples=500, n_features=10, n_classes=3, n_informative=8)
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4)
    X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5)

    nn = NearestNeighbors(n_neighbors=5).fit(X_train)
    labels = np.unique(y)

    def nn_score(x_batch, y_batch):
        # One minus average label agreement with 5 nearest training neighbors.
        dists, idx = nn.kneighbors(x_batch)
        neighbor_labels = y_train[idx]
        scores = 1.0 - np.mean(neighbor_labels == y_batch[:, None], axis=1)
        return scores

    predictor = split_conformal(nn_score, X_cal, y_cal, alpha=0.1)
    pred_sets = predict_set(predictor, X_test, labels)
    coverage = np.mean([y_test[i] in pred_sets[i] for i in range(len(y_test))])
    avg_size = np.mean([len(s) for s in pred_sets])
    print(f"Empirical coverage: {coverage:.3f}")
    print(f"Average set size: {avg_size:.3f}")
```
