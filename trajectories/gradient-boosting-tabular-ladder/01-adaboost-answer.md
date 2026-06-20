**Problem (from prior art).** Weak tree learners can be combined into a strong committee by training
each new learner on a reweighting of the data that emphasizes the committee's mistakes — but the early
boosting constructions needed the weak learner's error bounded *in advance* and used heavy filtering
schemes. Two holes remain: how much to reweight each round, and how much each learner's vote counts,
without knowing the weak learner's edge ahead of time.

**Key idea.** **AdaBoost** (adaptive boosting): maintain a distribution over examples; each round, fit
the weak learner on the weighted data, measure its weighted error εₜ, and let *both* the reweighting
and the vote weight be computed from εₜ. Down-weight correctly classified examples by βₜ = εₜ/(1−εₜ)
(so misclassified ones gain relative mass after renormalizing), and give the learner vote weight
αₜ = log((1−εₜ)/εₜ). For K classes, add a +log(K−1) term so αₜ is positive exactly when the learner
beats the random bar εₜ < (K−1)/K. The final classifier is the weighted vote
H(x) = argmax_y Σₜ αₜ·[hₜ(x) = y].

**Why it works.** βₜ and αₜ are not independent choices. Choosing βₜ = εₜ/(1−εₜ) makes the just-used
learner have weighted error 1/2 under the next distribution, and in the weighted-vote error bound it
minimizes the per-round factor Zₜ/√βₜ with Zₜ = (1−εₜ)βₜ + εₜ. The error bound shrinks geometrically
while every weak learner clears the random bar. The reweighting is *adaptive*: it is computed from the
realized εₜ, so no advance
knowledge of the weak learner's edge is needed and no reweighting schedule must be hand-set. Its limit:
the whole scheme is wired to a 0/1 "right or wrong" indicator and an implicit exponential loss — it
knows the *sign* of an error, not *how wrong*, so it does not extend to arbitrary differentiable losses
or regression.

**Change / code.** The per-round discrete (SAMME) boost step: weighted fit, weighted error, the αₜ vote
weight with the log(K−1) multiclass term, multiplicative reweight of the misclassified examples. Real
code from scikit-learn `sklearn/ensemble/_weight_boosting.py` (`AdaBoostClassifier._boost_discrete`,
tag 1.3.2):

```python
def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
    """Implement a single boost using the SAMME discrete algorithm."""
    estimator = self._make_estimator(random_state=random_state)
    estimator.fit(X, y, sample_weight=sample_weight)
    y_predict = estimator.predict(X)

    if iboost == 0:
        self.classes_ = getattr(estimator, "classes_", None)
        self.n_classes_ = len(self.classes_)

    # Instances incorrectly classified
    incorrect = y_predict != y

    # Error fraction (weighted by the current distribution)
    estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

    # Stop if classification is perfect
    if estimator_error <= 0:
        return sample_weight, 1.0, 0.0

    n_classes = self.n_classes_

    # Stop if the error is at least as bad as random guessing
    if estimator_error >= 1.0 - (1.0 / n_classes):
        self.estimators_.pop(-1)
        if len(self.estimators_) == 0:
            raise ValueError(
                "BaseClassifier in AdaBoostClassifier ensemble is worse than random, "
                "ensemble can not be fit."
            )
        return None, None, None

    # Boost weight using multi-class AdaBoost SAMME alg
    estimator_weight = self.learning_rate * (
        np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0)
    )

    # Only boost the weights if it will fit again
    if not iboost == self.n_estimators - 1:
        # Only boost positive weights
        sample_weight = np.exp(
            np.log(sample_weight) + estimator_weight * incorrect * (sample_weight > 0)
        )

    return sample_weight, estimator_weight, estimator_error
```

Sample weights are renormalized to a distribution at the top of each round in the caller
(`BaseWeightBoosting.fit`). The binary AdaBoost of Freund & Schapire (1997) is the K=2 special case
(log(K−1)=0), with weight update wᵢ ← wᵢ·βₜ^([hₜ(xᵢ)=yᵢ]) and vote weight αₜ = log(1/βₜ).
