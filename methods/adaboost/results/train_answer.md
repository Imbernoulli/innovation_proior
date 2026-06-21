The research question is whether a learning algorithm that is only guaranteed to do slightly better than random guessing on every distribution can be amplified into one that achieves arbitrarily small error. A naive first attempt is to call the weak learner many times on the original training set and take a majority vote, but that fails because every call sees the same distribution and can keep making the same mistakes; correlated errors do not cancel. The earliest positive answer used a recursive majority-of-three construction that does force disagreement, yet it builds a deep circuit, charges every sub-call at a single worst-case error level, and cannot take advantage of a round that happens to perform much better than the worst case. The next improvement flattened the recursion into one majority vote and used a binomial-tail weighting schedule, but that schedule still required the weak-learning edge to be fixed before the run started, and it counted every hypothesis equally in the final vote. A practical procedure needs to be driven by what actually happens each round.

AdaBoost, short for Adaptive Boosting, solves this by measuring each round's weighted error and letting that single number control both the next distribution and the hypothesis's vote weight. It maintains a distribution over training examples. At each round it hands the current distribution to the weak learner, gets back a hypothesis, and computes its weighted error eps on that distribution. If eps is at least one half, the round is useless and the process stops. Otherwise it sets beta = eps / (1 - eps) and alpha = (1/2) log((1 - eps) / eps). The weights of correctly classified examples are multiplied by beta, so they shrink, while misclassified examples keep their weight; the weights are then renormalized into a new distribution. This forces the next weak learner to focus on the examples the current committee is getting wrong. In the final classifier each hypothesis is weighted by alpha, and the prediction is the sign of the weighted sum of signed votes.

The constants are not guessed. They come from minimizing the per-round factor in a training-error bound. Writing the update as w_i^{t+1} = w_i^t beta_t^{1 - |h_t(x_i) - y_i|} and tracking the total weight gives an upper bound on the surviving weight, while any example the final vote gets wrong contributes a lower bound. Squeezing the two bounds gives error <= product_t [eps_t + (1 - eps_t) beta_t] / sqrt(beta_t), and minimizing each factor independently yields beta_t = eps_t / (1 - eps_t). Substituting back gives the clean result that training error is at most product_t 2 sqrt(eps_t (1 - eps_t)), which is at most exp(-2 sum_t gamma_t^2) when eps_t = 1/2 - gamma_t. If every round has edge at least gamma, a small number of rounds drives the training error below any target epsilon, and the algorithm never needs to know gamma in advance.

The same machinery extends to regression as AdaBoost.R2. The obstacle is that right/wrong is binary, while regression errors are real-valued and unbounded. The fix is to normalize the per-example absolute error by the largest error in the current round, producing a loss L_i in [0, 1]. The weighted average loss Lbar plays the same role as eps. If Lbar is at least one half the round is discarded; otherwise beta = Lbar / (1 - Lbar), weights are updated by beta^{1 - L_i}, and the round's confidence is log(1 / beta). Because a weighted mean would be dragged arbitrarily far by one bad learner, the predictions are combined by a weighted median: sort the predictions for an input and return the one at which the cumulative confidence first reaches half the total.

A shallow decision tree, typically a stump or depth-2/3 tree, is used as the weak learner. It only needs to clear the better-than-chance bar, and keeping it weak leaves room for later rounds to find fresh edges. A learning-rate shrinkage factor can be folded into alpha to trade step size for more rounds. The result is an adaptive ensemble whose reweighting strength and vote weights are both set on the fly by the measured performance of each round.

```python
import numpy as np


class AdaBoost:
    """Discrete classification boosting plus bounded-loss regression boosting."""

    def __init__(self, make_weak_learner, task_type="classification",
                 n_rounds=200, learning_rate=1.0, loss="linear", random_state=None):
        self.make_weak_learner = make_weak_learner
        self.task_type = task_type
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state
        self.learners_, self.estimator_weights_, self.estimator_errors_ = [], [], []

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        w = np.ones(n, dtype=float) / n
        if self.task_type == "classification":
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)

        for t in range(self.n_rounds):
            learner = self.make_weak_learner()

            if self.task_type == "regression":
                p = w / w.sum()
                idx = rng.choice(np.arange(n), size=n, replace=True, p=p)
                learner.fit(X[idx], y[idx])
            else:
                learner.fit(X, y, sample_weight=w)

            pred = learner.predict(X)
            p = w / w.sum()

            if self.task_type == "classification":
                incorrect = (pred != y)
                err = float(np.average(incorrect, weights=p))
                if err <= 0:
                    self.learners_.append(learner)
                    self.estimator_weights_.append(1.0)
                    self.estimator_errors_.append(0.0)
                    break
                if err >= 1.0 - 1.0 / n_classes:
                    break
                learner_weight = self.learning_rate * (
                    np.log((1.0 - err) / err) + np.log(n_classes - 1.0)
                )
                if t != self.n_rounds - 1:
                    w = np.exp(np.log(w) + learner_weight * incorrect * (w > 0))
            else:
                mask = w > 0
                loss_vec = np.abs(pred[mask] - y[mask])
                loss_max = loss_vec.max()
                if loss_max != 0:
                    loss_vec = loss_vec / loss_max
                if self.loss == "square":
                    loss_vec = loss_vec ** 2
                elif self.loss == "exponential":
                    loss_vec = 1.0 - np.exp(-loss_vec)
                err = float(np.dot(p[mask], loss_vec))
                if err <= 0:
                    self.learners_.append(learner)
                    self.estimator_weights_.append(1.0)
                    self.estimator_errors_.append(0.0)
                    break
                if err >= 0.5:
                    break
                beta = err / (1.0 - err)
                learner_weight = self.learning_rate * np.log(1.0 / beta)
                if t != self.n_rounds - 1:
                    w[mask] *= np.power(beta, (1.0 - loss_vec) * self.learning_rate)

            w = w / w.sum()
            self.learners_.append(learner)
            self.estimator_weights_.append(float(learner_weight))
            self.estimator_errors_.append(float(err))
        return self

    def predict(self, X):
        weights = np.asarray(self.estimator_weights_, dtype=float)
        if self.task_type == "classification":
            votes = np.zeros((len(X), len(self.classes_)))
            for weight, learner in zip(weights, self.learners_):
                pred = learner.predict(X)
                for j, cls in enumerate(self.classes_):
                    votes[:, j] += weight * (pred == cls)
            return self.classes_[np.argmax(votes, axis=1)]

        preds = np.array([learner.predict(X) for learner in self.learners_]).T
        out = np.empty(preds.shape[0])
        for i in range(preds.shape[0]):
            order = np.argsort(preds[i])
            cdf = np.cumsum(weights[order])
            j = np.searchsorted(cdf, 0.5 * cdf[-1])
            out[i] = preds[i, order[min(j, len(order) - 1)]]
        return out
```
