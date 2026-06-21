The hypothesis-boosting problem asks whether a black-box learning algorithm that only slightly outperforms random guessing can be turned into a classifier with arbitrarily low error. The difficulty is not simply collecting many weak hypotheses; if each weak learner is trained on the same original distribution, it will keep discovering the same easy patterns and keep missing the same hard region. A majority vote over correlated errors does not improve the error. The key leverage is distributional control: each new weak learner should be forced to focus on the examples that the current ensemble still gets wrong.

Earlier approaches demonstrated that weak and strong learnability are equivalent, but they had practical limitations. Recursive majority circuits reduce error through a sequence of filtered distributions, yet they treat every subcall with a worst-case error bound and cannot reward a weak hypothesis that performs much better than expected. A later flat majority construction uses a sharper schedule, but it requires a fixed edge parameter known in advance and weights every weak hypothesis equally in the final vote. A useful boosting algorithm must instead measure the actual weighted error of each returned hypothesis and use that measurement both to reweight the next distribution and to decide how much that hypothesis counts in the final vote.

The method is AdaBoost, the adaptive boosting algorithm. It maintains a distribution over training examples and iteratively trains weak learners on the current distribution. After each round, it computes the weighted error of the returned hypothesis, assigns it a confidence coefficient based on that error, and then updates the example weights so that correctly classified examples become less important while misclassified examples remain important. The final classifier is a weighted majority vote of all weak hypotheses, where stronger hypotheses receive larger weights.

More concretely, in the signed-label setting with labels in {-1, +1}, AdaBoost starts with a uniform distribution over examples. At round t, it trains a weak hypothesis h_t on the current distribution D_t and measures its weighted error epsilon_t. The confidence coefficient is alpha_t = (1/2) log((1 - epsilon_t) / epsilon_t). A smaller error gives a larger coefficient, so a strong weak learner contributes more to the final vote. The distribution is updated as D_{t+1}(i) proportional to D_t(i) exp(-alpha_t y_i h_t(x_i)). Correct examples, where y_i h_t(x_i) = +1, are downweighted by exp(-alpha_t), while incorrect examples are upweighted by exp(alpha_t). The final classifier predicts the sign of the accumulated score F(x) = sum_t alpha_t h_t(x). In the equivalent {0, 1} form, beta_t = epsilon_t / (1 - epsilon_t), and the update multiplies correct-example weights by beta_t while leaving incorrect-example weights unchanged; the final vote coefficient is log(1 / beta_t).

The algorithm is adaptive in two ways. First, it never needs to know the weak learner's edge in advance; each round's observed error sets the next reweighting strength and the next vote weight. Second, if a particular distribution is easy and yields a very accurate weak hypothesis, AdaBoost exploits that by giving that hypothesis large influence. If another distribution is hard and yields only a barely better-than-chance hypothesis, the algorithm barely shifts the distribution and gives the hypothesis near-zero voting weight. The training-error bound is the product over rounds of 2 sqrt(epsilon_t (1 - epsilon_t)); writing epsilon_t = 1/2 - gamma_t, this is at most exp(-2 sum_t gamma_t^2), showing exponential decay in the accumulated squared edges.

```python
import numpy as np


def adaboost(X, y, rounds, weak_learner_factory):
    # y must be in {-1, +1}; weak learner must support sample_weight.
    m = len(y)
    w = np.full(m, 1.0 / m)
    hypotheses = []
    alphas = []

    for _ in range(rounds):
        h = weak_learner_factory().fit(X, y, sample_weight=w)
        pred = h.predict(X)
        err = float(np.sum(w * (pred != y)))

        if err <= 0.0:
            hypotheses.append(h)
            alphas.append(np.inf)
            break
        if err >= 0.5:
            # No positive binary edge: flip externally if valid, otherwise reject/stop.
            break

        alpha = 0.5 * np.log((1.0 - err) / err)
        w *= np.exp(-alpha * y * pred)
        w /= np.sum(w)

        hypotheses.append(h)
        alphas.append(alpha)

    return hypotheses, np.array(alphas)


def predict(hypotheses, alphas, X):
    score = np.zeros(X.shape[0])
    for alpha, h in zip(alphas, hypotheses):
        if np.isinf(alpha):
            return h.predict(X)
        score += alpha * h.predict(X)
    return np.where(score >= 0, 1, -1)
```
