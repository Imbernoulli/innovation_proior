# Turning classifier scores into calibrated probabilities (circa late 1990s)

## Research question

A trained binary classifier hands me a real-valued score `f(x)` for each input, and
`sign(f(x))` gives a good *decision*. But the score itself is not a probability. I want
`P(y = +1 | x)` — a number that means what it says: of all the inputs the
model rates at confidence `p`, about a fraction `p` should actually belong to the class.

This matters whenever the classification is one piece of a larger decision and the outputs
must be *combined* or *acted on under a cost*. A Viterbi/HMM word recognizer composing
phoneme posteriors, a Bayes-optimal decision that trades off a known utility, a
cost-sensitive choice where a false positive and a false negative carry different prices — all
of these need a calibrated `P(y = +1 | x)`, not just a ranking.

The score-producing classifier is fixed: it was trained to minimize a *margin* or *accuracy*
objective, and I do not want to retrain it. The question is how to learn a mapping from the
raw scores to probabilities using a limited amount of labelled data.

## Background

By the late 1990s the dominant high-accuracy classifiers are large-margin kernel machines.
A support vector machine (Vapnik) outputs `f(x) = h(x) + b` with
`h(x) = Σ_i y_i α_i k(x_i, x)`, trained by minimizing a regularized hinge loss,

```
C Σ_i (1 − y_i f_i)_+  +  (1/2) ||h||²_F ,
```

the hinge term `(1 − y f)_+` penalizing margin violations and the RKHS-norm term controlling
capacity. Minimizing this bounds the test misclassification rate and yields a
*sparse* machine — only a subset of training points (the support vectors) carry nonzero `α_i`.
Other strong classifiers of the era — boosted ensembles, decision trees, naive Bayes — each
produce their own real-valued scores that rank examples well.

The geometry of these scores is informative. If I look at the
class-conditional distributions of the SVM output — `p(f | y = +1)` and
`p(f | y = −1)` — for a real linear SVM (the histograms can be read off cross-validated
scores), they are emphatically *not* Gaussian. They have kinks: the derivative of each density
is discontinuous at the margins `f = +1` and `f = −1`, which is unsurprising because the
training cost `(1 − y f)_+` itself is non-smooth exactly there. And between the two margins,
the densities fall off roughly *exponentially*.

Several conceptual frames are on the table. The **generative** frame: estimate the
class-conditional densities `p(f | y)` and combine them by Bayes' rule,
`P(y=1|f) = p(f|y=1)P(y=1) / Σ_i p(f|y=i)P(y=i)`. The **discriminative** frame: skip the
densities and model the posterior `P(y=1|f)` directly with a parametric family fit to labelled
data. The **regularized-likelihood** frame: replace the classifier's training objective with
one whose output *is* a probability (a logistic link plus a norm penalty). And a recurring
background idea from smoothing of empirical frequencies: a raw count ratio `k/n` is a
high-variance estimate of a probability (it saturates at 0 and 1), and the classical Bayesian
cure is to shrink it away from the boundary with Laplace's `(k+1)/(n+2)` rule of succession.

A second background fact: the relevant decision threshold need not sit at `f = 0`. The
Bayes-optimal threshold depends on the class priors `P(y=−1)/P(y=+1)` and on the loss; when the
priors are skewed, the point where the true posterior crosses `0.5` is shifted away from the
classifier's natural `f = 0` boundary.

## Baselines

**Direct regularized maximum likelihood (Wahba, 1992/1999).** Instead of post-processing,
train the kernel machine itself with a logistic link `P(y=1|x) = 1/(1 + exp(−f(x)))` and
minimize a penalized negative log multinomial likelihood,

```
−(1/m) Σ_i [ ((y_i+1)/2) log p_i + ((1−y_i)/2) log(1 − p_i) ]  +  λ ||h||²_F ,
```

so the output `p(x)` is a posterior probability by construction. Core idea: bake calibration
into training via the right loss.

**Gaussian class-conditional fit (Hastie & Tibshirani, 1996/1998).** Stay generative: fit a
Gaussian to each class-conditional density `p(f | y = ±1)`. With a single *tied* variance for
both Gaussians, Bayes' rule turns the posterior into a sigmoid in `f`, and the bias is then
adjusted so that `P(y=1|f) = 0.5` lands at `f = 0`. Core idea: two Gaussians ⇒ a logistic
posterior, one parameter estimated generatively from the variances. With *untied* variances the
posterior becomes `P(y=1|f) = 1/(1 + exp(a f² + b f + c))`.

**Cosine-expansion posterior (Vapnik, 1998).** Decompose the feature space into the direction
orthogonal to the separating hyperplane (parameterized by a scaled `t`) and the remaining
directions (a vector `u`), and fit
`P(y=1|t,u) = a_0(u) + Σ_{n=1}^N a_n(u) cos(n t)`, the coefficients minimizing a regularized
functional. Core idea: a flexible expansion that can depend on the full feature vector.

## Evaluation settings

The natural yardsticks are fixed by the frozen-classifier setting.

- **Score-producing classifiers, frozen.** Representative base learners trained on standard
  benchmarks: an SVM (linear and quadratic-kernel) on text categorization (Reuters), the UCI
  Adult census-income task, and web-page categorization. The quadratic kernels use
  `k(x_i,x_j) = ((x_i·x_j + 1)/c)²` with the normalizing constant `c` set from the average
  self-dot-product of the data, keeping the kernel in a reasonable range. The classifier and
  its hyperparameters are fixed; only the score-to-probability map is fit.
- **Data splits.** Because the classifier's own training scores are biased estimates of its
  out-of-sample scores, the calibration map must be fit on data the classifier did not train on:
  a held-out fraction (commonly ~30%) or `k`-fold cross-validation that produces an out-of-fold
  score for every training point. The test set is held back for evaluation only.
- **Metrics.** For checking *decisions*: misclassification count, with McNemar's test for paired
  significance. For checking *probabilities*: the negative log likelihood (cross-entropy) of the
  held-out labels under the calibrated probabilities, with a paired signed-rank test; reliability
  (binned agreement between predicted confidence and empirical accuracy); and squared-error
  (Brier-style) distance between the predicted positive-class probability and the binary label. All are
  computed on data disjoint from both classifier and calibrator training.

## Code framework

The calibrator is a small trainable object that plugs in *after* a frozen binary classifier.
The base classifier, the data pipeline, the train/calibration/test splits, optional example
weights, and an off-the-shelf scalar optimizer all already exist. The empty slot is the
score-to-probability map itself: the finite state it stores, the objective it fits on the
calibration split, and the probability it returns for new scalar scores. The scaffold is a bare
`fit`/`predict_proba` contract over that split.

```python
import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator


class CalibrationMethod(BaseEstimator):
    """Post-hoc calibrator: learn a map from a frozen classifier's confidence
    scores to calibrated probabilities, using only a held-out calibration set.

    fit(scores, labels, sample_weight=None):
        scores : (n,) real-valued confidence scores from the frozen classifier
        labels : (n,) binary labels, encoded as positive vs negative
        sample_weight : optional nonnegative calibration-example weights
    predict_proba(scores):
        returns calibrated positive-class probabilities.
    """

    def __init__(self, max_abs_prediction_threshold=30.0):
        self.max_abs_prediction_threshold = max_abs_prediction_threshold
        # TODO: state the learned map needs.
        pass

    def fit(self, scores, labels, sample_weight=None):
        # TODO: learn the score -> probability map on the calibration set.
        return self

    def predict_proba(self, scores):
        # TODO: apply the learned map; keep the output a valid probability.
        pass


# already-existing harness the calibrator plugs into
def calibrate_and_score(base_clf, X_cal, y_cal, X_test, calibrator):
    cal_scores  = base_clf.decision_function(X_cal) # frozen classifier's scores
    calibrator.fit(cal_scores, y_cal)               # fit the map on held-out data
    test_scores = base_clf.decision_function(X_test)
    return calibrator.predict_proba(test_scores)    # calibrated test probabilities
```

The frozen classifier supplies scores; `fit` sees only the calibration split; `predict_proba`
is where the learned map will live.
