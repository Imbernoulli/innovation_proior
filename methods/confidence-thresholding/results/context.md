## Research question

A classifier is already trained and fixed. It maps an input `x` to a class and, on most
modern models, also emits a vector of nonnegative scores over the classes that sum to one
(the softmax layer). In production this classifier will sometimes be wrong, and in a
mission-critical setting a wrong answer can be far more costly than no answer: a misread
autopilot scene, a missed tumor, a bad loan. The natural escape is to let the system *abstain*
on the cases it is least sure of and hand those off to a human or a slower backup, while still
answering on as many cases as possible.

That immediately poses a trade-off with two knobs. The fraction of inputs the system actually
answers on is its *coverage*; the error rate measured only over those answered inputs is its
*selective risk*. Abstaining on more inputs lowers coverage but should also lower the
selective risk, because the inputs you drop are the ones you were least sure of. The precise
problem: given the fixed classifier and a held-out labeled set, build an acceptance rule that,
at a *target coverage*, answers on that fraction of inputs while keeping the selective risk on
the answered set as low as possible — and do it offline, on modest compute, without retraining
the classifier. A solution needs (i) some per-input quantity that orders inputs from
most-trustworthy to least, and (ii) a principled way to turn that ordering into a concrete
accept/defer decision that hits the requested coverage. The acceptance rule, not the
classifier, is the object being designed.

## Background

The idea of a classifier with a *reject option* is old — about six decades by this point. The
seminal analysis is Chow's: assume the underlying class-posterior probabilities `P(y | x)` are
fully known, and ask for the rule that minimizes error at a fixed reject rate (or vice versa)
under 0/1 loss. The answer is *ambiguity rejection* — reject an input exactly when none of the
posteriors is dominant enough, i.e. when `max_y P(y | x)` falls below a threshold. In the
cost-based form the threshold is fixed by the costs of erroring, rejecting, and being correct
(`t = (C_r - C_c)/(C_e - C_c)`). Two structural facts from that analysis carry forward: the
optimal decision thresholds the *maximum posterior*, and the error and reject rates move
monotonically against each other as the threshold sweeps — there is a single trade-off curve.

Through the intervening decades the reject option was implemented inside many specific learning
schemes — SVMs, boosting, nearest-neighbors — almost always on an "ambiguity / lack of
confidence" principle: *when in doubt, refuse to classify*. El-Yaniv & Wiener (2010) gave this
folklore a clean formal frame, which is the language used throughout below. A *selective
classifier* is a pair `(f, g)`: a standard classifier `f` and a *selection function*
`g : X → {0,1}` that qualifies it,

```
(f, g)(x) = f(x)        if g(x) = 1     (accept / predict)
            don't-know  if g(x) = 0     (reject / defer)
```

Its two summary numbers are *coverage* `Φ(f,g) = E[g(X)]`, the probability mass it answers on,
and *selective risk*

```
R(f, g) = E[ ℓ(f(X), Y) · g(X) ] / Φ(f, g),
```

the average loss restricted to the answered region (this reduces to ordinary risk when
`g ≡ 1`). Risk as a function of coverage is the *risk-coverage (RC) curve*, the complete
performance profile of an abstaining classifier and the central object of study. El-Yaniv &
Wiener characterized the achievable region of this curve and, for the special *noise-free*
case where a perfect hypothesis exists, gave an optimal strategy (reject any point not
classified unanimously by the version space). They explicitly noted that this theoretically
optimal selection is *not* obtained by thresholding soft classification scores — which they
describe as "the commonly used heuristic." So at this point thresholding a confidence score is
the known, widely-used practical recipe, but it sits without a guarantee, and the
version-space construction that does have a guarantee assumes a perfect, noise-free
hypothesis — an assumption that does not hold for a deep net trained on noisy data, where the
version space is neither finite nor tractable.

Two empirical facts about *modern* trained networks set up everything that follows. First, the
maximum softmax probability separates correct from incorrect predictions surprisingly well:
correctly classified test inputs tend to receive a higher `max_y softmax(x)_y` than
misclassified or out-of-distribution ones, so the statistic, viewed as a detector of "this
prediction is wrong," scores well above chance across vision, language, and speech models.
Second, and in tension with the first, that same softmax maximum is *not* a calibrated
probability of correctness: networks are routinely overconfident, the mean predicted
probability on *wrong* examples still runs high (often 0.7-0.9), and pure Gaussian-noise inputs
can elicit a 0.9 "confidence" from an MNIST model. The softmax score is therefore untrustworthy
as an *absolute* probability but informative as a *relative* ordering. What remains unresolved is
how to convert such a signal into a dependable accept/defer rule. There is also a competing
source of a per-input reliability signal: Monte-Carlo dropout (Gal & Ghahramani 2016) leaves
dropout active at test time and runs several stochastic forward passes, taking the variance of
the predicted-class response as an uncertainty estimate — a more expensive signal requiring
many passes per input.

## Baselines

These are the prior acceptance/reject mechanisms a new rule is measured against and reacts to.

**Chow's Bayes-optimal reject rule (Chow 1957, 1970).** With the true posteriors known and
0/1 loss, accept iff `max_y P(y | x) ≥ 1 - t` and reject otherwise; the cost-based threshold is
`t = (C_r - C_c)/(C_e - C_c)`. This is the theoretical ideal and the origin of "threshold the
maximum posterior." **Gap:** it presumes the posteriors `P(y | x)` are known exactly, which a
finite-sample, miscalibrated trained model does not deliver; and the cost-based threshold
presumes you can quantify misclassification and abstention costs in commensurable units, which
is exactly what is hard to do in the mission-critical applications that most want abstention.

**Cost-based reject for neural networks (Cordella et al. 1995; De Stefano et al. 2000).** The
prior NN-specific instantiation: derive a reliability signal from the network's output layer
and reject below a cut, but inside a cost-driven objective — specify the cost of a
misclassification and the cost of an abstention, then tune the rejection mechanism to optimize
that cost. **Gap:** it requires meaningful, explicit costs. In settings like disengaging an autopilot or deferring a diagnosis, assigning a
numeric cost to "abstain" versus "err" is itself ill-posed, so the cost knob is the wrong
control surface; what a practitioner actually has is a tolerable error level or an affordable
coverage, not a cost ratio.

**Consistent selective strategy / version-space rejection (El-Yaniv & Wiener 2010).** In the
realizable, noise-free case, pick any classifier consistent with the sample and reject every
point not labeled unanimously by all consistent hypotheses; this attains zero selective risk
with provably maximal coverage. **Gap:** it is defined for finite or otherwise tractable
hypothesis classes and a noise-free target. For a deep network there is no enumerable version
space to test unanimity against, and real data is not realizable, so the construction does not
transfer; the 2010 analysis flags that the practical alternative people actually use —
thresholding a soft score — falls outside its optimality result.

**Maximum-softmax-probability detector (Hendrycks & Gimpel 2016).** Use `max_y softmax(x)_y`
directly as a score for "this prediction is reliable," and rank inputs by it; correct
predictions score higher than incorrect ones. Quality is read off threshold-free, via the
AUROC of the score as a predictor of correctness. **Gap:** as posed it is an *analysis* of a
score's discriminative power (does the score separate right from wrong?), not a constructed
acceptance rule tied to an operating point — it stops short of converting the ranking into a
selection function that meets a specified coverage or a specified risk, and it documents that
the score is uncalibrated, leaving open how to set a defensible cut.

**MC-dropout uncertainty (Gal & Ghahramani 2016).** Run several dropout-on forward passes and
take the variance of the predicted-class response as uncertainty; the negative of that is a
reliability score that can likewise be thresholded. **Gap:** it costs many forward passes per
input (a Monte-Carlo estimate whose quality grows with the number of samples), heavy at
inference time, and it is a uncertainty heuristic with no built-in tie to a target error or
coverage.

## Evaluation settings

The natural yardsticks already in use:

- **The risk-coverage curve.** Sweep the acceptance threshold over a held-out labeled set and
  plot selective risk against coverage; the whole curve characterizes an acceptance rule.
  Lower curves are better, and a single scalar summary is the area under it.
- **AUROC of the acceptance score as a predictor of correctness.** Treat each answered input as
  positive if the classifier got it right and negative if wrong, and measure the area under the
  ROC of the score; this equals `P(score on a correct input > score on a wrong input)`,
  a threshold-independent measure of how well the score *ranks* reliability. Chance is 0.5.
- **Selective error at a fixed coverage.** With 0/1 loss, the selective risk at a chosen
  operating coverage (e.g. answer on 80% of inputs) is the headline number; lower is better.
- **Datasets and protocol.** Standard image classification benchmarks (CIFAR-10, CIFAR-100,
  ImageNet) with off-the-shelf trained architectures (e.g. VGG-16, ResNet-50), and
  high-stakes tabular datasets where abstention matters. The data is split so the classifier is
  trained on one part, the acceptance rule is fit on a separate held-out calibration part, and
  risk/coverage are reported on an untouched test part — the threshold must never be tuned on
  the data used to judge it. A small confidence level `δ` is fixed when a high-probability
  guarantee is wanted.

## Code framework

The acceptance rule plugs into a fixed-classifier harness. The classifier is already trained
and its per-class softmax outputs are available. The editable object is the policy that, from
calibration-time probabilities/labels/groups, learns how to accept or defer at a target
coverage, then applies that decision at test time. None of the internals of the policy are
settled — what evidence it extracts from the available outputs, and how it turns that evidence
into a decision, are exactly what is to be designed — so the substrate is only the generic
interface and the quantities that already exist (the softmax matrix, the labels, a
target-coverage knob, and NumPy array primitives).

```python
import numpy as np

TARGET_COVERAGE_DEFAULT = 0.8


class SelectivePolicy:
    """Policy that maps calibration outputs to accept / defer decisions."""

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT,
                 random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.threshold_: float = 0.5
        self.group_thresholds_: dict[int, float] = {}
        self.meta_model_ = None
        self.strategy_name = "candidate_policy"

    def fit(self, probs: np.ndarray, y_true: np.ndarray,
            groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        # probs:  (n, n_classes) calibration-time base-model softmax probabilities
        # y_true: (n,) calibration labels
        # groups: (n,) integer subgroup ids
        # X:      optional raw features
        # TODO: the fitting rule we will design.
        pass

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray,
                         X: np.ndarray | None = None) -> np.ndarray:
        # Higher score = more confident -> more likely to accept.
        # TODO: the confidence-ranking signal we will design.
        pass

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray,
                       X: np.ndarray | None = None) -> np.ndarray:
        # Boolean array: True = accept (predict), False = defer (abstain).
        # TODO: turn the fitted state into an accept/defer decision.
        pass

    def calibration_summary(self) -> dict[str, float]:
        # TODO: report fitted policy state for logging.
        pass


# existing evaluation harness the policy plugs into (fixed, not editable)
def evaluate(policy, probs_test, y_test, groups_test):
    accept = policy.predict_accept(probs_test, groups_test)   # per-input accept/defer
    preds = np.argmax(probs_test, axis=1)                     # the fixed classifier's call
    correct = (preds == y_test)
    coverage = accept.mean()                                  # fraction answered
    selective_risk = (~correct[accept]).mean() if accept.any() else 1.0
    return coverage, selective_risk
```

The harness supplies the softmax matrix and labels; the body of `SelectivePolicy` — the score
and the decision rule — is the empty slot to be filled.
