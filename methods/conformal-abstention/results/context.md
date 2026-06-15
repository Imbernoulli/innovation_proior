# Context: deciding when a fixed classifier should abstain (circa 2017-2021)

## Research question

We are handed a *fixed*, already-trained classifier `f` for a high-stakes tabular decision —
credit, recidivism risk, admissions — and a downstream reviewer that can take over on hard
cases at some cost. The classifier emits, for each input `x`, a vector of class scores
`f(x) in [0,1]^K` (softmax / posterior estimates). We cannot retrain it, and we cannot change
the train/calibration/test pipeline. The only thing we get to design is the **acceptance
rule**: given a target coverage `c` (the fraction of the test stream we are willing to keep),
decide which test examples the model answers itself and which it *defers* to the reviewer.

What a good rule must achieve is sharper than "abstain when unsure". The model's score vector
is only a *heuristic* notion of confidence: it may be badly miscalibrated, overfit, or simply
wrong, and we have no parametric model of the data. So the rule must turn that heuristic score
into a cutoff with a **rigorous, finite-sample, distribution-free** statement about how often
future exchangeable points clear it, computed offline on modest compute and stated at the
user-specified target coverage. Selective risk on the accepted examples and subgroup deferral
gaps remain important evaluation metrics, but a label-free confidence cutoff does not by
itself certify them. The pain point is that the natural thing one does today — pick a cutoff
on the model's confidence — can match a calibration fraction yet still give no exact
finite-sample accounting for the fresh-test acceptance probability.

## Background

The landscape rests on a few load-bearing ideas.

**The reject option / error-reject tradeoff.** The classical formulation of "let a classifier
abstain" goes back to Chow (1970), *On optimum recognition error and reject tradeoff*. If the
true posteriors `P(Y=j | x)` were known, the Bayes-optimal rule for a fixed cost of rejection
is to *reject* (abstain on) an input when its maximum posterior `max_j P(Y=j | x)` falls below
a threshold `t`, and otherwise predict the argmax. Chow showed this rule is optimal in a
precise sense: for the reject rate induced by `t`, no other rule attains a lower error rate on
the accepted region, and sweeping `t` traces the optimal error-reject curve. The catch baked
in from the start: this is optimal *only if the posteriors are exact*. With an estimated,
possibly miscalibrated `f(x)` in place of the true posterior, thresholding `max_j f(x)_j`
is a heuristic with no guarantee.

**Risk-coverage / selective classification.** El-Yaniv & Wiener (2010), *On the foundations of
noise-free selective classification*, formalized abstention for modern predictors as a
**selective classifier**: a pair `(f, g)` where `g(x) in {0,1}` is a selection (accept) function.
Two quantities trade off — the **coverage** `phi = E[g(X)]` (fraction accepted) and the
**selective risk** `R = E[loss · g(X)] / E[g(X)]` (error on the accepted set). For deep nets,
Geifman & El-Yaniv (2017) took `g` to threshold the **softmax response** (SR), the maximum
softmax value, and reported smooth risk-coverage curves: a single confidence threshold yields
a usable accept/defer rule. The recurring fact about all of these: they rank examples well in
practice, but the threshold is chosen by *looking at* a held-out risk-coverage curve and
picking a point, with no finite-sample certificate that the chosen operating point will hold
on the test stream. The selective risk and fresh-test coverage they actually deliver are
whatever they turn out to be.

**Exchangeability and order statistics.** The other strand is a distribution-free idea from
classical statistics. A sequence of random variables is **exchangeable** if its joint law is
invariant to permutation; i.i.d. is a special case. The key consequence used everywhere below:
if `Z_1,...,Z_{n+1}` are exchangeable real scalars (assume no ties), then the rank of
`Z_{n+1}` among all `n+1` of them is **uniformly distributed on {1,...,n+1}** — every ordering
is equally likely, so the new point is equally likely to land in any of the `n+1` gaps. This
turns a statement about an unknown distribution into pure combinatorics. The same fact powered
classical **tolerance intervals** (Wilks 1941; Wald 1943; Tukey 1947): the order statistics of
a sample carve out a region whose coverage probability is governed by a Beta law, because for a
uniform sample the order statistics are Beta-distributed — coverage controlled with no model of
the data. This distribution-free tradition is exactly what is missing from the reject-option /
softmax-response threshold: those build the cutoff from a heuristic score with no finite-sample
coverage statement attached.

**Marginal versus conditional coverage.** A guarantee can be *marginal* — averaged over all
inputs — or *conditional* — required to hold within each subgroup `g` separately. For
prediction sets this is usually phrased as true-label inclusion; for an abstention score it
becomes the analogous question of whether the accept/defer event clears the target rate
overall or within each group. A marginal promise can be met while one subgroup is
systematically worse off: all the failures, or all the deferrals, can concentrate in a single
group and the average still looks fine. This gap is the reason a rule that controls only the
average behaves unevenly across subgroups under shift, and it is a known limitation, not
something one observes only after the fact.

## Baselines

**Confidence / softmax-response thresholding (Chow 1970; Geifman & El-Yaniv 2017).** Take the
model's top score `s_conf(x) = max_j f(x)_j` as a confidence signal, pick a single global
threshold `t`, accept when `s_conf(x) >= t`, defer otherwise. To hit a target acceptance rate
`c`, set `t` to the empirical `(1-c)`-quantile of `s_conf` on a held-out set so that a `c`
fraction clears it. Core idea, exactly the reject option with estimated posteriors. **Gap:**
the threshold is fit to an *empirical* quantile of a *heuristic* score, so the rule comes with
no exact finite-sample, distribution-free statement about the fresh acceptance probability,
and there is no `n`-dependent correction acknowledging that `t` was estimated from finite
calibration data. It also says nothing about the selective risk of the accepted examples or
how deferrals distribute across subgroups.

**Learned deferral / learning-to-defer (Mozannar & Sontag 2020; Cortes, DeSalvo, Mohri 2016).**
Train a compact meta-model to predict whether the base classifier will be wrong on a given
example (or to jointly optimize a classifier-plus-rejector with a consistent surrogate loss),
then defer the examples it flags as likely errors. **Gap:** it needs its own training, can
itself overfit or be miscalibrated, and inherits any distribution shift between its training
data and deployment; like confidence thresholding it offers no distribution-free accounting
of the fresh acceptance event, and its performance is only as good as the meta-model's own
generalization.

**Per-subgroup thresholding (group-stratified reject option).** Run the confidence-threshold
rule separately within each subgroup, tuning each group's cutoff to hit the target acceptance
rate on that group. This equalizes acceptance across groups by construction. **Gap:** it
requires the subgroup label at decision time and enough calibration data *per group* for each
empirical quantile to be stable; small groups give noisy thresholds, and like the global
version each per-group cutoff is still a heuristic empirical quantile with no finite-sample
rank correction.

The shared limitation across all three: every one of them sets a cutoff on a score and then
*hopes* the operating point holds, with no statement that survives the fact that the cutoff was
estimated from a finite calibration sample, and no acceptance-rate guarantee that is agnostic
to the model and the distribution.

## Evaluation settings

The natural yardsticks are cached high-stakes tabular datasets and a fixed pipeline:

- **Adult** (UCI / AIF360 Census income) — binary income prediction; subgroup attributes sex,
  race.
- **COMPAS** (ProPublica recidivism) — binary recidivism risk; subgroup attributes race, sex.
- **Law School** (admissions/GPA, outcome binarized at the training-set median) — subgroup
  attributes race, gender.

Each dataset is split into **train / calibration / test**. A base classifier is trained on
train and frozen; the acceptance rule is fit using only calibration-time base-model
probabilities, labels, and subgroup ids; it is then applied to test. The metrics that would
be the yardstick (settings, not results): selective risk at a fixed target coverage (error
among accepted examples), worst-subgroup selective risk, the deferral-rate gap across
subgroups (max minus min deferral rate), and the AUROC of the acceptance score as a predictor
of base-model correctness. The target acceptance rate (e.g. 80%) is the operating point at
which the rule is compared.

## Code framework

The rule plugs into a fixed offline harness: a base classifier is already trained and frozen,
the data is already split, and the harness hands the policy the calibration-time probabilities,
labels, and subgroup ids to `fit`, then calls the policy on the test set. Nothing about *how*
the acceptance threshold is set from the calibration scores is settled yet — that rule is
exactly what is to be designed — so the substrate is only the generic selective-policy
interface that already exists, with one empty slot for the calibration logic.

```python
import numpy as np

TARGET_COVERAGE_DEFAULT = 0.8


class SelectivePolicy:
    """Accept/defer rule for a fixed base classifier. Fits on calibration-time
    base-model probabilities/labels/subgroups; applied to test. The base
    classifier and the train/calibration/test split are fixed and not editable."""

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT,
                 random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.threshold_: float = 0.5
        self.group_thresholds_: dict[int, float] = {}
        self.meta_model_ = None
        self.strategy_name = "selective_policy"

    def fit(self, probs: np.ndarray, y_true: np.ndarray,
            groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        # probs: (n, n_classes) calibration base-model probabilities
        # y_true: (n,) calibration labels;  groups: (n,) subgroup ids
        # TODO: the calibration rule we will design -- turn the calibration
        #       scores into the acceptance threshold(s) this policy will use.
        pass

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray,
                         X: np.ndarray | None = None) -> np.ndarray:
        # Higher score = more confident -> more likely to accept.
        # TODO: the per-example confidence signal we will use.
        pass

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray,
                       X: np.ndarray | None = None) -> np.ndarray:
        # Boolean array: True = accept, False = defer.
        return self.acceptance_score(probs, groups, X) >= self.threshold_

    def calibration_summary(self) -> dict[str, float]:
        return {"threshold": float(self.threshold_)}
```

The harness supplies the calibration probabilities/labels/groups once to `fit`; the policy
then exposes a per-example acceptance score and the accept/defer decision. The single empty
slot is the rule that converts calibration scores into the threshold.
