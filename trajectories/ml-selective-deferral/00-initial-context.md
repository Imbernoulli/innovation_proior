## Research question

High-stakes tabular decisions — credit, recidivism risk, admissions — run a fixed classifier in front
of a human reviewer. At a **target coverage** (here 80%) the system must answer on most test cases and
**defer** the rest to the reviewer. The base classifier and the entire train / calibration / test
pipeline are frozen; the single thing being designed is the **acceptance rule** — given a test point's
base-model probabilities (and, optionally, its subgroup id and raw features), decide *accept* or
*defer*. A good rule must do four things at once, and they pull against each other:

- keep **selective risk** (error on accepted points) low at the target coverage,
- not concentrate deferrals on one subgroup (small **deferral-rate gap**),
- keep **worst-group selective risk** low under subgroup shift,
- preserve the acceptance score's **AUROC** as a correctness-ranking signal.

The hard part is the third and fourth goals together: under subgroup shift a naive confidence threshold
can defer disproportionately on one group while leaving others under-covered, so "good on average" and
"good per group" are different objectives.

## Prior art before the first rung (selective-prediction lineage)

The first rung reacts to a sixty-year line of abstention rules; these precede the ladder and define the
vocabulary the scaffold uses.

- **Chow's reject option (Chow 1970).** With the *true* posterior `P(y|x)` and a fixed cost for a wrong
  prediction versus a deferral, the Bayes-optimal abstention rule rejects exactly when the maximum class
  posterior `max_y P(y|x)` falls below a threshold, and the error/reject rates trade off monotonically
  along one curve. Gap: it assumes the true posterior and a known cost; a deployed net gives neither.
- **Softmax response selective classification (El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017).**
  Substitute the model's `max_j softmax(x)_j` for Chow's posterior and threshold it; sweeping the
  threshold traces the risk-coverage curve. Practical and training-free, and across many models the
  max-softmax separates correct from wrong predictions well above chance. Gap: the cutoff is an
  empirical choice on a heuristic score — no finite-sample statement that a fresh point clears it at the
  target rate, and nothing about subgroups.
- **Calibration critique (Guo et al. 2017).** Modern nets are over-confident: the softmax max is *not*
  a calibrated probability. This sinks any reading of `max softmax` as `P(correct)` — but the selection
  rule only ever needs the score's *ranking*, not its absolute value, so the critique constrains how the
  score may be used, not whether it can be used.
- **Learning with a reject option / learning to defer (Cortes, DeSalvo & Mohri 2016; Mozannar & Sontag
  2020).** Instead of thresholding a fixed score, *learn* the abstention decision against a combined
  predictor-plus-expert loss. Gap: the consistent versions retrain the predictor jointly with a
  reject head — which this task forbids, since the base classifier is frozen and the only learnable
  object is a post-hoc policy on calibration outputs.
- **Subgroup fairness of selective prediction.** A single global cutoff controls only the *marginal*
  accept event; it does not equalize per-group coverage or per-group risk, and under subgroup shift it
  can over-defer the worst group. This is the gap the strongest rung targets directly.

## The fixed substrate

Everything except the acceptance rule is frozen and must not be touched. For each dataset (Adult,
COMPAS, Law-School-GPA, all cached from AIF360, with two protected attributes binned into integer
subgroup ids) the loop: stratifies a train/test split, carves the train half into a **fit** set and a
**calibration** set (75/25), trains a fixed base model — `StandardScaler → GradientBoostingClassifier`
(200 trees, depth 3, lr 0.1) — on the fit set, and produces `predict_proba` matrices on calibration and
test. It then constructs the policy, calls `fit(cal_probs, y[cal], groups[cal], X[cal])`, and at test
time reads `predict_accept(...)` and `acceptance_score(...)`. The metrics are computed *for me*: at the
accepted set, `selective_risk_at80` (error among accepted), `worst_group_selective_risk` (worst-subgroup
error among accepted), `deferral_rate_gap` (max minus min subgroup deferral rate), and `auroc` (of the
acceptance score against per-point correctness). The loop also exposes a helper, `_confidence_features`,
that turns a probability matrix (and optional groups/X) into a small feature table — `[p1, max_prob,
margin, entropy, group, X[:,0]]` — that a policy may use to build a learned score.

## The editable interface

Exactly one region is editable — the `SelectivePolicy` class in `scikit-learn/custom_selective.py`
(lines 253–287). Every rung on the ladder is a fill of this same contract. The policy sees only what the
loop hands it: calibration probabilities, labels, integer subgroup ids, and (optionally) raw features at
`fit`; probabilities, groups, and features (never labels) at decision time.

```python
class SelectivePolicy:
    """Policy that maps calibration outputs to accept / defer decisions.

    The default implementation is intentionally conservative:
    it accepts the top-confidence examples needed to reach the target coverage.
    Baselines replace this class with more specialized policies.
    """

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.threshold_: float = 0.5
        self.group_thresholds_: dict[int, float] = {}
        self.meta_model_ = None
        self.strategy_name = "global_threshold"

    def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        scores = self.acceptance_score(probs, groups, X)
        quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, quantile))
        self.group_thresholds_ = {}
        self.meta_model_ = None
        return self

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        return np.max(probs, axis=1)

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        scores = self.acceptance_score(probs, groups, X)
        return scores >= self.threshold_

    def calibration_summary(self) -> dict[str, float]:
        return {
            "threshold": float(self.threshold_),
        }
```

The starting point is the scaffold default above: score by `max(probs)`, fit one global threshold at the
`(1 − target_coverage)` calibration quantile, accept above it. Each rung replaces exactly this class and
nothing else.

## Evaluation settings

Three datasets — **Adult** (income; subgroups sex×race), **COMPAS** (recidivism; race×sex), and
**Law-School-GPA** (outcome binarized at the train-set median; race×gender), the last hidden. One seed
(42). Target coverage 0.80 throughout. Four metrics: `selective_risk_at80`, `worst_group_selective_risk`,
`deferral_rate_gap` (all lower is better) and `auroc` (higher is better).
