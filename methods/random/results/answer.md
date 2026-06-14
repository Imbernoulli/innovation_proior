# Random sampling (passive learning baseline), distilled

Random sampling is the passive, non-adaptive query strategy for pool-based active learning: on
each round it selects `n` inputs uniformly at random, without replacement, from the
currently-unlabeled pool, sends them to the oracle for labeling, and lets the fixed harness
retrain. It consults no model, no labels-seen-so-far, and no notion of uncertainty. That
refusal to use information is the point: it is the unbiased, assumption-free reference against
which every adaptive query strategy is measured.

## Problem it solves

A large unlabeled pool of `N` inputs is available cheaply; labeling is the expensive step and
the label budget is small, spent in rounds (select a batch, label, retrain, repeat). The task
is to choose which `n` unlabeled inputs to label each round so the model reaches the best
accuracy for the fewest labels. Random sampling answers the prior question — what is the
principled *baseline* selection rule — rather than trying to be the cleverest one.

## Key idea

Draw a uniform simple-random-sample-without-replacement (SRS-WOR) of size `n` from the
currently-unlabeled indices. Why this and not something model-aware:

- **Equal inclusion probability `pi_i = n/N`.** For the current unlabeled set `U`, write
  `N = |U|`. Under SRS-WOR, unit `i` appears in `C(N-1,n-1)` of the `C(N,n)` possible samples,
  so `pi_i = C(N-1,n-1)/C(N,n) = n/N`. Every unlabeled input is equally likely to be picked,
  so the unweighted labeled subset is an unbiased, representative miniature of the pool: the
  Horvitz-Thompson estimator `(1/N) sum_{i in S} g(x_i)/pi_i` is unbiased for any population
  average, and with equal `pi_i` it collapses to the plain sample mean.
- **Preserves the distributional premise of generalization theory.** The PAC/VC bridge from
  training error to test error — a consistent hypothesis on `m` examples has `err <= eps` w.p.
  `1-delta` once `m = O((1/eps)(d ln(1/eps) + ln(1/delta)))` — is stated for examples drawn
  from the deployment distribution `P`. If the pool is drawn from `P`, value-independent
  uniform selection keeps the labeled subset distributed like that pool, hence like `P` up to
  the finite-pool approximation. Any rule that biases selection toward a model-chosen region
  must justify giving up that premise.
- **Without replacement.** A repeated label buys zero information, and WOR has strictly lower
  estimator variance than with-replacement. For finite-population variance `sigma^2` with
  denominator `N`, the sample-mean variance is:
  `Var_WOR = (sigma^2/n)(N-n)/(N-1) <= sigma^2/n = Var_WR`.
- **Restricted to the unlabeled set.** Labeled inputs are already bought; spending the budget
  on them buys nothing.

## Known limitations (why a smarter rule might be wanted)

- **Rare-class starvation.** Faithfully reproducing `P` reproduces class imbalance: if the
  positive class is ~1 in 1000, a budget of 500 labels yields ~500 negatives and essentially no
  positives, training nothing useful for the minority class.
- **Diminishing information.** Let `R` be the region where hypotheses consistent with the
  labeled data still disagree, and `alpha = Pr_{x~P}[x in R]`. Only points in `R` can change
  the model. As labels accumulate `R` shrinks and `alpha -> 0`, but uniform sampling draws from
  all of `P` obliviously, so a fraction `alpha` of draws land somewhere informative and the
  information per label decays toward zero. In the clean threshold-on-an-interval case, random
  examples need `O((1/eps) ln(1/eps))` labels to reach error `eps` while chosen queries (binary
  search) need `O(ln(1/eps))` — an exponential gap, the prize adaptivity chases.

The adaptive alternatives buy past these but pay for it: selective/uncertainty sampling biases
the labeled set off `P` (losing the i.i.d. guarantee and risking a self-reinforcing bias from a
bad early model); membership-query synthesis assumes an oracle for arbitrary, possibly
un-labelable, synthesized inputs. Random sampling keeps the unbiasedness and the
assumption-free guarantee for free, which is exactly why it is the yardstick.

## Evaluation role

Active learners are compared by *learning curves*: accuracy (or another metric) vs. number of
labeled inputs over the rounds. An adaptive strategy is judged "better" only if its learning
curve dominates the random-sampling curve across the budget range. Common pool-based metrics are
final-round test accuracy at a fixed budget and the area under the learning curve (sample
efficiency).

## Batching

Because the rule consults no model, drawing `n` at once has the same law as drawing one,
marking it labeled, retraining, and drawing again until the batch is filled. The picks inside
the batch are not independent; they are one size-`n` without-replacement draw, which is exactly
what prevents duplicates. The per-round batch size is therefore free of the batch-diversity
concerns that adaptive batch methods must engineer around.

## Final algorithm

```
input: unlabeled mask idxs_lb over a pool of size n_pool, batch size n
1. U <- { pool indices i : idxs_lb[i] is False }     # still-unlabeled inputs
2. draw a uniformly random SRS-WOR of size n from U   # equal inclusion prob n/|U|
3. return those n indices                              # oracle labels them; harness retrains
```

## Working code

Filling the `query` slot of the pool-based `Strategy` harness, matching the task edit:

```python
import numpy as np


class CustomSampling(Strategy):
    """Random sampling baseline -- selects samples uniformly at random."""

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        # unlabeled pool indices
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        # uniform SRS-without-replacement of size n:
        # random permutation => every size-n subset equally likely and distinct,
        # so each unlabeled input has inclusion probability n / len(idxs_unlabeled).
        return idxs_unlabeled[np.random.permutation(len(idxs_unlabeled))][:n]
```
