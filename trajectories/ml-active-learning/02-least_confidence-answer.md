**Problem.** Random sampling spends the budget representatively — on the bulk the model already handles —
while the examples that would actually move the model, the contested ones near the boundary, almost never
get drawn, and its information per label decays as the region of uncertainty shrinks. I want to *choose*
the informative examples using the model the harness already trains, scored as a sortable scalar so I can
spend a budget of `n`.

**Key idea (uncertainty sampling).** Only examples in the region of uncertainty `R` (where consistent
hypotheses disagree) inform the learner. Synthesizing the ideal query gives unlabelable gibberish and
isn't even possible here (`query` returns pool indices); computing `R` exactly needs the whole version
space; a committee needs noise-free realizability and extra models. But the single probabilistic
classifier already reports "I'm on the fence" as its posterior: score each unlabeled row by its top-class
probability `max_y p(y|x)` and query the `n` smallest — equivalently the `n` maximizing `1 − max_y p(y|x)`,
which in the binary case is the posterior nearest 0.5 and in general is the model's own estimated
probability of mislabeling `x`.

**Why this over margin/entropy.** All three coincide in the binary case (monotone in `|p−0.5|`) and differ
only for ≥3 classes; least confidence is the minimal form, scoring nothing but the quantity
`self.predict_prob` already exposes. Margin/entropy are the refinements to try if it leaves value behind.

**Hyperparameters.** None. One `predict_prob` forward pass per round; cost is the pass the loop already does.

**Step-2 edit.** Replace the random body: get the unlabeled rows' posteriors, take the per-row max, sort
ascending, return the first `n`.

**What to watch.** Expect the largest gain on letter (where random was weakest — 26 classes, decaying `α`),
the smallest or even a slight loss on spambase (near-solved balanced binary, thin contested region), a
modest gain on splice. Risk: scoring points in isolation can return a redundant batch of near-duplicate
boundary points — if gains undershoot, that is the culprit, and the next rung must make uncertainty
distribution-aware.

```python
# EDITABLE region of badge/query_strategies/custom_sampling.py (lines 28-54) — step 2: least confidence
class CustomSampling(Strategy):
    """Least Confidence (Uncertainty Sampling) — selects samples with lowest
    maximum predicted probability."""

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], np.asarray(self.Y)[idxs_unlabeled])
        U = probs.max(1)[0]
        return idxs_unlabeled[U.sort()[1][:n]]
```
