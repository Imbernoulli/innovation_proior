## Research question

Pool-based active learning for tabular classification: a huge pile of unlabeled inputs sits there for
free, but each label costs real money or expert time. I get a fixed budget, spent in rounds — each
round I pick a batch of `n` still-unlabeled inputs, pay to label them, fold them into the training set,
the harness retrains, repeat for 20 rounds. The single thing I design is the **batch acquisition rule**:
given the current model and the pool, which `n` unlabeled examples do I send to the oracle. Everything
else — the model family, the optimizer, the retrain-each-round loop, the data handling — is fixed. A
strong rule trades off uncertainty (where is the model unsure), diversity (don't buy `n` near-duplicates),
representativeness (cover the pool), and information gain (which labels most shrink the model's error).
The contribution is the rule itself, not preprocessing or training-loop changes.

## Prior art before the first rung (the passive-learning lineage)

The first rung is the passive baseline — the rule that uses *no* model, *no* labels-seen-so-far, *no*
notion of uncertainty. It is the resolution of a much older line of work on "how do I pick a subset of a
population that estimates the whole population well," and the ladder above it is the line of work on "how
do I do strictly better by *using* the model." These are the methods the first rung reacts to:

- **PAC learning / uniform convergence (Valiant 1984; Blumer–Ehrenfeucht–Haussler–Warmuth 1989).** The
  bridge from training error to test error — a hypothesis consistent with `m` labeled examples has
  `err ≤ ε` with confidence `1−δ` once `m = O((1/ε)(d ln(1/ε) + ln(1/δ)))` — rests on exactly one
  premise: the training examples are an i.i.d. sample from the deployment distribution `P`. Gap: it says
  *nothing* about which examples to pick beyond "draw them like `P`"; the budget is spent blindly.
- **Survey sampling (Cochran, *Sampling Techniques*).** Simple random sampling without replacement gives
  every unit inclusion probability `n/N`, under which the plain sample mean is unbiased for the population
  mean (Horvitz–Thompson collapses), and without-replacement strictly cuts variance by the
  finite-population correction `(N−n)/(N−1)`. Gap: representativeness reproduces the pool's imbalance —
  if the class I care about is rare, a representative draw starves it.
- **The region of uncertainty (Cohn–Atlas–Ladner 1994).** Only examples on which two still-consistent
  hypotheses disagree can move the model; the mass of that region, `α`, decays toward zero as labels
  accumulate, so a rule oblivious to the model spends ever more of its budget re-confirming settled
  territory. Gap: a passive rule's information per label decays — in the cleanest case the gap between
  random draws and chosen queries is `1/ε` vs `ln(1/ε)`, exponential.

The fixed substrate below is what these converged to: a pool, a model retrained each round, and one
empty acquisition slot.

## The fixed substrate

A pool-based active-learning loop is frozen and must not be touched. It owns `n_pool` tabular inputs
`self.X` (numpy `[n_pool, n_features]`) with their hidden labels `self.Y` (LongTensor `[n_pool]`), a
boolean mask `self.idxs_lb` of which are already labeled, and a small neural classifier `self.clf`. Each
round it calls the acquisition rule for `n` indices, reveals those labels, flips the mask, retrains
`self.clf` from scratch on the enlarged labeled set (`Strategy.train`, Adam, cross-entropy, trained to
near-zero training error), and evaluates. The base `Strategy` class also hands the rule everything it
could need to *read* the current model without retraining:

- `self.predict_prob(X, Y)` — softmax probabilities `[len(X), n_classes]`.
- `self.predict_prob_dropout_split(X, Y, n_drop)` — `n_drop` stochastic (dropout-on) forward passes,
  `[n_drop, len(X), n_classes]`.
- `self.get_embedding(X, Y, return_probs=)` — penultimate-layer embeddings `[len(X), emb_dim]` (and the
  softmax if asked).
- `self.get_grad_embedding(X, Y)` — last-layer gradient embeddings `[len(X), emb_dim·n_classes]`
  (gradient of the cross-entropy at the model's predicted label).
- `self.get_exp_grad_embedding(X, Y)` — per-class Fisher embeddings `[len(X), n_classes, emb_dim·n_classes]`,
  each class scaled by `√p`.

## The editable interface

Exactly one region is editable — the `CustomSampling` class in
`badge/query_strategies/custom_sampling.py`, lines 28–54. Every method on the ladder is a fill of the
same contract: a `query(self, n)` that returns an `np.ndarray` of `n` indices into `self.X`, chosen from
the *currently unlabeled* pool (`self.idxs_lb` False). It may add helper methods, but `query(n)` is the
entry point the loop calls. `numpy as np` and the `Strategy` base are already imported at module scope.

The starting point is the scaffold default: **uniform random sampling**. Each later method replaces
exactly this class body and nothing else.

```python
# EDITABLE region of badge/query_strategies/custom_sampling.py (lines 28-54) — default fill
class CustomSampling(Strategy):
    """Custom active learning query strategy.

    Must implement query(n) -> np.ndarray of n indices from the unlabeled pool.
    You may add helper methods, but query(n) is the entry point called by the
    active learning loop.
    """

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        """Select n samples from the unlabeled pool to label next.

        Args:
            n: number of samples to select

        Returns:
            np.ndarray of n indices (into self.X) of selected unlabeled samples
        """
        # Default: random sampling (replace with a better strategy)
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        return idxs_unlabeled[np.random.permutation(len(idxs_unlabeled))][:n]
```

## Evaluation settings

Three OpenML tabular classification datasets spanning sizes and difficulties — **letter** recognition
(26-class, the largest pool), **spambase** (binary), and **splice** (3-class). Protocol: 20 rounds of
batch active learning, the model retrained from scratch after each round, a fixed total label budget.
Two metrics, higher is better on both:

- `accuracy` — test accuracy after the final round (final-budget quality).
- `auc` — area under the learning curve (accuracy vs. number of labeled samples) across all 20 rounds,
  capturing sample efficiency — how fast accuracy rises per label, not just where it ends.
