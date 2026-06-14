**Problem.** Pool-based active learning: labels cost money, unlabeled inputs are free. Each round I pick
`n` of the unlabeled pool to label, the harness retrains, repeat for 20 rounds. The acquisition rule is
the only knob. Before using the model at all I need the floor — the model-free rule whose statistical
behavior I can state without conditions, the yardstick every adaptive rule must beat.

**Key idea (the passive baseline).** Uniform random sampling without replacement over the unlabeled pool.
Make the selection independent of the data values, so the labeled set stays distributed like the
deployment distribution `P` — the exact premise the PAC bridge from training to test error needs. Under
simple random sampling every unlabeled input has inclusion probability `n/(#unlabeled)`; with equal
inclusion the plain sample mean is unbiased for pool averages (Horvitz–Thompson collapses), and
without-replacement cuts variance by the finite-population correction `(N−n)/(N−1)`.

**Why this is the floor.** It uses no model, no labels-seen-so-far, no uncertainty — precisely what makes
it unbiased and assumption-free, and precisely what makes it weak: it reproduces a rare class's imbalance
(starving the minority) and its information per label decays as the region of uncertainty shrinks while it
keeps drawing obliviously from all of `P`. In the cleanest case that non-adaptivity is the exponential
`1/ε`-vs-`ln(1/ε)` gap between random draws and chosen queries.

**Step-1 edit.** Leave `query` at the scaffold default: take the unlabeled indices, permute, take `n`.

**What to watch.** Expect random to be hardest to beat on the easy balanced binary problem (spambase) and
easiest to beat where many classes and a thin budget make most draws land in settled territory (letter).
That failure forces a model-aware rule at step 2.

```python
# EDITABLE region of badge/query_strategies/custom_sampling.py (lines 28-54) — step 1: random
class CustomSampling(Strategy):
    """Random sampling baseline — selects samples uniformly at random."""

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        return idxs_unlabeled[np.random.permutation(len(idxs_unlabeled))][:n]
```
