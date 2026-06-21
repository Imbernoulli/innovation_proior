## Research question

Pool-based active learning for tabular classification: a large pool of unlabeled inputs is free, but each label costs money or expert time. The budget is spent in rounds — each round the rule selects a batch of `n` still-unlabeled examples, the oracle reveals their labels, they are folded into the training set, and the classifier is retrained from scratch. The only design choice is the **batch acquisition rule**: given the current model and the unlabeled pool, which `n` examples do I label next? The contribution is the rule itself; nothing else in the loop changes.

## Prior art / Background / Baselines

- **PAC / uniform-convergence bounds.** They guarantee that a hypothesis consistent with an i.i.d. training sample of size `m` generalizes with error `ε` and confidence `1−δ`, assuming examples are drawn like the deployment distribution.
- **Survey sampling / simple random sampling.** Every unit receives the same inclusion probability, producing an unbiased estimate of population statistics.
- **Region of uncertainty / query by disagreement.** Only examples on which competing consistent hypotheses disagree can change the model, and the volume of that region shrinks as labels arrive.

## Fixed substrate / Code framework

The loop is frozen. It owns `n_pool` tabular inputs `self.X` (`[n_pool, n_features]`), hidden labels `self.Y`, a boolean mask `self.idxs_lb` of already-labeled examples, and a small neural classifier `self.clf`. Each round it calls the acquisition rule for `n` indices, reveals those labels, flips the mask, retrains `self.clf` from scratch on the enlarged labeled set (`Strategy.train`, Adam, cross-entropy, trained to near-zero training error), and evaluates. The base `Strategy` exposes model-read helpers such as softmax probabilities, stochastic dropout forward passes, penultimate-layer embeddings, and gradient embeddings.

## Editable interface

Only the `CustomSampling` class in `badge/query_strategies/custom_sampling.py` (lines 28–54) is editable. Every method fills the same contract: `query(self, n)` returns an `np.ndarray` of `n` indices into `self.X`, chosen from the currently unlabeled pool. Helper methods may be added, but `query(n)` is the entry point. `numpy as np` and the `Strategy` base are already imported.

The starting point is uniform random sampling:

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

Three OpenML tabular classification datasets spanning sizes and difficulties — **letter** recognition (26-class, the largest pool), **spambase** (binary), and **splice** (3-class). Protocol: 20 rounds of batch active learning, the model retrained from scratch after each round, fixed total label budget. Two metrics, higher is better on both:

- `accuracy` — test accuracy after the final round (final-budget quality).
- `auc` — area under the learning curve (accuracy vs. number of labeled samples) across all 20 rounds, capturing sample efficiency.
