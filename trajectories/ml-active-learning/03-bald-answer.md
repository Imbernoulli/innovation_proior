**Problem.** Least confidence chases the marginal predictive uncertainty `1 − max_y p(y|x)`, but a single
deterministic forward pass cannot tell "unsure because I lack data here" (a label fixes it) from "unsure
because the point is intrinsically noisy" (no label ever fixes it). It spends budget on both, which on
multi-class problems like letter dragged its mean *below* random. I want to query only the uncertainty a
label can resolve.

**Key idea (BALD — Bayesian Active Learning by Disagreement).** The myopic-best query maximizes the
expected drop in parameter-posterior entropy, `H[θ|D] − E_y[H[θ|y,x,D]]` — an intractable parameter-space
object. But it *is* a mutual information `I[θ,y|x,D]`, and MI is symmetric, so it equals the output-space
form `H[y|x,D] − E_{θ∼p(θ|D)}[H[y|x,θ]]`: do inference once, then per candidate compute two tiny class
entropies, no re-inference. The subtraction separates epistemic (resolvable) from aleatoric (irreducible)
uncertainty — high exactly when each posterior sample is confident yet the samples disagree. It contains
uncertainty sampling as the noise-free special case (`H[y|x,θ]=0`).

**Why it fits this scaffold.** The weight posterior is intractable and there is no GP/probit closed form
here, so use the model-agnostic estimator: `n_drop` dropout-on forward passes
(`predict_prob_dropout_split`) are `n_drop` approximate posterior samples. `p̄` = mean softmax over passes;
`I(x) ≈ H[p̄] − mean_t H[pᵗ]`, non-negative by Jensen. (The harness exposes only the dropout posterior, so
the GP closed form, nuisance-parameter, and preference-learning extensions of the full method are out.)

**Hyperparameters.** `n_drop = 10` (posterior samples — enough to estimate two entropies, cheap over the
pool). Rank by `U = mean-per-pass-entropy − entropy-of-mean = −I(x)`, sort ascending, take first `n`.

**Step-3 edit.** `T` dropout passes → mean prediction `pb`; total entropy `entropy1`; mean per-pass entropy
`entropy2`; `U = entropy2 − entropy1`; smallest `U` = largest MI.

**What to watch.** Expect BALD to repair letter (the noise-subtraction term is exactly what least confidence
lacked there), roughly match least confidence on spambase (little irreducible noise; coarse dropout
estimate), small gain on splice. Residual risk it does *not* fix: scoring points in isolation still allows
a redundant near-duplicate batch — the next rung must fuse uncertainty with diversity.

```python
# EDITABLE region of badge/query_strategies/custom_sampling.py (lines 28-54) — step 3: BALD (MC dropout)
class CustomSampling(Strategy):
    """BALD — Bayesian Active Learning by Disagreement (MC Dropout).
    Selects samples where there is maximal disagreement across stochastic
    forward passes, approximating mutual information."""

    def __init__(self, X, Y, idxs_lb, net, handler, args, n_drop=10):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.n_drop = n_drop

    def query(self, n):
        import torch
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob_dropout_split(
            self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], self.n_drop
        )
        # Mean prediction across MC samples
        pb = probs.mean(0)
        # Total entropy: H[y | x, D]
        entropy1 = (-pb * torch.log(pb + 1e-10)).sum(1)
        # Expected entropy: E_theta[H[y | x, theta]]
        entropy2 = (-probs * torch.log(probs + 1e-10)).sum(2).mean(0)
        # BALD score = total entropy - expected entropy = mutual information
        U = entropy2 - entropy1
        return idxs_unlabeled[U.sort()[1][:n]]
```
