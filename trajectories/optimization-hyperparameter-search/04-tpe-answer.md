**Problem.** DEHB's multi-fidelity won convergence AUC but exposed its price: AUCs above 1.0
(normalized-curve artifact from cheap noisy scores) and an NN final best that *slipped* to −3086 on one seed
because low-fidelity scores mis-ranked configs and triage promoted the wrong survivors. This rung isolates a
different ingredient — a probabilistic model of where good configs live, trusted at full fidelity, with no
triage risk.

**Key idea.** Tree-structured Parzen Estimator. Factor the joint the easy way — model p(x|y), not p(y|x):
split the history at a quantile γ into a "good" set (top γ) and the rest, fit densities `l(x)` and `g(x)`
over the configs in each, and suggest the config maximizing the ratio `l(x)/g(x)`. Pushing Expected
Improvement through Bayes shows EI is monotone in exactly this ratio, so ranking by `l/g` *is* maximizing EI —
with no whole-vector metric (densities use per-coordinate kernels, native to mixed types) and no GP O(n³) or
variance-collapse failure.

**Why this and not DEHB.** Every full-price evaluation goes to a model-chosen config, so final quality and
reliability recover (no low-fidelity mis-promotion can hurt it). It is the model component in isolation —
single fidelity by design — to be grafted onto a multi-fidelity skeleton later.

**Implementation notes (task-specific).** Encode to [0,1] (log-linear; categorical → scalar choice-index,
lossy but EI only ranks). KDEs use a *single global* bandwidth `max(0.05, std + ε)` (not per-point adaptive —
a deliberate small-data simplification). EI is optimized by *sampling* 24 candidates and taking the argmax of
`log l − log g`, sidestepping the gradient-free categorical axis.

**Hyperparameters.** `γ = 0.25`, `n_startup = 10` random warm-up, `n_ei_candidates = 24`, fidelity fixed at
1.0.

**What to watch.** `total_evals` pinned to the budget (single fidelity); NN final best recovered toward
random search's band; but convergence AUC *below* DEHB's inflated numbers, because the 10-config warm-up plus
full-price evals cannot front-load the curve. Diagnosis: a good model fixes quality but not anytime
convergence — complementary to multi-fidelity, which sets up hedging the tradeoff next.

```python
# EDITABLE region of scikit-learn/custom_hpo.py (lines 255-326) — step 4: TPE
class CustomHPOStrategy:
    """Tree-structured Parzen Estimator (TPE)."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.gamma = 0.25  # fraction of best observations for l(x)
        self.n_startup = 10  # random search before modelling
        self.n_ei_candidates = 24  # candidates to score with EI

    def _encode(self, config, space):
        """Encode a config to a numeric vector in [0,1]."""
        vec = []
        for p in space.params:
            val = config[p.name]
            if p.type == "categorical":
                # One-hot-ish: use index / len
                idx = p.choices.index(val)
                vec.append(idx / max(len(p.choices) - 1, 1))
            elif p.type in ("float", "int"):
                if p.log_scale:
                    v = (np.log(val) - np.log(p.low)) / (np.log(p.high) - np.log(p.low))
                else:
                    v = (val - p.low) / (p.high - p.low)
                vec.append(float(np.clip(v, 0, 1)))
        return np.array(vec)

    def _kde_logpdf(self, x, samples, bw):
        """Simple Gaussian KDE log-density at x."""
        diffs = x[None, :] - samples  # (n_samples, dim)
        return float(
            np.log(np.mean(np.exp(-0.5 * np.sum(diffs**2 / bw**2, axis=1))) + 1e-30)
        )

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        if len(history) < self.n_startup:
            return space.sample_uniform(self.rng), 1.0

        # Split observations into good (l) and bad (g)
        scores = np.array([t.score for t in history])
        n_good = max(1, int(self.gamma * len(history)))
        threshold = np.sort(scores)[-n_good]

        good_vecs = np.array([
            self._encode(t.config, space)
            for t in history if t.score >= threshold
        ])
        bad_vecs = np.array([
            self._encode(t.config, space)
            for t in history if t.score < threshold
        ])

        if len(bad_vecs) == 0:
            bad_vecs = good_vecs.copy()

        # Bandwidth: Scott's rule
        bw_good = max(0.05, good_vecs.std() + 1e-6)
        bw_bad = max(0.05, bad_vecs.std() + 1e-6)

        # Generate candidates and score them by l(x)/g(x)
        best_score = -np.inf
        best_config = None
        for _ in range(self.n_ei_candidates):
            candidate = space.sample_uniform(self.rng)
            x = self._encode(candidate, space)
            log_l = self._kde_logpdf(x, good_vecs, bw_good)
            log_g = self._kde_logpdf(x, bad_vecs, bw_bad)
            ei = log_l - log_g
            if ei > best_score:
                best_score = ei
                best_config = candidate

        return best_config, 1.0
```
