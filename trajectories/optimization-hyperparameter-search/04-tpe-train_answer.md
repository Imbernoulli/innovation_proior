DEHB behaved exactly as multi-fidelity should and exposed its own price. Convergence AUC jumped where the prior rungs were weakest — SVM $0.981$ (versus CMA-ES's $0.664$ and random search's $0.789$), XGBoost $0.949$, several seeds reading *above* $1.0$ — and `total_evals` ballooned to 104 on XGBoost, 246 on SVM, 92 on the NN, the cheap-fidelity lever working. But the NN final best *slipped*: $-3049$ mean, seed 42 at $-3086$, worse than random search's $-3070$. That is precisely the weak-rank-correlation failure I flagged — at low fidelity the NN's 50-iteration scores rank configs differently from the 500-iteration truth, so aggressive triage promoted the wrong survivors. DEHB's residual weakness is structural and shared by SH and Hyperband: it *guesses* the configurations-versus-fidelity tradeoff, and where the cheap fidelity lies, it throws away the eventual winner early.

Before I reach for the full multi-fidelity hedge, let me isolate a different ingredient all three prior rungs lacked in clean form: a *probabilistic model of where good configurations live* that I can trust at full fidelity. CMA-ES learned from history but with a model too expensive to amortize; DEHB learned model-free and inherited the low-fidelity risk. The question this rung answers is narrow — if I commit to single-fidelity evaluation (no triage risk at all) but spend each full evaluation on the configuration a *model* says is most promising, how far does pure model-guided search get? This is the model component the strongest combination later will graft onto a multi-fidelity skeleton, so getting it right in isolation matters.

I propose the **Tree-structured Parzen Estimator (TPE)**. The template is sequential model-based optimization: keep a cheap surrogate of the expensive loss built from the history, and each round maximize an acquisition over it to pick the next point, evaluate, append, refit. Two choices define the method — what acquisition and what surrogate. I take Expected Improvement as the acquisition because it balances exploit against explore with no hand-set target: given the best score $y^*$ seen and a surrogate's predictive distribution over the value $Y(x)$ at a candidate, EI is the expected amount by which $x$ would beat the incumbent. Outcomes worse than $y^*$ contribute zero improvement, and the expectation over the predictive distribution automatically rewards both high predicted score (exploit) and high predictive spread that could clear $y^*$ (explore) — that trade falls out of one integral with no tuning. So I commit to EI and spend the design effort on the surrogate, which is where the space's structure bites.

The textbook surrogate is a Gaussian process — elegant, with analytic posterior mean and variance and closed-form EI — but it is the wrong fit here, and naming why forces the alternative. A GP kernel needs a metric on the whole configuration vector, and my spaces have a categorical axis (SVM kernel, NN activation) with no natural metric; conditioning costs $O(n^3)$; and EI under a GP stakes all exploration on a single point estimate of the predictive variance, which a sparse early sample (my normal condition at 40 evaluations) can collapse to near-zero, killing exploration silently. So I stare at the EI integral and ask what it actually needs. EI needs $p(y\mid x)$, the distribution over loss for any candidate. The GP models the forward map $x \to$ distribution-over-$y$ directly, which is why it needs a metric on $x$. But I never need a calibrated number — I draw candidates and *rank* them by EI. The data I have a lot of is the other direction: for each trial I have $(\text{config}, \text{score})$, samples of the joint. So I factor it the other way. Write $p(y\mid x) = p(x\mid y)\,p(y)/p(x)$ and model $p(x\mid y)$ — the density over *configurations* given the outcome — which is the direction my data is easy in.

Split the outcome at a quantile: pick $\gamma$ and set the threshold $y^*$ so a fraction $\gamma$ of observations fall on the good side, then estimate two densities — $l(x)$ over the configs whose score is in the top $\gamma$, $g(x)$ over the rest. Push EI through this factorization. The improvement integral over the good region factors because there $p(x\mid y)$ is, by construction, exactly $l(x)$ and pulls out of the $y$-integral as a constant in $x$; the denominator $p(x)$ is the total mixture $\gamma\,l(x) + (1-\gamma)\,g(x)$ by total probability over the same split; and what survives collapses to

$$\mathrm{EI}(x) \;\propto\; \Big(\gamma + (1-\gamma)\,\tfrac{g(x)}{l(x)}\Big)^{-1},$$

which is strictly *increasing* in the ratio $l(x)/g(x)$. So maximizing EI is *exactly* maximizing $l(x)/g(x)$ — pick the config most likely under the good density relative to the bad one. This is the payoff of the backwards factorization: I never built a regressor on $x$, never needed a whole-vector metric, and the acquisition reduced to a ratio of two densities estimated directly from the grouped history, each with a per-coordinate kernel that composes natively over mixed types. The quantile is forced by the construction: GP-EI uses $y^*$ = the best observed score, but then the good set would be a single point and $l$ would have no data to fit, so a quantile $\gamma$ is mandatory, and it doubles as the explore/exploit knob — larger $\gamma$ admits more configs into the "good" set (more exploration, since $l$ then covers more space), smaller $\gamma$ is greedier. I use $\gamma = 0.25$, more exploratory than the classic $0.15$, the right bias under a tiny budget where the good set must not collapse to two or three points and over-concentrate the ratio.

The implementation departs from the clean ideal in two honest ways. I encode each config to $[0,1]$ (log-linear for scale knobs, the categorical to its choice-index over the number of choices minus one — the same lossy scalar flattening I accepted for CMA-ES, fine here because EI only ranks and the continuous knobs dominate). I run $n_{\text{startup}} = 10$ uniform-random configs first, because below a handful of observations the densities are meaningless. Then I split: $n_{\text{good}} = \max(1, \lfloor\gamma\cdot|\text{history}|\rfloor)$, threshold at the $n_{\text{good}}$-th best score. I estimate $l$ and $g$ as simple Gaussian KDEs, $\log p(x) = \log\big(\tfrac1n\sum_i \exp(-\tfrac12\lVert x - x_i\rVert^2/\text{bw}^2) + \varepsilon\big)$, with a *single global* bandwidth per density set by a Scott-style floor, $\text{bw} = \max(0.05, \text{samples.std}() + \varepsilon)$. This is the key simplification to be honest about — not the per-point adaptive bandwidth of the full method, just one scale per density, which under 40 observations is a reasonable bias-variance trade (per-point bandwidths need more data) but does smear fine structure. Finally I optimize EI by *sampling*: draw $n_{\text{ei}} = 24$ uniform configs, score each by $\log l(x) - \log g(x)$, and return the argmax — sampled optimization rather than gradient ascent, which sidesteps the gradient-free categorical axis and is cheap at 24 candidates. Every suggestion is full fidelity $1.0$; this rung deliberately uses *no* multi-fidelity, to measure the model in isolation.

TPE's strength is *final quality* and *low-variance* convergence, because it spends every full-price evaluation on a model-chosen config. So I expect its NN final best to *recover* from DEHB's slip — no low-fidelity mis-promotion can hurt it when there is no low fidelity — landing back near random search's band. But its convergence AUC should be *lower* than DEHB's inflated multi-fidelity numbers: `total_evals` back to exactly the budget (the single-fidelity tell), and a quarter of the budget on the small benchmarks spent on the 10-config warm-up before the model even turns on, so the early curve cannot front-load the way cheap triage did. If I see budget-pinned `total_evals`, a recovered NN best, but AUC distinctly below DEHB's, the diagnosis is clean: a *good model alone* fixes final quality and reliability but cannot match the *anytime* convergence cheap multi-fidelity buys. The two strengths are complementary — which is the setup for hedging the configs-versus-fidelity tradeoff next, and then putting a model *inside* that hedge.

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
