Random search split exactly the way the theory predicted. On final quality it is genuinely strong — XGBoost mean best $-0.394$, SVM $0.978$, NN $-3050$, competitive optima everywhere — but it bled convergence AUC, and bled it with high variance: SVM AUC swung $0.563 \to 0.913 \to 0.890$ across seeds for a mean of only $0.789$, and the NN told the same story ($0.665 / 0.691 / 0.962$ for $0.772$). That is the signature of statelessness: a seed whose draw order did not stumble onto a good config until late had nothing pulling it back toward where the early evidence pointed, so the best-so-far curve stayed flat and the area under it collapsed. The deficiency is not final quality; it is that random search throws away every loss it has paid for. The obvious next move is to *use the history* — to maintain a model of where the good configurations live and concentrate draws there — and the most principled way to do that for the continuous knobs is to adapt the shape of the sampling distribution itself.

I propose **(μ/μ_w, λ)-CMA-ES**, Covariance Matrix Adaptation Evolution Strategy, run in a normalized $[0,1]$ box. I encode every knob to $[0,1]$ — linear for plain ranges, log-linear for `log_scale` knobs, and a categorical to its choice-index over the number of choices minus one — search in that box, and decode back. The categorical flattening is a real distortion I accept knowingly: the optimizer sees a continuous relaxation of a discrete axis with no metric meaning, and for 3-D SVM, where the kernel is one of three coordinates, that is a third of the space; I bet the continuous knobs dominate the loss and the rounding noise is tolerable, because a separate discrete handler is more machinery than this rung — whose job is to test "adapt the continuous shape" cleanly — should carry.

The core idea is to sample from a multivariate normal $\mathcal N(m, \sigma^2 C)$ over the box, rank the offspring by score, and use the ranks to manufacture a better mean and a better ellipsoid for the next generation. A normal because, among all distributions on $\mathbb R^n$ with prescribed second moments, it has maximum entropy — it commits to nothing beyond the covariance I have actually learned and privileges no direction a priori. And the *shape* $C$ matters more than the mean because near a good region the objective looks locally quadratic, $f \approx \tfrac12 (x-x^*)^\top H (x-x^*)$, where the ideal sampling covariance is $C \propto H^{-1}$: in the coordinates $C^{-1/2}(x-x^*)$ the level sets become spheres and search is isotropic. Learning $C$ is the gradient-free analogue of the inverse-Hessian preconditioner a quasi-Newton method maintains — except I have only ranks. The NN space, 6-D and almost certainly ill-conditioned (a narrow viable learning-rate band against a much broader capacity band), is the one that should reward learning that anisotropy.

The update is derived in the order the pieces become forced. The population is $\lambda = 4 + \lfloor 3\ln n\rfloor$ offspring per generation (5–8 here for $n=3$–$6$), $\mu = \lambda//2$ parents, with rank-based recombination weights $w_i \propto \ln(\mu+\tfrac12) - \ln i$ normalized to sum to one, so the best parent counts most and the weights decay smoothly. The variance-effective selection mass $\mu_{\text{eff}} = 1/\sum_i w_i^2$ summarizes how many parents really contribute, and it appears in every learning rate below — that is the amount of selection information per generation, and every rate has to be throttled by it or the estimate is noise. The new mean is the weighted recombination $m \leftarrow \sum_i w_i\, x_{i:\lambda}$.

The covariance is where the care goes, and the principle is *derandomization*: do not infer the strategy parameters indirectly from which offspring won — read them off the steps actually taken. The rank-$\mu$ update estimates a covariance from this generation's selected steps, $C_\mu = \sum_i w_i\, y_i y_i^\top$ with $y_i = (x_{i:\lambda} - m_{\text{old}})/\sigma$, referenced to the *old* mean the points were sampled around. Referencing the *selected* mean instead would measure spread *within* the winners rather than the displacement of the winners, systematically shrinking the variance along exactly the directions selection just favored — backwards. The rank-one update adds cross-generation information through the evolution path $p_c$, an exponentially-smoothed sum of realized mean-shifts,

$$p_c \leftarrow (1-c_c)\,p_c + h_\sigma\sqrt{c_c(2-c_c)\,\mu_{\text{eff}}}\;\frac{m - m_{\text{old}}}{\sigma},$$

and $C$ gains $c_1\, p_c p_c^\top$. The path captures the sign information rank-$\mu$ throws away: consecutive steps in the same direction mean a long correlated ridge to stretch $C$ along, whereas zig-zagging steps cancel in the path and signal an over-long step. The full update is the convex blend

$$C \leftarrow (1-c_1-c_\mu)\,C + c_1\, p_c p_c^\top + c_\mu \sum_i w_i\, y_i y_i^\top,$$

with $c_1 = 2/((n+1.3)^2 + \mu_{\text{eff}})$ and $c_\mu = \min(1-c_1,\, 2(\mu_{\text{eff}}-2+1/\mu_{\text{eff}})/((n+2)^2+\mu_{\text{eff}}))$ — both small, both scaled by $\mu_{\text{eff}}$ and shrinking with dimension, because $C$ can only move as fast as the selection signal justifies. I then symmetrize $C$ and floor its eigenvalues to keep it positive-definite, since the rank-deficient rank-$\mu$ term plus finite arithmetic can otherwise push an eigenvalue negative and break the next eigendecomposition.

The step size $\sigma$ rides a separate path, and keeping it separate from $C$ is the whole trick of cumulative step-size adaptation. The conjugate path

$$p_\sigma \leftarrow (1-c_\sigma)\,p_\sigma + \sqrt{c_\sigma(2-c_\sigma)\,\mu_{\text{eff}}}\;C^{-1/2}\frac{m-m_{\text{old}}}{\sigma}$$

accumulates the mean-shift in the *isotropic* frame: the $C^{-1/2}$ whitening removes the shape so only length and cross-generation correlation survive. Then

$$\sigma \leftarrow \sigma\,\exp\!\left(\frac{c_\sigma}{d_\sigma}\left(\frac{\lVert p_\sigma\rVert}{\mathbb E\lVert \mathcal N(0,I)\rVert} - 1\right)\right).$$

If successive selected steps line up, the path is longer than an independent random walk would be, so I am making consistent progress and lengthen $\sigma$; if they anticorrelate (overshoot, zig-zag), the path is short and $\sigma$ shrinks. This is also why CSA is unbiased under random selection: with no signal the expected path length is exactly $\mathbb E\lVert\mathcal N(0,I)\rVert$, so $\mathbb E[\ln\sigma]$ is stationary — $\sigma$ does not drift when I am learning nothing, a safety property. The $h_\sigma$ flag couples the two paths: when $\lVert p_\sigma\rVert$ is anomalously large, it stalls the rank-one accumulation for one step so a single outsized move does not blow up $C$.

In the scaffold this becomes a generational `suggest`. On the first call I initialize $m = 0.5\cdot\mathbf 1$, $C = I$, $\sigma = 0.3$, sample a whole population, and queue the decoded configs; each later call matches the returned trial back to its pending candidate by `np.allclose` on the encoded vector, fills in its score, and once the whole generation is scored runs the update and resamples. Every evaluation is full fidelity — CMA-ES has no notion of cheap partial looks, so it spends the budget one generation at a time.

I have to be honest about the regime, because it will bite. CMA-ES is a method for *many* generations: it pays an up-front cost learning $C$ and $\sigma$, with the payoff asymptotic. My budget is 40–50, which at $\lambda\approx 5$–$8$ is only 5–10 generations — barely enough for the evolution paths to accumulate, let alone for $C$ to bend into the right ellipsoid. Worse, $\sigma_0 = 0.3$ in a unit box is large, so the first generations sample widely and the early best-so-far curve can sit flat while the distribution contracts — and the convergence AUC integrates exactly that early flatness. So I expect this rung to *lose* to random search on AUC despite being more sophisticated, with the worst single seed cratering on the benchmark where $\sigma_0$ is most mismatched and the categorical relaxation compounds the risk (SVM). Final best scores should stay in the same competitive band. If I see CMA-ES post AUCs below random search's — a SVM mean toward $0.6$, an isolated seed near $0.2$ — that is not a bug but the diagnosis confirmed: a pure continuous-shape optimizer is the wrong tool under a few-dozen-evaluation budget because it cannot amortize its model-learning cost. The fix is not a better model but *cheaper evaluations* — which forces multi-fidelity at the next rung.

```python
# EDITABLE region of scikit-learn/custom_hpo.py (lines 255-326) — step 2: CMA-ES (Optuna sampler)
class CustomHPOStrategy:
    """CMA-ES: Covariance Matrix Adaptation Evolution Strategy."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._initialized = False
        self._mean = None
        self._sigma = 0.3
        self._C = None  # covariance matrix
        self._p_sigma = None  # evolution path for sigma
        self._p_c = None  # evolution path for C
        self._gen = 0
        self._lam = None  # population size
        self._mu = None
        self._weights = None
        self._mu_eff = None
        self._candidates = []
        self._pending_evals = []

    def _encode(self, config, space):
        vec = []
        for p in space.params:
            val = config[p.name]
            if p.type == "categorical":
                idx = p.choices.index(val)
                vec.append(idx / max(len(p.choices) - 1, 1))
            elif p.type in ("float", "int"):
                if p.log_scale:
                    v = (np.log(val) - np.log(p.low)) / (np.log(p.high) - np.log(p.low))
                else:
                    v = (val - p.low) / (p.high - p.low)
                vec.append(float(np.clip(v, 0, 1)))
        return np.array(vec)

    def _decode(self, vec, space):
        config = {}
        for i, p in enumerate(space.params):
            v = float(np.clip(vec[i], 0, 1))
            if p.type == "categorical":
                idx = int(round(v * max(len(p.choices) - 1, 1)))
                idx = min(idx, len(p.choices) - 1)
                config[p.name] = p.choices[idx]
            elif p.type == "float":
                if p.log_scale:
                    config[p.name] = float(np.exp(
                        np.log(p.low) + v * (np.log(p.high) - np.log(p.low))))
                else:
                    config[p.name] = float(p.low + v * (p.high - p.low))
            elif p.type == "int":
                if p.log_scale:
                    config[p.name] = int(round(np.exp(
                        np.log(p.low) + v * (np.log(p.high) - np.log(p.low)))))
                else:
                    config[p.name] = int(round(p.low + v * (p.high - p.low)))
        return config

    def _init_cma(self, dim):
        self._mean = np.full(dim, 0.5)
        self._C = np.eye(dim)
        self._p_sigma = np.zeros(dim)
        self._p_c = np.zeros(dim)
        self._lam = 4 + int(3 * np.log(dim))
        self._mu = self._lam // 2
        weights = np.log(self._mu + 0.5) - np.log(np.arange(1, self._mu + 1))
        self._weights = weights / weights.sum()
        self._mu_eff = 1.0 / np.sum(self._weights ** 2)
        self._initialized = True

    def _sample_population(self, space):
        dim = space.dim
        # Eigendecomposition of C
        eigvals, eigvecs = np.linalg.eigh(self._C)
        eigvals = np.maximum(eigvals, 1e-20)
        sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        self._candidates = []
        self._pending_evals = []
        for _ in range(self._lam):
            z = self.rng.randn(dim)
            x = self._mean + self._sigma * sqrt_C @ z
            x = np.clip(x, 0, 1)
            cfg = self._decode(x, space)
            cfg = space.clip(cfg)
            self._candidates.append((x, cfg, None))
            self._pending_evals.append(cfg)

    def _update(self, space):
        """CMA-ES update step after a full generation is evaluated."""
        dim = space.dim

        # Sort by score (descending — we maximize)
        scored = [(s, x) for x, _, s in self._candidates if s is not None]
        scored.sort(key=lambda p: p[0], reverse=True)

        # Recombination
        old_mean = self._mean.copy()
        self._mean = np.zeros(dim)
        for i in range(self._mu):
            self._mean += self._weights[i] * scored[i][1]

        # Evolution paths
        c_sigma = (self._mu_eff + 2) / (dim + self._mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((self._mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + self._mu_eff / dim) / (dim + 4 + 2 * self._mu_eff / dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        eigvals, eigvecs = np.linalg.eigh(self._C)
        eigvals = np.maximum(eigvals, 1e-20)
        inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        self._p_sigma = (1 - c_sigma) * self._p_sigma + \
            np.sqrt(c_sigma * (2 - c_sigma) * self._mu_eff) * \
            inv_sqrt_C @ (self._mean - old_mean) / self._sigma

        h_sigma = 1.0 if (np.linalg.norm(self._p_sigma) /
                          np.sqrt(1 - (1 - c_sigma) ** (2 * (self._gen + 1)))
                          < (1.4 + 2 / (dim + 1)) * chi_n) else 0.0

        self._p_c = (1 - c_c) * self._p_c + \
            h_sigma * np.sqrt(c_c * (2 - c_c) * self._mu_eff) * \
            (self._mean - old_mean) / self._sigma

        # Covariance matrix update
        c1 = 2.0 / ((dim + 1.3) ** 2 + self._mu_eff)
        c_mu = min(1 - c1, 2 * (self._mu_eff - 2 + 1.0 / self._mu_eff) /
                   ((dim + 2) ** 2 + self._mu_eff))

        rank_one = np.outer(self._p_c, self._p_c)
        rank_mu = np.zeros((dim, dim))
        for i in range(self._mu):
            diff = (scored[i][1] - old_mean) / self._sigma
            rank_mu += self._weights[i] * np.outer(diff, diff)

        self._C = (1 - c1 - c_mu) * self._C + c1 * rank_one + c_mu * rank_mu
        # Ensure symmetry and positive definiteness
        self._C = (self._C + self._C.T) / 2
        eigvals_check = np.linalg.eigvalsh(self._C)
        if np.min(eigvals_check) < 1e-20:
            self._C += np.eye(dim) * (1e-20 - np.min(eigvals_check))

        # Step-size update
        self._sigma *= np.exp(
            (c_sigma / d_sigma) * (np.linalg.norm(self._p_sigma) / chi_n - 1))
        self._sigma = np.clip(self._sigma, 1e-10, 1.0)

        self._gen += 1

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        if not self._initialized:
            self._init_cma(space.dim)
            self._sample_population(space)

        # Update scores for pending candidates
        if history:
            last = history[-1]
            last_vec = self._encode(last.config, space)
            for i, (x, cfg, score) in enumerate(self._candidates):
                if score is None and np.allclose(x, last_vec, atol=0.01):
                    self._candidates[i] = (x, cfg, last.score)
                    break

        # If all candidates evaluated, do CMA update and resample
        if self._candidates and all(s is not None for _, _, s in self._candidates):
            self._update(space)
            self._sample_population(space)

        # Return next pending evaluation
        if self._pending_evals:
            cfg = self._pending_evals.pop(0)
            return cfg, 1.0

        # Fallback
        return space.sample_uniform(self.rng), 1.0
```
