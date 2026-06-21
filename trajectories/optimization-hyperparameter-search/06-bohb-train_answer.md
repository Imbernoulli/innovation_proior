Hyperband closed the gap exactly where I expected and left exactly the residual I expected. Its convergence AUC recovered above TPE's on every benchmark — NN $0.950$ (TPE $0.831$), SVM $0.954$ (TPE $0.876$), XGBoost $0.960$ — and `total_evals` came back high and *dead stable across seeds* (105 on XGBoost, 95 on SVM and NN, every seed identical), the deterministic-bracket tell that distinguishes it from DEHB's data-dependent swing. But on final best it did *not* clearly beat TPE: XGBoost $-0.391$ (TPE $-0.392$, a wash), SVM $0.978$ (TPE $0.980$, if anything slightly worse), NN $-3053$ (TPE $-3048$). That is the cap I predicted: Hyperband allocates budget brilliantly but samples every configuration uniformly at random, so it never *aims* — its quality is bounded by random sampling. The TPE rung had the model and no allocation; the Hyperband rung has the allocation and no model. They were traded off against each other for two rungs running. The final move stops trading and *compounds* them.

I propose **BOHB** — Bayesian Optimization + Hyperband: keep Hyperband's exact bracket schedule, but replace its random configuration sampling with TPE's $l(x)/g(x)$ model. This is the right composition, not a hopeful stapling, and the reason is that every HPO method makes two logically separate decisions — *which* configuration to evaluate, and *how much* resource to spend on it before judging — and the ladder has attacked them one at a time. Random search aimed at neither. CMA-ES aimed the which but fixed the how-much at full and could not amortize. DEHB aimed the how-much and learned the which model-free, inheriting the cheap-rank risk. TPE aimed the which (the $l/g$ model) and fixed the how-much. Hyperband aimed the how-much (the bracket hedge) and left the which to random sampling. The two strongest single-axis rungs — TPE on *which*, Hyperband on *how-much* — are precisely the two I hold. Hyperband's strength is the *evaluation* axis: it spends little on hopeless configs, reserves full fidelity for survivors, and hedges the configs-versus-fidelity dilemma; its weakness is the *selection* axis, random sampling. TPE's strength is exactly the selection axis: a probabilistic model that aims each draw at high $l(x)/g(x)$; its weakness was the evaluation axis. The weaknesses are *disjoint*, and each method's strength covers the other's weakness — so the composition is the unique way to be strong on both axes at once. Hyperband still decides how much resource each config gets and when to kill it; TPE decides which configs to propose. Replace "sample uniformly" with "sample from the model" and nothing else about the bracket machinery changes.

The one genuinely new design question is that the model must now be fed by the *multi-fidelity* history. In plain TPE every observation was a full-fidelity score; here most are cheap low-fidelity scores, and they are noisier. The choice is pragmatic: maintain a single pooled set of all completed trials — $(\text{encoded\_vec}, \text{score}, \text{fidelity})$ — and fit one $l/g$ model over the *whole pool* regardless of fidelity, rather than the purist's per-fidelity models. Under a 40–50-cost budget there are simply not enough observations at the highest fidelity to fit a model there early, so pooling across fidelities is what lets the model turn on at all within the budget; the cost is that it treats a cheap noisy score like an expensive clean one, a real approximation but the only one that yields a usable model this early. The model warms up after $n_{\text{startup}} = 8$ pooled trials (below TPE's 10, because the budget is tighter and the brackets generate observations fast), then splits at $\gamma = 0.15$ — *more aggressive* than TPE's $0.25$, a smaller "good" set, because the pooled multi-fidelity history has more observations, so a tighter, more exploitative elite is both affordable and what I want once allocation is handling exploration. The KDEs use the same single-global-bandwidth simplification, $\text{bw} = \max(0.05, \text{std}\cdot\text{bw\_factor} + \varepsilon)$ with $\text{bw\_factor} = 1.0$, and EI is again optimized by sampling $n_{\text{candidates}} = 24$ and taking the argmax of $\log l - \log g$.

It is worth being precise about why the pooled model still composes correctly with the bracket schedule despite mixing fidelities, because that is the subtle joint where the staple could fail. The model's job is only to *rank* candidate configs by $l(x)/g(x)$ — it needs no calibrated scores, only a roughly correct notion of which region the good configs occupy. Cheap low-fidelity scores are biased estimators of the terminal score, but on most of the space they are *order-correlated* with it — a config in the top $\gamma$ at a third of the trees is, more often than not, genuinely in a good region — so pooling them sharpens $l(x)$ toward the right region faster than waiting for scarce full-fidelity data. The places this breaks are exactly the configs whose cheap-vs-expensive ranking inverts, rare on XGBoost and SVM but the known NN exception. And crucially, even when the model is mildly misled, Hyperband's *allocation* is the backstop: a config the model over-rates still has to survive the successive-halving rungs at increasing fidelity before it consumes a full evaluation, so a model error the cheap fidelity would have masked gets caught by the schedule. The model aims; the schedule vetoes. That mutual correction is the real reason the composition is more than the sum of its parts.

The bracket schedule is Hyperband's with one tightening for the model: $s_{\max} = \min(3, \lfloor\log_\eta(\text{total})\rfloor)$ caps brackets at 4 rather than Hyperband's 5, because each bracket now consumes model-guided samples and a tighter budget should not spread itself across more brackets than the model can inform. On the first call the brackets are built by drawing each bracket's configs from `_sample_from_model` — which falls back to uniform until the 8-trial warm-up is met, then samples model-guided — and queued at their initial fidelities. As trials return, each is appended to the pooled trial set (so the model keeps improving), matched back to its bracket by exact `config == last.config` and fidelity proximity, and when a bracket's rung completes the successive-halving advance runs inline: sort, keep top $\lfloor\text{len}/\eta\rfloor$, raise the fidelity, re-queue the survivors. When the queue empties, instead of Hyperband's random full-fidelity fallback, BOHB generates a *model-guided* full-fidelity config — so even the fallback is aimed.

This should be the first rung strong on *both* axes at once. On convergence AUC it should at least match Hyperband (the schedule is identical) and plausibly beat it where aiming the early cheap configs helps the curve climb — the cheap brackets now flood model-guided rather than random configs, so the survivors that reach high fidelity are better from the start. On final best it should finally *break the Hyperband cap*: XGBoost the clearest case, model-aimed full-fidelity survivors pushing past $-0.391$ toward the ladder's best (around $-0.389$), with SVM/NN at or above the strongest band. The benchmark I am still watching is the NN: pooling cheap and expensive scores into one model means a misleading cheap score can pull the elite toward a region good at 50 iterations but not 500, so I expect a real risk of a single NN seed cratering on convergence AUC even while the NN final best improves. If I see final best improve over Hyperband *while* convergence AUC stays at Hyperband's level — strong on selection and allocation at once — that is the composition working as designed, and it is why BOHB is the strongest rung: it is the only method on the ladder weak on neither axis. The honest residual is the pooled-fidelity model's NN vulnerability, the cheap-rank-correlation problem re-entering through the model rather than through triage — which a method modeling each fidelity separately, or learning the cross-fidelity correlation, would be the next thing to attack beyond this ladder.

```python
# EDITABLE region of scikit-learn/custom_hpo.py (lines 255-326) — step 6: BOHB
class CustomHPOStrategy:
    """BOHB: Bayesian Optimization + Hyperband."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.eta = 3
        self.gamma = 0.15  # fraction for good KDE
        self.n_startup = 8  # random configs before model-guided
        self.n_candidates = 24
        self.bw_factor = 1.0
        self._brackets = []
        self._queue = []
        self._initialized = False
        self._all_trials = []  # (vec, score, fidelity)

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

    def _kde_logpdf(self, x, samples, bw):
        diffs = x[None, :] - samples
        return float(np.log(
            np.mean(np.exp(-0.5 * np.sum(diffs**2 / bw**2, axis=1))) + 1e-30
        ))

    def _sample_from_model(self, space):
        """Sample config guided by TPE model or random."""
        if len(self._all_trials) < self.n_startup:
            return space.sample_uniform(self.rng)

        vecs = np.array([t[0] for t in self._all_trials])
        scores = np.array([t[1] for t in self._all_trials])
        n_good = max(1, int(self.gamma * len(scores)))
        threshold = np.sort(scores)[-n_good]

        good = vecs[scores >= threshold]
        bad = vecs[scores < threshold]
        if len(bad) == 0:
            bad = good.copy()

        bw_good = max(0.05, good.std() * self.bw_factor + 1e-6)
        bw_bad = max(0.05, bad.std() * self.bw_factor + 1e-6)

        best_ei = -np.inf
        best_cfg = None
        for _ in range(self.n_candidates):
            cfg = space.sample_uniform(self.rng)
            x = self._encode(cfg, space)
            log_l = self._kde_logpdf(x, good, bw_good)
            log_g = self._kde_logpdf(x, bad, bw_bad)
            ei = log_l - log_g
            if ei > best_ei:
                best_ei = ei
                best_cfg = cfg
        return best_cfg

    def _init_brackets(self, space, total_budget):
        s_max = max(0, int(np.floor(np.log(total_budget) / np.log(self.eta))))
        s_max = min(s_max, 3)

        for s in range(s_max, -1, -1):
            n = int(np.ceil((s_max + 1) / (s + 1)) * self.eta ** s)
            n = min(n, total_budget)
            r = max(1.0 / self.eta ** s, 0.1)

            configs = [self._sample_from_model(space) for _ in range(n)]
            for cfg in configs:
                self._queue.append((cfg, r))

            self._brackets.append({
                "configs": configs,
                "fidelity": r,
                "scores": [None] * len(configs),
            })

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        # Track all completed trials for the TPE model
        if history:
            last = history[-1]
            vec = self._encode(last.config, space)
            self._all_trials.append((vec, last.score, last.budget))

        if not self._initialized:
            self._init_brackets(space, budget_left + len(history))
            self._initialized = True

        # Update bracket scores
        if history:
            last = history[-1]
            for bracket in self._brackets:
                for i, cfg in enumerate(bracket["configs"]):
                    if (bracket["scores"][i] is None
                            and cfg == last.config
                            and abs(bracket["fidelity"] - last.budget) < 0.05):
                        bracket["scores"][i] = last.score

                # Advance complete brackets
                if all(s is not None for s in bracket["scores"]):
                    if bracket["fidelity"] < 1.0 and len(bracket["configs"]) > 1:
                        # Successive halving
                        paired = list(zip(bracket["scores"], bracket["configs"]))
                        paired.sort(key=lambda x: x[0], reverse=True)
                        n_keep = max(1, len(paired) // self.eta)
                        survivors = paired[:n_keep]
                        new_fid = min(bracket["fidelity"] * self.eta, 1.0)
                        bracket["configs"] = [c for _, c in survivors]
                        bracket["scores"] = [None] * len(survivors)
                        bracket["fidelity"] = new_fid
                        for cfg in bracket["configs"]:
                            self._queue.append((cfg, new_fid))

        if self._queue:
            return self._queue.pop(0)

        # Generate new configs using TPE model at full fidelity
        cfg = self._sample_from_model(space)
        return cfg, 1.0
```
