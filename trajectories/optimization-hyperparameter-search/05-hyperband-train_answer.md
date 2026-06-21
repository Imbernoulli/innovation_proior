TPE landed exactly on the profile I predicted, and the contrast with DEHB is the whole point. `total_evals` came back pinned to the budget (50/40/40 — the model-only rung does no triage), and the NN final best *recovered*: $-3048$ mean, seed 42 at $-3064$ instead of DEHB's $-3086$, so removing the low-fidelity mis-promotion fixed the slip. But the convergence AUC distinctly *lost* to DEHB: SVM $0.876$ (DEHB $0.981$), NN $0.831$ (DEHB $0.935$), XGBoost $0.933$ (DEHB $0.949$). The cause is structural and visible: TPE spends 10 of its 40–50 evaluations on a random warm-up before the model turns on, and every evaluation thereafter is full price, so the best-so-far curve cannot climb early the way cheap multi-fidelity triage made DEHB's climb. A good *model* fixes final quality and reliability; cheap *multi-fidelity* fixes anytime convergence; neither has both. DEHB had the multi-fidelity half but bolted a model-free DE learner onto it and inherited the low-fidelity-correlation risk. This rung isolates the multi-fidelity skeleton *on its own*, done right — no DE, no model — so I can see what principled resource allocation buys alone and have a clean substrate to graft a model onto next.

I propose **Hyperband**. Strip the problem to the evaluation side: the waste random search and TPE both pay is that every configuration is trained to full resource before I look at it, so with budget $B$ I see only $B$ configurations. But most randomly drawn configs declare themselves hopeless after a sliver of resource — a learning rate an order of magnitude too high diverges in the first few NN iterations, a too-shallow ensemble plateaus after a fraction of its trees. The lever is not *which* config (TPE's lever) but *how much* I spend on each before I throw it out. The clean framing is a non-stochastic best-arm-identification bandit: a config is an arm, training it one more unit is a pull, and the partial loss $\ell_{i,j}$ converges to a terminal value $\nu_i$ at an *unknown rate*. I want the arm with the best $\nu_i$, but I cannot invoke i.i.d. concentration — the loss sequence is an arbitrary converging sequence, not noise around a mean.

Successive halving exploits this without knowing the rate: evaluate $N$ arms at the lowest resource, keep the top $1/\eta$, multiply the resource by $\eta$, repeat. The reason it works against an arbitrary converging sequence is an envelope argument worth spelling out, because it is what makes the method principled rather than a heuristic. Define $\gamma(j)$ as the smallest non-increasing function bounding how far any partial loss at resource $j$ can sit from its terminal value — it exists because the limits exist. Two arms with terminal gap $\Delta$ have envelopes that stop overlapping once $2\gamma(j) \le \Delta$, i.e. at resource $\tau = \gamma^{-1}(\Delta/2)$; at or beyond $\tau$ the ordering of the *partial* losses is guaranteed to match the ordering of the *terminal* losses. So a config separates from the best once the resource clears its gap-dependent threshold — small when curves converge fast or the gap is large, large when curves crawl or the gap is tiny. SH climbs the resource ladder exactly so that, rung by rung, more configs cross their separation threshold and the surviving ordering converges to the terminal one without ever needing to know $\gamma$.

But SH has one input it cannot set for itself: $N$. For a fixed budget, many configs each run cheaply (large $N$) is right *only if* the cheap resource ranks configs like the expensive one; if the cheap fidelity is misleading, large $N$ discards at low resource exactly the config that would have won at full. That is the configurations-versus-fidelity dilemma — the same failure that bit DEHB's NN ($-3086$). Hyperband refuses to guess $N$ and *hedges* across the whole spectrum: run several SH instances ("brackets"), each starting at a different number-of-configs-versus-initial-fidelity tradeoff. The most aggressive bracket throws a flood of configs at the cheapest fidelity (great if cheap predicts expensive); the least aggressive bracket is essentially random search at full fidelity (the safe fallback when cheap is misleading). By spanning the brackets, Hyperband covers the entire dilemma and is at most about (number of brackets) times slower than the best fixed choice — a guarantee no single SH setting has. The brackets are *complementary*: each is the optimal allocation under a different assumption about how predictive the cheap fidelity is, and since I do not know which holds, paying a logarithmic-factor overhead to run all of them buys robustness to being wrong about the one thing SH most needs to assume. On a faithful benchmark the aggressive bracket carries the run and the safe bracket is cheap insurance; on a misleading one the safe bracket protects the winner the aggressive bracket would have killed — the DEHB-NN failure turned into a hedge instead of a gamble. Critically, every config is still sampled *uniformly at random*: Hyperband is purely an allocation method, it never uses one evaluation's outcome to decide where to look next. That is exactly its limitation, and exactly why it is the right rung here — it measures the multi-fidelity skeleton in isolation, with random sampling, so the gap to the next rung will be attributable purely to adding a model.

The implementation is derived against the harness. With $\eta = 3$, compute $s_{\max} = \min(4, \lfloor\log_\eta(\text{total budget})\rfloor)$ — the cap at 4 keeps the number of brackets bounded so a small budget does not spawn more brackets than it can fund. For each bracket $s$ from $s_{\max}$ down to $0$, the number of configs is $n = \min(\lceil(s_{\max}+1)/(s+1)\rceil\cdot\eta^s,\ \text{total budget})$ and the initial fidelity is $r = \max(1/\eta^s, 0.1)$ — the $0.1$ floor is the loop's clip, so the cheapest rung is fidelity $0.1$, not arbitrarily small. Each bracket samples its $n$ configs uniformly and queues them all at fidelity $r$. The `suggest` loop is queue-driven: it pops the next queued $(\text{config}, \text{fidelity})$, and when a trial returns it matches it back to its bracket by exact `config == last.config` and $|\text{fidelity} - \text{last.budget}| < 0.05$, fills in the score, and when a bracket's current rung is fully scored, *advances* it — sort by score, keep the top $\lfloor\text{len}/\eta\rfloor$, multiply the fidelity by $\eta$ (capped at $1.0$), and re-queue the survivors at the higher fidelity. When the queue empties, fall back to a random full-fidelity draw. The exact-equality match is a real constraint of this harness, and it works precisely because the configs are reproducibly generated and never perturbed, so the returned config compares equal to the queued one — where DEHB, which perturbs vectors with DE, needed `np.allclose` on encoded vectors instead.

Hyperband should recover the *anytime* strength TPE lost, because the aggressive brackets surface a decent config from the cheap fidelity fast — so I expect its convergence AUC to beat TPE's clearly (TPE NN $0.831$, SVM $0.876$) and sit near DEHB's. Its `total_evals` should climb well past the budget like DEHB's, but more predictably: the bracket schedule is deterministic given the budget, so I expect a *fixed* `total_evals` per benchmark across seeds (around 105 on XGBoost, 95 on SVM and NN), unlike DEHB's seed-to-seed swing (65 to 354 on SVM) that came from DE's data-dependent promotion. That predictability is itself a virtue. On final best, Hyperband should match the competitive band but *not* clearly beat TPE, because it samples randomly — it allocates budget brilliantly but never aims, so against TPE's model-aimed configs it should be roughly even on final quality and win only on convergence. The NN remains the risk benchmark: Hyperband triages at low fidelity too, so if the NN's cheap rank is misleading, an aggressive bracket can still kill the eventual winner. If I see AUC recovered and stable, `total_evals` fixed and inflated, and final best even with TPE, the residual is named precisely — Hyperband's quality is capped by random sampling. It has the perfect skeleton and the wrong sampler, which forces the final move: keep this bracket schedule, but replace the random sampling with the TPE model, so multi-fidelity *allocation* and model-guided *selection* compound instead of being traded off.

```python
# EDITABLE region of scikit-learn/custom_hpo.py (lines 255-326) — step 5: Hyperband
class CustomHPOStrategy:
    """Hyperband: multi-fidelity with successive halving."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.eta = 3  # halving rate
        self.brackets = []  # list of (configs, fidelity, scores)
        self._initialized = False
        self._queue = []  # queue of (config, fidelity) to suggest

    def _init_brackets(self, space, total_budget):
        """Initialize Hyperband brackets (Successive Halving instances)."""
        s_max = max(0, int(np.floor(np.log(total_budget) / np.log(self.eta))))
        s_max = min(s_max, 4)  # cap brackets

        for s in range(s_max, -1, -1):
            n = int(np.ceil((s_max + 1) / (s + 1)) * self.eta ** s)
            n = min(n, total_budget)
            r = max(1.0 / self.eta ** s, 0.1)

            # Generate random configs for this bracket
            configs = [space.sample_uniform(self.rng) for _ in range(n)]
            # Queue low-fidelity evaluations
            for cfg in configs:
                self._queue.append((cfg, r))

            self.brackets.append({
                "configs": configs,
                "fidelity": r,
                "scores": [None] * len(configs),
                "round": 0,
                "s": s,
            })

    def _advance_bracket(self, bracket):
        """Advance a bracket: keep top 1/eta, increase fidelity."""
        configs = bracket["configs"]
        scores = bracket["scores"]

        # Sort by score, keep top 1/eta
        paired = [(s, c) for s, c in zip(scores, configs) if s is not None]
        if not paired:
            return
        paired.sort(key=lambda x: x[0], reverse=True)
        n_keep = max(1, len(paired) // self.eta)
        survivors = paired[:n_keep]

        new_fidelity = min(bracket["fidelity"] * self.eta, 1.0)
        bracket["configs"] = [c for _, c in survivors]
        bracket["scores"] = [None] * len(survivors)
        bracket["fidelity"] = new_fidelity
        bracket["round"] += 1

        for cfg in bracket["configs"]:
            self._queue.append((cfg, new_fidelity))

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        if not self._initialized:
            self._init_brackets(space, budget_left + len(history))
            self._initialized = True

        # Update bracket scores from history
        if history:
            last = history[-1]
            for bracket in self.brackets:
                for i, cfg in enumerate(bracket["configs"]):
                    if (bracket["scores"][i] is None
                            and cfg == last.config
                            and abs(bracket["fidelity"] - last.budget) < 0.05):
                        bracket["scores"][i] = last.score

                # Check if bracket round complete
                if all(s is not None for s in bracket["scores"]):
                    if bracket["fidelity"] < 1.0 and len(bracket["configs"]) > 1:
                        self._advance_bracket(bracket)

        # Return from queue
        if self._queue:
            return self._queue.pop(0)

        # Fallback: random
        return space.sample_uniform(self.rng), 1.0
```
