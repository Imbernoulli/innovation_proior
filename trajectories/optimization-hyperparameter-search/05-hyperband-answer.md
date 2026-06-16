**Problem.** TPE recovered final quality (NN best back to −3048) but lost convergence AUC to DEHB
(SVM 0.876 vs 0.981) because its 10-config warm-up plus full-price evaluations cannot front-load the
best-so-far curve. A model fixes quality; cheap multi-fidelity fixes anytime convergence; neither has both.
This rung isolates the multi-fidelity skeleton done right — no model, no DE — so the next rung's gain is
attributable purely to adding a model.

**Key idea.** Hyperband. Successive halving triages: evaluate many configs at low fidelity, keep the top
1/η, raise the fidelity, repeat — the lever is *how much* to spend per config, not which. SH cannot set its
own N (configs-vs-fidelity tradeoff), so Hyperband *hedges*: run several brackets spanning the spectrum, from
"flood of cheap configs" (good if cheap predicts expensive) to "random search at full fidelity" (safe
fallback). This covers the dilemma with an at-most-(#brackets)× slowdown guarantee. Every config is still
sampled uniformly at random — pure allocation, no aiming — which is exactly the limitation the final rung
removes.

**Why this and not DEHB.** Same multi-fidelity benefit, but stripped to the allocation skeleton with random
sampling (no DE learner), so it measures what principled resource allocation buys alone. The deterministic
bracket schedule also makes `total_evals` stable across seeds, unlike DEHB's data-dependent swing.

**Implementation notes (task-specific).** `s_max = min(4, ...)` caps brackets for small budgets; initial
fidelity `r = max(1/η^s, 0.1)` respects the loop's 0.1 floor. Queue-driven; trials match back by *exact*
`config == last.config` and `|fidelity − budget| < 0.05` (works because random configs are never perturbed).

**Hyperparameters.** `η = 3`, brackets capped at 5 (`s_max ≤ 4`), per-bracket `n = min(ceil((s_max+1)/(s+1))
·η^s, budget)`, keep top `len//η` per advance, fidelity floor 0.1.

**What to watch.** Convergence AUC back above TPE's on every benchmark (esp. NN) and near DEHB's;
`total_evals` high and *stable across seeds* (~105/95/95 — the deterministic tell); final best in the band but
not clearly beating TPE, since random sampling allocates but never aims. That cap forces grafting the TPE
model onto this skeleton next.

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
