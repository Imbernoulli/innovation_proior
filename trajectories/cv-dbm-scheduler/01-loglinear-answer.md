**Problem.** With a frozen bridge denoiser and a frozen DBIM update rule, the only freedom under NFE = 5
is *where* the `n + 1` stops land on `[t_min, t_max]`. The schedule must strictly decrease, end exactly
at `t_min`, and carry no per-dataset constants. The scaffold default is even spacing in `t` (a "no bet"
placement); I want to start from the schedule in this family that has a real derivation behind it.

**Key idea.** Equal spacing in `log t` — constant ratio `t_{i-1}/t_i = γ`, a geometric ladder. In
*annealed* sampling this is the spacing that equalizes adjacent-level overlap: in `R^D` the mass of
`N(0, σ^2 I)` sits on a thin shell of radius `√D·σ`, and the chance a level-`i` draw lands in level
`(i-1)`'s high-density band depends only on `γ = σ_{i-1}/σ_i`, so equal overlap across the ladder forces
equal `γ` → geometric → linear in `log σ`. Drop that closed form into `get_sigmas_uniform`.

**Why (and the caveat).** Geometric spacing packs the stops against the data endpoint `t_min` — three of
the five stops resolve the near-data region where image content is committed. But the derivation assumes
a *warm-start, re-equilibrating* sampler; the frozen DBIM marcher (η = 0) has neither. With
`t_max = 1.0`, `t_min = 1e-4`, `n = 4`, the ratio is `γ = 10` and the schedule is
`{1.0, 0.1, 0.01, 0.001, 1e-4}`: the first denoiser call must carry the state across `1.0 → 0.1` — 90%
of the interval — in one deterministic step. On a curving, conditioned bridge that single oversized
high-`t` step is uncovered.

**Scaffold edit / hyperparameters.** No hyperparameters beyond the count `n`. Floor `t_min` before the
log (`max(t_min, 1e-10)`, since `log 0 = −∞`); space evenly across `n + 1` nodes in the log domain;
exponentiate; pin the terminal node to `t_min` exactly (round-trip ULP fix). Monotonicity, length, and
endpoints all hold by construction.

**What to watch.** If near-data packing is what matters, FID drops versus the uniform default. If instead
the lone coarse high-`t` opening step dominates on a curving conditioned bridge — the likely outcome,
since the frozen sampler has no overlap mechanism to reward — this principled-looking ladder lands
*behind* uniform, worst on the workloads with the most high-`t` bridge structure.

```python
def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    import math
    log_max = math.log(t_max)
    log_min = math.log(max(t_min, 1e-10))               # floor before log: log(0) = -inf
    sigmas = torch.exp(torch.linspace(log_max, log_min, n + 1))  # even spacing in log t
    sigmas[-1] = t_min                                   # pin the data endpoint exactly
    return sigmas.to(device)
```
