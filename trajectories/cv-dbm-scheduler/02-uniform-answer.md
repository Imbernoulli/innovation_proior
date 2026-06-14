**Problem (from step 1).** The geometric ladder collapsed (gmean 10.97; DIODE 17.0, ImageNet 13.7)
because it left one oversized high-`t` step that a frozen DBIM marcher — which has no warm-start
re-equilibration to protect — could not absorb. The sampler's currency is not adjacent-level overlap; it
is "no single step oversized across a region that bends."

**Key idea.** Place the stops at constant spacing in the sampler's native `t`: `Δt = (t_max − t_min)/n`
everywhere. The deterministic DBIM step is Euler-style integration with local error `≈ (Δt_i)^2·C_i`; if
the curvature `C_i` were known, optimal placement is `Δt_i ∝ 1/C_i`, but with three different bridges and
no per-dataset constants allowed, the curvature is unknown. The problem is then **minimax**: any
non-uniform grid has a largest interval `Δt_max`, into which an adversarial curvature budget dumps error
`≈ (Δt_max)^2·B`; the uniform grid is the unique placement with no oversized interval to exploit.

**Why it works.** Uniform is the minimax-optimal "no bet" under no curvature knowledge, and it removes the
exact vulnerability loglinear exposed (its huge high-`t` step *was* the largest interval, the bridge's
high-`t` curvature *was* the adversary). It is also the only fill with **no shape hyperparameter** (just
the count `n`), so it transfers across the three workloads with no forbidden tuning. Spacing evenly in any
warped coordinate (`t^{1/ρ}`, `log t`, cosine) is itself a curvature assumption; `t` is the integrator's
native variable, so constant `Δt` presupposes nothing.

**Scaffold edit / hyperparameters.** None beyond `n`. `torch.linspace(t_max, t_min, n + 1)` hits both
endpoints exactly, so — unlike the geometric fill — it needs no terminal patch: length `n + 1`, strictly
decreasing, terminal exactly `t_min`, on `device`, all by construction.

**What to watch.** The catastrophic workloads should recover most: DIODE (loglinear 17.0, its lone step
covered 99% of the interval) and ImageNet (13.7) should drop sharply once the high-`t` jump is split into
equal halves; edges2handbags (loglinear 5.634, least bad) should improve modestly. Uniform is the floor,
not the ceiling — a correctly curvature-matched warp would beat it, which is the question the next rung
asks.

```python
def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    return torch.linspace(t_max, t_min, n + 1).to(device)
```
