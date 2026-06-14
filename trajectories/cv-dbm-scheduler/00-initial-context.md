## Research question

A trained diffusion *bridge* model turns one image into another — edges into a handbag, a masked
ImageNet crop into a completed scene, a depth map into a photo — by running a sampler that marches a
state `x_t` along a one-dimensional time axis `t`, from the noisy/source endpoint `t_max` down to the
data endpoint `t_min`, calling a frozen denoiser once per stop. The cost of a sample is the number of
denoiser calls (NFE): each call is a full U-Net forward pass, and the calls are sequential. Here the
budget is brutal — **NFE = 5** (and NFE = 3 on one workload) — so the sampler takes only a handful of
steps from `t_max` to `t_min`.

The denoiser is trained and frozen, and the per-step update rule of the bridge sampler is fixed. The
only remaining freedom is **which `t` values to evaluate the denoiser at** — a short, ordered list of
times `t_max = t_0 > t_1 > ... > t_n = t_min` at which the sampler stops, calls the denoiser, and
takes a step. With only `n = 5` steps this placement is quality-critical even though nothing about the
model or the update rule changes. The single thing being designed is the **time schedule**: given the
budget `n` and the endpoints `t_min`, `t_max`, produce the `n + 1` evaluation times that spend those
few calls well. Hard requirements: the list strictly decreases, starts at `t_max`, ends *exactly* at
`t_min`, has length `n + 1`, and carries **no constants secretly tuned to one dataset** — it must
generalize across image-to-image workloads. What that list should be is the open question.

## Prior art before the first rung (sampling-schedule lineage)

The first rung reacts to the schedules that few-step diffusion samplers had already converged to. These
are the placements that precede the ladder; each bakes in a different fixed bet about where the
trajectory is hard, and each is a candidate fill of the same `n + 1`-node contract below.

- **Uniform / linear placement (the dense-sampler inheritance).** Space the times evenly in the
  sampler's native `t`: constant `Δt = (t_max − t_min)/n`. It is the direct downsampling of the
  thousand-step sampler's even integer spacing, carries no shape hyperparameter, and hits both
  endpoints exactly. Gap: it assumes the trajectory curves *nowhere* — it pays equal resolution to the
  near-noise region (cheap, the start is almost arbitrary) and the near-data region (decisive), so at
  NFE = 5 it under-resolves exactly where error becomes a visible artifact.
- **Power-law / EDM rho-schedule (Karras et al., NeurIPS 2022, arXiv:2206.00364).** Warp a uniform
  grid through `t^{1/ρ}`: long steps at high noise, short steps at low noise, with `ρ = 7`. Derived by
  measuring per-step truncation error (large at low noise) and then deliberately over-allocating to low
  noise because high-noise accuracy is perceptually cheap. Gap: it was tuned for the *variance-exploding
  diffusion* trajectory ending at `σ = 0`; on a bridge the data endpoint is `t_min ≠ 0` and the source
  end carries real conditioning structure, so the heavy low-noise concentration of `ρ = 7` may over-bet
  one end of a curve whose shape is set by the bridge, not the EDM ODE.
- **Cosine placement (after Nichol & Dhariwal, ICML 2021, arXiv:2102.09672).** Originally a
  training-time noise schedule, reused to derive sampling times: a smooth S-curve that starts and ends
  flat. Read as a time warp, it concentrates steps in the *middle* of `[t_min, t_max]`, where the bridge
  trajectory bends most, and eases gently into both endpoints. Gap: it is a borrowed forward-process
  curve, not a curve derived from sampling integration error — whether its mid-range concentration
  matches where *this* bridge sampler actually needs resolution is exactly what has to be measured.
- **Log-linear / geometric placement.** Space the times evenly in `log t` (constant ratio
  `t_{i-1}/t_i`), the standard noise ladder of annealed score-based samplers. Equalizes the overlap
  between adjacent perturbed distributions — the right object when each level *warm-starts* the next.
  Gap: equal multiplicative spacing crams almost all the nodes against `t_min` and leaves the
  high-`t` region with one enormous jump; on a frozen few-step bridge sampler that does not re-equilibrate
  per level, that single coarse high-`t` step is an uncovered gap rather than a saved rung.

## The fixed substrate

The bridge sampler is frozen and must not be touched. A trained, frozen denoiser `D(x_t, t)` is queried
once per stop; the deterministic DBIM update (η = 0) computes its step coefficients from the bridge's
own noise schedule (the `get_abc` / `get_alpha_rho` coefficients of a VP or I2SB bridge) at the current
and next scheduled times and marches `x` from `t_max` toward `t_min`. The number of steps follows from
the budget (`n = NFE − 1`), and the endpoints are the bridge's own `t_min`, `t_max`. Dataset handling,
the denoiser-call budget, the sampler update rule, and the FID computation are all off-limits. The only
quantity the schedule controls is **where along `[t_min, t_max]` the `n + 1` stops land**.

## The editable interface

Exactly one region is editable — the body of `get_sigmas_uniform(n, t_min, t_max, device)` in
`dbim-codebase/ddbm/karras_diffusion.py` (the legacy name is fixed; despite it the intended
contribution is a non-trivial schedule *curve*, not a literal uniform grid). Every method on the ladder
is a fill of this same contract and nothing else:

- **Length:** return a 1-D `torch.Tensor` of exactly `n + 1` elements.
- **Monotonicity:** strictly decreasing from `t_max` to `t_min`.
- **Terminal:** the final element (index `n`) equals `t_min` *exactly*.
- **Device:** the returned tensor lives on `device`.

The sampler reads this tensor as its ordered stop times `ts`, with `ts[0] = t_max`, `ts[n] = t_min`,
and `n` deterministic steps between them. For NFE = 5, `n = 4` (five stops); on the NFE = 3 workload,
`n = 2`. The starting point is the scaffold default below — the no-information **uniform** grid. Each
method on the ladder replaces exactly this function body and nothing else.

```python
# EDITABLE region of dbim-codebase/ddbm/karras_diffusion.py — default fill (uniform/linear)
def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    """
    Schedule contract for the frozen few-step bridge sampler:
      1. Length:      1-D tensor of length n + 1.
      2. Monotonic:   strictly decreasing from t_max to t_min.
      3. Terminal:    final element (index n) == t_min exactly.
      4. Device:      returned on the requested device.
    For this task n = 4 (NFE = 5); n = 2 on the NFE = 3 workload.
    """
    # No-information default: constant spacing in t, exact at both endpoints.
    return torch.linspace(t_max, t_min, n + 1).to(device)
```

## Evaluation settings

Three image-to-image bridge workloads spanning the difficulty range — **edges2handbags** (e2h, a VP
bridge, NFE = 5), **ImageNet center-inpainting** (an I2SB bridge, NFE = 5), and **DIODE** depth→image
(a VP bridge, NFE = 3) — each at seed 42. The same closed-form schedule is dropped into all three
without per-dataset constants. Metric: **FID**, lower is better, reported per workload
(`best_fid_edges2handbags`, `best_fid_Imagenet`, `best_fid_DIODE`); the task aggregate is the
**geometric mean** of the three. All endpoints share `t_min = 1e-4`, `t_max = 1.0`.
