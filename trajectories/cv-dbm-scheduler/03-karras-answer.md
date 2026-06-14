**Problem (from step 2).** Uniform fixed loglinear's high-`t` blowup (gmean 7.78) but is the "no bet"
floor: its residual is DIODE at 15.0, where NFE = 3 leaves the entire near-data decade (`t < 0.5`) under
one coarse step. The near-data region — where the bridge curves and image content is committed — is
under-resolved.

**Key idea.** Warp the uniform grid through a power law: interpolate linearly in `t^{1/ρ}` and undo the
warp, `t_i = (t_max^{1/ρ} + u_i·(t_min^{1/ρ} − t_max^{1/ρ}))^ρ`. This shrinks `Δt` toward `t_min`,
matching the curvature `C_i` (concentrated at low `t`, since the deterministic step error is large at low
noise). The exponent is the one knob: `ρ = 1` is uniform, `ρ → ∞` is loglinear, so the family is a
continuous bridge between the two measured points. Fix `ρ = 7` — past the numerical-balance point
(`ρ ≈ 2–3`) because high-`t` accuracy is perceptually cheap and low-`t` accuracy is decisive — a single
constant, no per-dataset tuning.

**Why it works.** It returns the resolution uniform wasted on the nearly-straight high-`t` half to the
bending near-data region, but stops well short of loglinear's extreme: for DIODE the interior stop moves
from uniform's 0.50 to ≈ 0.041, deep into the near-data decade, while the high-`t` step lands at 0.041 —
not loglinear's γ = 100, 99%-in-one-step placement.

**Scaffold edit (bridge form, not EDM).** Despite the name, the canonical EDM schedule appends a terminal
*zero* over `linspace(0, 1, n)` levels. This is a *bridge*: the data endpoint is `t_min = 1e-4`, not 0, so
ramp across the **full** `n + 1` nodes (`linspace(0, 1, n + 1)`), warp from `t_max` to `t_min`, and pin
the terminal element to `t_min` (round-trip fix in the `(·)^{1/ρ}`/`(·)^ρ` pair) — no appended zero, no
query at `t = 0`. `ρ = 7`; length/monotonicity/terminal hold by construction.

**What to watch.** DIODE (uniform 15.0, near-data region under one step) has the most to gain and should
improve most. The NFE = 5 workloads (edges2handbags, ImageNet) already had two-to-three low-`t` steps
under uniform, so gains are smaller and not guaranteed; if `ρ = 7` over-bets low `t` where ImageNet's mask
needs high-`t` resolution, it could cost a little there. Karras should edge below uniform on the gmean,
driven by DIODE, without sliding back toward loglinear.

```python
def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    rho = 7.0
    ramp = torch.linspace(0, 1, n + 1)
    min_inv_rho = t_min ** (1 / rho)
    max_inv_rho = t_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas[-1] = t_min  # ensure exact terminal value (bridge data endpoint, not 0)
    return sigmas.to(device)
```
