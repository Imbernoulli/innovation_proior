**Problem.** Construct a non-negative step function minimizing `R = 2N·max_k(a*a)_k / (Σ a_n)^2` (lower = tighter
upper bound on `C1`). The flat profile is pinned at `R = 2` with no gradient — every later gain must come from
breaking that symmetry. This rung takes the first real drop off the ceiling at a coarse `N ≈ 50`.

**Key idea.** Break the flat symmetry by *simulated annealing* on the heights, then *gradient-polish* the result.
The anneal proposes a multiplicative kick to one randomly chosen height and accepts it by the Metropolis rule —
always if `R` drops, with probability `exp(-ΔR/T)` if it rises — cooling the temperature `T` and shrinking the
kick geometrically; several restarts (flat, ramp, hump, Gaussian seeds) keep the best shape found. Accepting uphill
moves is essential, because the flat profile and most simple shapes are local optima where any single perturbation
*raises* `R` before a coordinated reshape lowers it. The settled shape is then slid to the bottom of its basin by
`β`-annealed projected-Adam ascent on a *softmax* surrogate of the `max` (the hard peak is non-differentiable; the
softmax is smooth and is annealed sharp so it tracks the true peak), keeping the best *true* `R`.

**Why these choices.** The kick is multiplicative because the objective is scale-free and the good profiles span a
wide dynamic range, so a tall height and a thin one should move on comparable *relative* terms. Annealing directly
on `R` (no log/rescale) is fine because `R` is bounded above `1.28` and its single-coordinate changes are small and
well-scaled. The optimizer concentrates mass asymmetrically and drives several heights toward zero — the signature
of a peak-suppressing construction, which lowers `R` by reducing the central self-overlap of the function with
itself. The coarse `N` is deliberate: short enough for the stochastic search to canvas the shape space, with the
fine-grid refinement deferred to the next rung.

**Hyperparameters / contract.** `N = 50`; anneal `40000` proposals/restart, `T: 0.05 → ~0` (×0.9995/step), kick
scale `~0.3·(T/T0)+0.01`; seeds `{flat, ramp, hump, gaussian}`; Adam polish `6000` steps, `lr 0.01`, `β: 20 →
5000`, periodic kicks `0.02` every `1500`. Stochastic but seeded (`seed=0/1`), so the returned vector and its `R`
are reproducible.

```python
import numpy as np, time
from scipy.signal import fftconvolve

def R_eval(a):
    a = np.clip(np.asarray(a, float), 0.0, 1000.0); N = len(a)
    b = fftconvolve(a, a); s = float(np.sum(a))
    return float("inf") if s < 0.01 else 2.0 * N * float(np.max(b)) / (s * s)

def _grad_surrogate(a, beta):
    n = len(a); b = fftconvolve(a, a); s = float(np.sum(a))
    cmax = float(np.max(b)); w = np.exp(beta * (b - cmax)); w /= w.sum()
    full = fftconvolve(w, a[::-1]); dM = 2.0 * full[n - 1 + np.arange(n)]
    R = R_eval(a); dR = (2.0 * n / s**2) * dM - (2.0 * R / s) * np.ones(n)
    return R, dR

def _adam(a, iters, lr, b0, b1, seed, kick_every, kick_scale):
    rng = np.random.default_rng(seed); a = np.clip(a, 1e-9, None)
    m = np.zeros_like(a); v = np.zeros_like(a); best = a.copy(); bR = R_eval(a)
    for t in range(1, iters + 1):
        beta = np.exp(np.log(b0) + (np.log(b1) - np.log(b0)) * (t - 1) / max(1, iters - 1))
        R, g = _grad_surrogate(a, beta)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        a = np.clip(a - lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-12), 0.0, 1000.0)
        if R < bR: bR = R; best = a.copy()
        if kick_every and t % kick_every == 0 and t < iters - 1:
            a = np.clip(best * (1 + kick_scale * rng.standard_normal(len(a))), 0.0, 1000.0)
    return best, R_eval(best)

def _anneal(n, iters, seed):
    rng = np.random.default_rng(seed); x = np.linspace(0, 1, n)
    inits = [np.ones(n), 0.5 + x, 1.0 + 0.5 * np.sin(np.pi * x), np.exp(-((x - 0.5)**2) / 0.1)]
    best = None; bR = np.inf
    for a0 in inits:
        a = np.clip(a0.astype(float), 1e-6, None); Rc = R_eval(a); T = 0.05
        for _ in range(iters):
            T *= 0.9995; scale = 0.3 * (T / 0.05) + 0.01; cand = a.copy(); j = rng.integers(n)
            cand[j] = max(0.0, cand[j] * (1.0 + scale * rng.standard_normal())); Rn = R_eval(cand)
            if Rn < Rc or rng.random() < np.exp(-(Rn - Rc) / max(T, 1e-6)): a = cand; Rc = Rn
            if Rc < bR: bR = Rc; best = a.copy()
    return best

def construct(N: int = 50):
    a = _anneal(N, 40000, seed=0)
    a, _ = _adam(a, 6000, 0.01, 20.0, 5000.0, seed=1, kick_every=1500, kick_scale=0.02)
    return [float(x) for x in a]

if __name__ == "__main__":
    print(R_eval(construct()))   # ~1.5371
```
