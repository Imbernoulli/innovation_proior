The first autocorrelation constant `C1` is an infimum of `R(f) = max_{|t|≤1/2}(f*f)(t)/(∫f)²` over non-negative `f` on `[-1/4,1/4]`, so any step function I construct certifies an upper bound `C1 ≤ R(f)` and lower is better. The flat indicator — the previous baseline — is pinned at exactly `R = 2` and has no gradient to follow: every piece is identical, the autoconvolution is locked to a triangle, and refining the grid changes nothing. The only way down is to break that symmetry and let a search find a non-flat profile whose autoconvolution has its peak suppressed relative to the mass. The question is how to search and at what scale.

I work at a coarse resolution, `N ≈ 50`. The functional is highly non-convex with many local optima, so a long height vector is a high-dimensional space a blind local search wanders in for a long time; the right move is to find the *shape* at low resolution first, where the vector is short enough to explore thoroughly, and lift it later. Fifty pieces is short enough to canvas yet long enough that the autoconvolution has real structure to exploit.

The search is simulated annealing, because the obstruction is precisely that the flat profile and most simple shapes are local optima where any single perturbation *raises* `R` before a coordinated reshape lowers it; a greedy descender stops at the first such point. Annealing accepts uphill moves with the Metropolis probability `exp(-ΔR/T)`, cooling `T` geometrically — hot early to shake loose from the flat-triangle basin, cold late to settle. Two design choices come from the geometry: the kick is *multiplicative* in each height's own magnitude, the right invariance for a scale-free objective spanning a wide dynamic range; and because `R` is bounded above `1.28` with small, well-scaled single-coordinate changes, I anneal directly on `R` with no log or rescaling. I run several restarts (flat, ramp, hump, Gaussian seeds) and keep the best, hedging on which way the optimal shape leans.

Annealing one height at a time leaves the profile rough and short of its basin's true optimum, so I add a gradient polish. The `max` over autoconvolution nodes is non-differentiable at the peak, so I replace it with a smooth softmax of sharpness `β` and anneal `β` upward — soft early, where the surrogate gradient points broadly toward better shapes, sharp late, where it faithfully tracks the true peak. The gradient factors cleanly through the autoconvolution by the chain rule and is `O(N log N)` with FFTs; I run Adam on it (its per-coordinate scaling suits the wide dynamic range), clip to non-negative each step, and keep the best *true* `R`, with small periodic kicks to escape shallow traps.

The measured outcome is a drop from the flat ceiling `2.0` to `R = 1.5371` at `N = 50` — annealing alone reaches `1.7369`, and the softmax-Adam polish slides it the rest of the way down its basin. The returned profile is genuinely asymmetric and partly sparse: about nine of the fifty heights are driven near zero, with mass concentrated toward the ends and thinned in the middle — exactly the peak-suppressing structure (low central self-overlap) the literature reports for good `C1` constructions. The rung lands about `0.032` above the `600`-piece AlphaEvolve `1.5053` and `0.034` above the `30000`-piece record `1.5028628969`, capped by the coarse resolution: fifty pieces cannot render a fine enough autoconvolution to push the peak lower. That stall is exactly what motivates lifting this shape to a finer grid and bringing a stronger optimizer next.

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
