**Problem.** Maximize `R = ||f*f||_2^2 / (||f*f||_inf·||f*f||_1)` over non-negative step functions. The
coarse `N = 20` annealing rung reached `0.8848`, capped by resolution. This rung lifts that shape onto a
finer grid and refines with a gradient method.

**Key idea.** A **hierarchical lift**. Upscaling a height vector by replacing each piece with `k` copies is
*exactly ratio-preserving* — same function, same `R` — so it costs nothing and hands the optimizer a finer
canvas. After each lift, refine with **β-annealed Adam gradient ascent**: the non-differentiable
`||f*f||_inf = max_j L_j` is replaced by a softmax with sharpness `β` annealed from soft to sharp (broad
search early, faithful `max` late); the analytic gradient of the surrogate factors through the
self-convolution and is computed with FFTs in `O(N log N)`. Adam's per-coordinate scaling handles the wide
height dynamic range (spike `≈28×` the shoulder); a small multiplicative kick right after each upscale
breaks the flat-block degeneracy so the gradient has something to grab. Ladder: optimized `20`-piece seed →
`×5` to `100` → `×5` to `500`, refining at each level.

**Why these choices.** Upscaling alone leaves a degenerate plateau (flat blocks, zero symmetric gradient),
so the kick-plus-gradient is what carves the asymmetric fine structure that a coarse vector cannot
represent — which is precisely what flattens the autoconvolution's cap and raises `R`. The softmax-`β` anneal
turns a non-smooth objective into a differentiable one whose sharp-`β` optimum is the true ratio's optimum.
Adam (not plain gradient ascent) is needed because the heights span a wide range and a fixed step either
crawls on the spike or blows up the shoulder. The best *true* `R` is tracked (not the surrogate), since they
differ slightly. The honest ceiling of this rung is the high `0.89`s: the published `~0.9016` step-function
results spent vastly more compute (Boyer–Li `~10^6` trajectories), so the last fraction of a percent is left
for the endpoint rung.

**Hyperparameters / contract.** `N: 20 → 100 → 500` by `×5` lifts; per level two Adam passes,
`β` annealed geometrically `~3N → ~80N`, lr `~0.012 → 0.004` decaying with `N`, multiplicative kick `~0.03–0.06`
after each lift, `8000–10000` iters/pass; all seeds fixed. Deterministic given the fixed seeds; runs in
`~100 s`. Returns `500` non-negative heights.

```python
import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import minimize

def autoconv_ratio(v):
    v = np.clip(np.asarray(v, float), 0.0, None); N = len(v)
    c = fftconvolve(v, v); L = np.zeros(2 * N + 1); L[1:2 * N] = c
    Lj, Ljp = L[:-1], L[1:]
    l2 = (1/3) * np.sum(Lj**2 + Lj*Ljp + Ljp**2); l1 = 0.5*np.sum(Lj + Ljp); linf = np.max(L)
    return float(l2 / (linf * l1))

def _obj_grad(v, beta):                          # smooth -R surrogate + analytic gradient (FFT)
    v = np.abs(v); N = len(v); c = fftconvolve(v, v)
    L = np.zeros(2*N+1); L[1:2*N] = c; Lj, Ljp = L[:-1], L[1:]
    A = (1/3)*np.sum(Lj**2 + Lj*Ljp + Ljp**2); C = 0.5*np.sum(Lj + Ljp)
    m = np.max(L); w = np.exp(beta*(L - m)); Z = w.sum(); B = m + np.log(Z)/beta
    Q = A / (B * C)
    dA = np.zeros(2*N+1); dA[:-1] += (1/3)*(2*Lj + Ljp); dA[1:] += (1/3)*(2*Ljp + Lj)
    dC = np.zeros(2*N+1); dC[:-1] += 0.5; dC[1:] += 0.5; dB = w / Z
    dL = dA/(B*C) - A*dB/(B*B*C) - A*dC/(B*C*C)
    dc = dL[1:2*N]; g = 2*fftconvolve(dc, v[::-1])[N-1:N-1+N]   # chain rule through self-convolution
    return Q, g

def _adam(v, beta_lo, beta_hi, iters, lr, perturb, seed):
    rng = np.random.default_rng(seed); v = np.abs(v).copy()
    if perturb > 0: v = np.abs(v * (1 + perturb * rng.standard_normal(len(v))))   # break plateau
    m = np.zeros_like(v); s = np.zeros_like(v); b1, b2, eps = 0.9, 0.999, 1e-8
    best, bv = autoconv_ratio(v), v.copy()
    for it in range(iters):
        beta = beta_lo * (beta_hi / beta_lo) ** (it / iters)     # anneal softmax sharpness
        _, g = _obj_grad(v, beta)
        m = b1*m + (1-b1)*g; s = b2*s + (1-b2)*g*g
        v = np.abs(v + lr * (m/(1-b1**(it+1))) / (np.sqrt(s/(1-b2**(it+1))) + eps))
        r = autoconv_ratio(v)
        if r > best: best, bv = r, v.copy()       # track true R, not the surrogate
    return bv

def _coarse20(seed):                              # rung-2 style coarse seed at N=20
    rng = np.random.default_rng(seed); x = np.linspace(0, 1, 20)
    def sneg(v, beta):
        v = np.abs(v); N = len(v); c = fftconvolve(v, v); L = np.zeros(2*N+1); L[1:2*N] = c
        Lj, Ljp = L[:-1], L[1:]; A = (1/3)*np.sum(Lj**2+Lj*Ljp+Ljp**2); C = 0.5*np.sum(Lj+Ljp)
        m = np.max(L); return -A / ((m + np.log(np.sum(np.exp(beta*(L-m))))/beta) * C)
    best, bv = -1.0, None
    for r in range(60):
        v0 = np.exp(-(x-0.5)**2/0.12) + 0.15*rng.random(20) if r % 2 == 0 else rng.random(20)+0.1
        for beta in [5, 15, 40, 120, 400, 1500, 6000]:
            v0 = np.abs(minimize(sneg, v0, args=(beta,), method='L-BFGS-B',
                                 options={'maxiter': 4000}).x)
        if autoconv_ratio(v0) > best: best, bv = autoconv_ratio(v0), v0
    return bv

def construct():
    v = _coarse20(2024)
    v = np.repeat(np.abs(v), 5); N = len(v)                          # 20 -> 100
    v = _adam(v, N*3, N*43, 8000, 0.012, 0.06, 100)
    v = _adam(v, N*3, N*83, 8000, 0.006, 0.00, 101)
    v = np.repeat(np.abs(v), 5); N = len(v)                          # 100 -> 500
    v = _adam(v, N*3, N*63, 10000, 0.008, 0.03, 200)
    v = _adam(v, N*3, N*123, 10000, 0.004, 0.00, 201)
    return [float(x) for x in np.abs(v)]

if __name__ == "__main__":
    print("R =", round(autoconv_ratio(construct()), 6))   # 0.894706
```
