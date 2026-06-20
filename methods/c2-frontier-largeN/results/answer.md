**Problem.** Maximize `R = ||f*f||_2^2 / (||f*f||_inf·||f*f||_1)` over non-negative step functions. The
`N = 500` hierarchical-gradient rung reached `0.894706` with the gradient still moving. This endpoint lifts
once more to thousands of pieces and spends a long, sharply-annealed refinement to reach the published
step-function frontier.

**Key idea.** Lift the optimized `500`-piece profile `×4` to `N = 2000` (free, ratio-preserving), kick to
break the flat-block plateau, then run a **long β-annealed Adam ladder** with the budget and schedule tuned
for high resolution: (1) a moderate-`β`, kicked pass to let the lifted shape reorganize; (2) a long
(`~40000`-step) pass with `β` annealed up to several hundred×`N` and periodic small kicks, which carves the
irregular fine structure the published step-function constructions rely on; (3) a low-lr, sharpest-`β`
polish. Best *true* `R` tracked throughout. This reaches the band of Boyer–Li (`575`-step `0.901564`) and
Jaech–Joseph (`539`-step `~0.9016`).

**Why these choices.** At `2000` pieces the optimum is a finer, more irregular shape than at `500`, so the
gradient keeps paying for tens of thousands of steps — hence the long budget (affordable because the FFT
evaluator is `O(N log N)`). The softmax-`L_inf` `β` must be pushed *much* sharper here: with a tall spike
the node values span a wide range, and a `β` adequate at `500` lets the surrogate peak fall below the true
peak, so the optimizer chases a slightly wrong objective — annealing `β` to `~400N` makes the surrogate
track the hard `max`. Periodic (not just initial) kicks act as mild restarts that keep the long run out of
shallow traps, shrinking as it sharpens. The honest endpoint is the step-function frontier this reaches;
the absolute record (AlphaEvolve-V2's `~50000`-step `0.96102`) sits far above and was bought by an
evolutionary search with orders of magnitude more compute, so the remaining gap is the open part of the
problem.

**Hyperparameters / contract.** `N = 2000` (via `20→100→500→2000` lifts); endpoint passes at `2000`:
reorganize (`12000` it, lr `0.006`, kick `0.03`), grind (`10000`+`40000` it, lr `0.003→0.002`, `β` to
`~400N`, periodic kicks `0.006–0.008`), polish (`20000` it, lr `0.0008`, `β` to `~800N`); all seeds fixed.
Deterministic; runs in `~130 s`. Returns `2000` non-negative heights.

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

def _obj_grad(v, beta):
    v = np.abs(v); N = len(v); c = fftconvolve(v, v)
    L = np.zeros(2*N+1); L[1:2*N] = c; Lj, Ljp = L[:-1], L[1:]
    A = (1/3)*np.sum(Lj**2 + Lj*Ljp + Ljp**2); C = 0.5*np.sum(Lj + Ljp)
    m = np.max(L); w = np.exp(beta*(L - m)); Z = w.sum(); B = m + np.log(Z)/beta
    Q = A / (B * C)
    dA = np.zeros(2*N+1); dA[:-1] += (1/3)*(2*Lj + Ljp); dA[1:] += (1/3)*(2*Ljp + Lj)
    dC = np.zeros(2*N+1); dC[:-1] += 0.5; dC[1:] += 0.5; dB = w / Z
    dL = dA/(B*C) - A*dB/(B*B*C) - A*dC/(B*C*C)
    dc = dL[1:2*N]; g = 2*fftconvolve(dc, v[::-1])[N-1:N-1+N]
    return Q, g

def _adam(v, beta_lo, beta_hi, iters, lr, perturb, seed, kick_every=0, kick=0.0):
    rng = np.random.default_rng(seed); v = np.abs(v).copy()
    if perturb > 0: v = np.abs(v * (1 + perturb * rng.standard_normal(len(v))))
    m = np.zeros_like(v); s = np.zeros_like(v); b1, b2, eps = 0.9, 0.999, 1e-8
    best, bv, N = autoconv_ratio(v), v.copy(), len(v)
    for it in range(iters):
        beta = beta_lo * (beta_hi / beta_lo) ** (it / iters)
        _, g = _obj_grad(v, beta)
        m = b1*m + (1-b1)*g; s = b2*s + (1-b2)*g*g
        v = np.abs(v + lr * (m/(1-b1**(it+1))) / (np.sqrt(s/(1-b2**(it+1))) + eps))
        if kick_every and it > 0 and it % kick_every == 0:
            v = np.abs(v * (1 + kick * rng.standard_normal(N)))      # periodic mild restart
        r = autoconv_ratio(v)
        if r > best: best, bv = r, v.copy()
    return bv

def _coarse20(seed):
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
    v = np.repeat(np.abs(v), 5); N = len(v)                              # 20 -> 100
    v = _adam(v, N*3, N*43, 8000, 0.012, 0.06, 100); v = _adam(v, N*3, N*83, 8000, 0.006, 0.0, 101)
    v = np.repeat(np.abs(v), 5); N = len(v)                             # 100 -> 500
    v = _adam(v, N*3, N*63, 10000, 0.008, 0.03, 200); v = _adam(v, N*3, N*123, 10000, 0.004, 0.0, 201)
    v = np.repeat(np.abs(v), 4); N = len(v)                             # 500 -> 2000 (endpoint)
    v = _adam(v, N*3, N*80,  12000, 0.006,  0.03, 300)                  # reorganize
    v = _adam(v, N*3, N*200, 10000, 0.003,  0.0,  301, kick_every=4000, kick=0.008)
    v = _adam(v, N*3, N*400, 40000, 0.002,  0.0,  500, kick_every=5000, kick=0.006)  # long grind
    v = _adam(v, N*4, N*800, 20000, 0.0008, 0.0,  501)                  # polish
    return [float(x) for x in np.abs(v)]

if __name__ == "__main__":
    print("R =", round(autoconv_ratio(construct()), 6))   # 0.901804
```
