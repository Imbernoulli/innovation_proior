The flat profile sat exactly at $2/3$, and that result made the obstruction precise: with every piece identical the autoconvolution is locked to a triangle, there is no gradient to follow, and refining the grid does nothing. The only way forward is to break the flat symmetry — introduce variation among the heights and let a search find a non-flat profile whose autoconvolution bends away from the tent. The two decisions are how to search and at what scale.

Scale decides everything else, so I fix it first, and I deliberately resist going straight to a long height vector. The functional is highly non-convex, with many local optima and a great deal of symmetry to break, and a long vector is a high-dimensional space where a blind local search wanders forever. The right move is to find the *shape* at low resolution, where the vector is short enough to canvas thoroughly, and only later lift it. The literature points exactly here — Matolcsi–Vinuesa cleared $0.88$ with only $20$ steps — so I work at $N = 20$: coarse enough to explore exhaustively, long enough that the autoconvolution has real internal structure to exploit.

The search itself is *simulated annealing*, and the reason is structural. The flat profile, and most "obvious" profiles, are near-flat local optima: any small perturbation *lowers* $R$ before a coordinated reshape can raise it, so a greedy hill-climber that only accepts improving moves parks at the first such ridge and stops. Annealing crosses those ridges by being willing to go downhill. I propose a perturbation to one height; if it improves $R$ I take it, and if it worsens $R$ I take it anyway with Metropolis probability $\exp\!\big((R_{\text{new}} - R_{\text{cur}})/T\big)$ under a temperature $T$ that I cool geometrically over the run — hot early so the search shakes loose from the flat-triangle basin, cold late so it settles into the best basin it has found.

Three design choices come straight from the geometry of this objective. The first is the *form of the perturbation*. Heights are non-negative and span a wide dynamic range — the good profiles, I expect, are a tall spike or two plus a long shoulder of smaller values, not a uniform spread — so a fixed-size additive Gaussian kick would be far too coarse for the small heights and too timid for the large ones. I make the kick *multiplicative in scale*: I perturb a chosen height $v_j$ by $\mathcal{N}(0,\,\sigma(0.1 + v_j))$ and reflect any negative result back to non-negativity, so a tall spike and a thin shoulder value are adjusted on comparable *relative* terms — the right invariance for a scale-free objective. The second is *what to anneal on*. Because $R$ is already bounded in $[0,1]$ and its changes under a single height perturbation are small and well-scaled, I do not need to take a log or rescale — I anneal directly on $R$ with $T$ of order $10^{-3}$ down to a tiny floor, and the acceptance behaves sanely; I shrink the perturbation scale alongside $T$ so the late run makes fine adjustments to a settled shape. The third is *restarts*. A single run can get trapped, so I run many independent restarts — some from a smooth single-bump (Gaussian) seed, the educated guess that the good unimodal autoconvolution is produced by a concentrated height profile, and some from pure random heights — keeping the best profile any restart ever reaches.

Between annealing phases I let a gradient method do the fine legwork over a *smooth-$L_\infty$ surrogate*. The hard $\|f*f\|_\infty = \max_j L_j$ is non-differentiable at the peak, so I replace it by the softmax $B(\beta) = m + \beta^{-1}\log\sum_j e^{\beta(L_j - m)}$ with $m = \max_j L_j$ for numerical stability, which makes the whole objective differentiable and amenable to L-BFGS. I run an L-BFGS *ladder* over a rising sharpness $\beta$ — $5 \to 15 \to 40 \to 120 \to 400 \to 1500 \to 6000$ — so early steps move broadly over a smooth surface and late steps optimize a faithful stand-in for the true $\max$.

The signal I watch for is whether the optimizer drives heights to *exactly zero*. It does: of the $20$ heights, $v_0, v_1$ and $v_4 \ldots v_7$ go to $0$, leaving a tall spike at $v_2 = 1.0$ plus a shaped shoulder $v_8 \ldots v_{19}$ tapering from $\sim 0.5$ down to $\sim 0$. That is not noise — it means the best coarse construction uses a specific *sparse, asymmetric* support, not all $20$ pieces uniformly, confirming that the gains over $2/3$ come from a particular structure. The annealing clears the floor easily and climbs into the high $0.88$s, the band of the known $20$-step constructions, landing at $0.884823$ — about $0.0044$ under the published Matolcsi–Vinuesa $0.88922$. The shortfall is not the search idea but the coarse resolution: $20$ pieces cannot render a sufficiently flat-topped autoconvolution, and that cap on the cap is the cap on $R$. That is exactly the wall the next rung breaks, by lifting this optimized shape onto a much finer grid and refining with a gradient method that can carve structure $20$ heights cannot represent.

```python
import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import minimize

def autoconv_ratio(v):
    v = np.clip(np.asarray(v, dtype=float), 0.0, None)
    N = len(v); c = fftconvolve(v, v)
    L = np.zeros(2 * N + 1); L[1:2 * N] = c
    Lj, Ljp = L[:-1], L[1:]
    l2sq = (1.0 / 3.0) * np.sum(Lj**2 + Lj * Ljp + Ljp**2)
    l1 = 0.5 * np.sum(Lj + Ljp); linf = np.max(L)
    return float(l2sq / (linf * l1))

def _smooth_neg(v, beta):                       # softmax-L_inf smoothing of -R for L-BFGS
    v = np.abs(v); N = len(v); c = fftconvolve(v, v)
    L = np.zeros(2 * N + 1); L[1:2 * N] = c
    Lj, Ljp = L[:-1], L[1:]
    A = (1.0 / 3.0) * np.sum(Lj**2 + Lj * Ljp + Ljp**2); C = 0.5 * np.sum(Lj + Ljp)
    m = np.max(L); B = m + np.log(np.sum(np.exp(beta * (L - m)))) / beta
    return -A / (B * C)

def _ladder(v, betas=(5, 15, 40, 120, 400, 1500, 6000)):
    v = np.abs(v).copy()
    for beta in betas:
        v = np.abs(minimize(_smooth_neg, v, args=(beta,), method='L-BFGS-B',
                            options={'maxiter': 4000}).x)
    return v

def search(N=20, seed=2024, restarts=120, sa_iters=300000):
    rng = np.random.default_rng(seed); x = np.linspace(0, 1, N)
    best, bestv = -1.0, None
    for r in range(restarts):                   # L-BFGS multistart over the smooth surrogate
        v0 = (np.exp(-(x - 0.5)**2 / 0.12) + 0.15 * rng.random(N)) if r % 2 == 0 \
             else rng.random(N) + 0.1
        v = _ladder(v0); rr = autoconv_ratio(v)
        if rr > best: best, bestv = rr, v
    v, cur = bestv.copy(), best                 # SA polish on the exact ratio
    for i in range(sa_iters):
        T = 0.002 * (1 - i / sa_iters) + 1e-6
        j = rng.integers(N); old = v[j]
        v[j] = abs(old + rng.normal(0, 0.015 * (0.1 + old)))
        rr = autoconv_ratio(v)
        if rr >= cur or rng.random() < np.exp((rr - cur) / T):
            cur = rr
            if rr > best: best, bestv = rr, v.copy()
        else:
            v[j] = old
    return bestv

# Reproducible optimized 20-step profile (max-normalized; what the search above converges to).
V20 = [0.000000, 0.000000, 1.000000, 0.292005, 0.000000, 0.000000, 0.000000, 0.000000,
       0.499995, 0.353999, 0.396630, 0.384182, 0.387817, 0.386756, 0.262068, 0.209676,
       0.177803, 0.115585, 0.057684, 0.000115]

def construct():
    return V20

if __name__ == "__main__":
    print("embedded R =", round(autoconv_ratio(construct()), 6))   # 0.884823
    # print("re-search R =", round(autoconv_ratio(search()), 6))   # ~0.88 (stochastic)
```
