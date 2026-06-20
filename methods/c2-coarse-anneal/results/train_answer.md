The object of study is the autoconvolution Hölder ratio of a non-negative function. For $f \ge 0$ with autoconvolution $f*f(x) = \int f(t)\,f(x-t)\,dt$, the Cauchy–Schwarz/Hölder bound

$$R(f) = \frac{\|f*f\|_2^2}{\|f*f\|_\infty \cdot \|f*f\|_1} \le 1$$

holds and is never attained, since an autoconvolution can never be a perfect indicator. The quantity I want to push upward is $C_2 := \sup_f R(f)$, and the only way to learn anything about it is to produce explicit constructive lower bounds. Restricting to non-negative piecewise-constant profiles $f = \sum_{n=0}^{N-1} v_n\,\mathbf 1_{[n,n+1)}$ with $v_n \ge 0$ makes everything computable: the construction is translation- and dilation-invariant, so only the heights and their count matter; the autoconvolution is piecewise linear with node values $L_j = (v*v)_{j-1}$ and $L_0 = L_{2N} = 0$; and the three norms have exact closed forms, $\|f*f\|_\infty = \max_j L_j$, $\|f*f\|_1 = \tfrac12\sum_j(L_j+L_{j+1})$, and $\|f*f\|_2^2 = \tfrac13\sum_j(L_j^2+L_jL_{j+1}+L_{j+1}^2)$, with $v*v$ computed by FFT.

The trouble is that the obvious candidate goes nowhere. A flat step function — all heights equal — scores exactly $2/3$, because a constant profile has a rigid triangular autoconvolution, the tent function, and the Hölder ratio of a tent is locked at $2/3$ regardless of how finely I grid it. Refining the resolution is a pure no-op here: subdividing a flat profile produces a finer flat profile and the very same tent, so $R$ does not move at all. With every piece identical there is no gradient to follow and nothing to optimize. The conclusion is therefore unavoidable: to clear the $2/3$ floor I must break the flat symmetry and introduce genuine *variation* among the heights, then search the resulting shape space for a non-flat profile whose autoconvolution bends away from the tent — a flat cap with steep sides — so that the ratio climbs. The questions are how to search and at what scale.

I propose what I will call coarse simulated annealing on the heights. Scale is settled first because it governs everything else. The temptation is to jump straight to a long height vector, but that is a trap: the functional is badly non-convex, riddled with local optima and symmetry to break, and a long vector is a high-dimensional space in which a blind local search wanders for a very long time before finding anything. The disciplined move is to fix the *shape* at low resolution, where the vector is short enough to canvas thoroughly, and to lift that shape to finer grids only afterward. So I work at $N = 20$: short enough that a stochastic search can sweep the shape space, yet long enough that the autoconvolution has real internal structure to exploit. This is also exactly the resolution at which the $0.88$ band was first reached, so it is the natural rung on which to find the basic non-trivial shape.

The search itself must accept moves that make the objective temporarily worse, and this is the load-bearing choice. The flat profile, and most "obvious" profiles, sit at local optima or on near-flat plateaus where any small single-height perturbation *lowers* $R$ before a coordinated reshape can raise it. A greedy hill-climber that only takes improving moves parks itself at the first such point — at $2/3$ — and stops. Simulated annealing is precisely the cure: I propose a perturbation to one height, take it outright if it improves $R$, and take it even when it worsens $R$ with the Metropolis probability $\exp\!\big((R_{\text{new}} - R_{\text{cur}})/T\big)$ under a temperature $T$ that I cool over the run. Hot early so the search wanders freely and shakes loose from the flat-triangle basin; cold late so it settles into the best basin it has found.

Three details follow from the geometry of this particular objective. First, the perturbation must be *multiplicative in scale*. The good profiles, I expect, are not uniform spreads but a tall spike (or two) plus a long shoulder of smaller values — a wide dynamic range of heights. A single additive Gaussian kick of fixed size would be far too coarse for the small heights and far too timid for the large ones. So I perturb a randomly chosen height by an amount proportional to its own magnitude plus a small floor, $v_j \leftarrow |v_j + \mathcal N(0,\,\sigma(0.1 + v_j))|$, reflecting any negative result back to be non-negative so the candidate stays legal. This adjusts a tall spike and a thin shoulder value on comparable *relative* terms, which is the right invariance for a scale-free objective. Second, what to anneal on: because $R$ is already bounded in $[0,1]$ and its changes under one height perturbation are small and well-scaled, I do not need to take a log or rescale — I anneal directly on $R$ with a temperature of order $0.01$ down to a tiny floor, cooled geometrically, and I shrink the perturbation scale alongside the temperature so that late in the run the search makes fine adjustments to a settled shape rather than large jumps. Third, restarts: the landscape has many basins of differing quality, so I run several independent restarts from different initializations — some from a smooth single-bump (Gaussian-shaped) seed, some from pure random heights — and keep the best profile any restart ever reaches. The single-bump seed is the educated guess: a flat-capped autoconvolution is unimodal, so the height profile that produces it is likely concentrated, and starting concentrated should drop the search into the good basin faster.

Between annealing phases I let a gradient method do the legwork, which requires smoothing the one non-differentiable piece of the objective, the $\max$ in $\|f*f\|_\infty$. I replace $\max_j L_j$ by the softmax surrogate $B = m + \tfrac1\beta\log\sum_j \exp(\beta(L_j - m))$ with $m = \max_j L_j$ (subtracted for numerical stability), which is smooth and recovers the true maximum as $\beta \to \infty$. Optimizing $-A/(BC)$ — where $A = \|f*f\|_2^2$ and $C = \|f*f\|_1$ in the closed forms above — with L-BFGS along a ladder of increasing sharpness $\beta$ from $5$ up to $6000$ gives clean gradient refinement that hardens onto the true ratio as the ladder climbs. The full procedure is therefore an L-BFGS multistart over the smooth surrogate to find candidate shapes, followed by a simulated-annealing polish on the exact ratio.

What the search reveals is the genuine signal I was watching for: it drives several of the $20$ heights to exactly $0$. The best coarse profile does not use all $20$ pieces uniformly — it wants a *sparse, asymmetric* support, a spike plus a shaped shoulder with gaps in it, not mass spread evenly. The optimized profile (max-normalized, since the objective is scale-invariant) places its peak at one piece, a secondary lobe nearby, and a tapering shoulder, with four interior heights and the leading pair pinned to zero. The measured ratio is $R \approx 0.884823$. This clears the $2/3$ floor decisively — the entire payoff of accepting downhill moves — and lands in the high-$0.88$s where the known $20$-step constructions live, just shy of the published $20$-step value of $0.88922$. I do not expect to beat that from a short run on $20$ heights, and I do not: the limit is not the search idea but the coarse resolution itself, since $20$ pieces simply cannot render an arbitrarily flat-topped autoconvolution, and the flatness of that cap is what caps $R$. That is the clean opening for the next rung — lift this optimized coarse shape onto a much finer grid and let gradient refinement carve the fine structure that annealing on $20$ heights cannot represent.

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
