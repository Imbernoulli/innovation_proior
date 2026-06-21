The flat profile sat exactly at $R = 2.0$, and the way it stops there is sharp: with every piece identical the autoconvolution is locked to a triangle with one dominant central peak, there is no gradient to follow, and refining the grid does nothing because the score is invariant to piece count. The only way down is to break the symmetry — to introduce *variation* among the heights and let a search discover a non-flat profile whose self-convolution spreads its energy so the single tallest node is suppressed. The first decision is scale, and going straight to a long height vector is a trap: the functional is badly non-convex, full of local optima and symmetry to break, and a blind local search wanders a high-dimensional space for a very long time. So I find the *shape* at low resolution first, where the vector is short enough to canvas thoroughly, and defer the fine grid to a later rung. I work at $N \approx 50$ — short enough to explore, long enough that the autoconvolution already has real internal structure to exploit.

The method is **coarse simulated annealing on the heights followed by a $\beta$-annealed softmax-max Adam polish**. The reason to anneal rather than hill-climb is structural: the flat profile, and most "obvious" shapes, sit at local optima where any single small perturbation *raises* $R$ before a coordinated reshape can lower it. A greedy descender that only accepts improving moves parks at the first such point. Annealing crosses those ridges by accepting moves that make the ratio temporarily worse. I propose a perturbation to one randomly chosen height; if it lowers $R$ I take it, and if it raises $R$ I take it anyway with Metropolis probability $\exp(-\Delta R / T)$, where the temperature $T$ is cooled geometrically over the run — hot early so the search shakes loose from the flat-triangle basin, cold late so it settles into whatever better basin it has found.

Two design choices come straight from the geometry of this objective. First, the perturbation is *multiplicative in scale*: I perturb the chosen height by an amount proportional to its own magnitude, $a_j \leftarrow \max(0, a_j (1 + s \cdot \xi))$ with $\xi$ standard normal, clipping back to non-negative so the candidate stays legal. The good profiles span a wide dynamic range — likely heavier toward the ends, lighter in the middle, the arrangement that reduces central self-overlap — and a single additive Gaussian kick of fixed size would be far too coarse for the small heights and far too timid for the large ones. A multiplicative kick lets a tall value and a thin one move on comparable *relative* terms, which is the right invariance for a scale-free objective. The kick scale shrinks alongside $T$ ($s \approx 0.3\,(T/T_0) + 0.01$) so late iterations make fine adjustments to a settled shape. Second, I anneal directly on $R$ with no log or rescale, because $R$ is bounded above $1.28$ and its single-coordinate changes are small and well-scaled, so a temperature of order a few hundredths cooled geometrically keeps the Metropolis acceptance sane. Because the landscape is non-convex I run several independent restarts — the flat vector itself, a monotone ramp, a smooth single hump, and a Gaussian bump — and keep the best profile any restart ever reaches; the ramp and bump are educated guesses that, if the gains come from breaking left-right symmetry, should land in the good basin faster than starting flat.

Annealing one height at a time leaves the profile rough and tends to stop short of the local optimum its basin contains, so I add a gradient polish to slide the settled shape to the basin's bottom. The obstruction is that $R$ involves a hard $\max$ over the autoconvolution nodes, which is non-differentiable at the peak. I replace it with a smooth softmax surrogate of sharpness $\beta$: weights $w_k \propto \exp(\beta (b_k - \max_j b_j))$ form a temperature-$\beta$ soft-argmax over the nodes, and I anneal $\beta$ *upward* — starting soft, where the surrogate is smooth and its gradient points broadly toward better shapes, and ending sharp, where the surrogate faithfully tracks the true peak. The gradient factors cleanly through the self-convolution by the chain rule. Writing $M = \sum_k w_k b_k$ for the softmax peak, $\partial M / \partial a$ is $2$ times the correlation of the weights $w$ against the heights, computed as another FFT (`fftconvolve(w, a[::-1])` read at the appropriate offsets), and then

$$\frac{\partial R}{\partial a} = \frac{2N}{s^2}\,\frac{\partial M}{\partial a} - \frac{2R}{s}\,\mathbf{1},$$

where $s = \sum_n a_n$ and the second term is the derivative of the $1/(\sum a)^2$ mass normalization. Everything is $O(N \log N)$, so the polish is cheap. I run Adam on this gradient — its per-coordinate adaptive scaling suits the wide dynamic range of the heights — project to non-negative after each step, inject a small multiplicative kick periodically to escape shallow traps, and crucially track the best *true* $R$ ever seen, never the surrogate's value. I expect, and watch for, the optimizer driving several heights to zero and concentrating mass toward the ends: that is the genuine signature of a peak-suppressing construction, an asymmetric and partly sparse profile, not an even spread. The whole pipeline should clear the flat ceiling easily and drop well into the low $1.5$s, then stall — limited not by the search idea but by the coarse $N = 50$ resolution, which caps how fine an autoconvolution can be rendered and so how low the peak can be pushed. That stall is the opening for the next rung: lift this shape to a finer grid with a stronger optimizer that attacks the whole flat top of near-peak nodes at once.

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
