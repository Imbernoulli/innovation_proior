The coarse annealing gave me a $20$-piece profile at $0.884823$, just under the published $0.88922$, and the obstruction was exactly the resolution: $20$ pieces is too coarse a grid to render a sufficiently flat-topped autoconvolution, and the cap on how flat that cap can get is the cap on $R$. The optimized shape itself looks right — a tall spike plus a tapering shoulder, several heights pinned at zero — so I do not want to throw it away and search a long vector from scratch. I want to *keep this shape* and give it more resolution, so the autoconvolution can develop fine structure a coarse vector cannot represent.

The method is a *hierarchical lift refined by $\beta$-annealed Adam gradient ascent*, and the first piece is the lift, which is mechanical and, crucially, ratio-preserving. If I take the $20$ heights and replace each by $k$ copies of itself, the result is a $20k$-piece step function that is *the same function* — same graph, same autoconvolution, same $R$. Upscaling costs nothing and risks nothing; it just hands the optimizer a finer canvas. But the upscaled point is a degenerate plateau: every block of $k$ copies is flat, and a symmetric refinement that respects that flatness will not move. The new degrees of freedom — and the new gains — live in letting neighbouring pieces *within* a block differ.

To exploit them I need a real optimizer on the heights, and at $N$ in the hundreds a coordinate-wise annealing that perturbs one height at a time is hopelessly slow — tens of thousands of acceptance tests to make one coherent change across a shoulder. I want to move *all* heights together along a good direction, which means a gradient. The obstruction is the same non-differentiable $\|f*f\|_\infty = \max_j L_j$; I replace it by the softmax $B(\beta) = m + \beta^{-1}\log\sum_j e^{\beta(L_j-m)}$ and then *anneal* $\beta$ upward over each pass — soft early, where the surrogate is smooth and the gradient points broadly toward better shapes, sharp late, where the surrogate is a faithful stand-in for the true $\max$ and its optimum is the optimum of $R$. The gradient factors cleanly through the autoconvolution. Writing $A = \|f*f\|_2^2$, $C = \|f*f\|_1$, and the surrogate ratio $Q = A/(BC)$, the chain rule gives the derivative with respect to each node value,
$$\frac{\partial Q}{\partial L_j} = \frac{1}{BC}\frac{\partial A}{\partial L_j} - \frac{A}{B^2 C}\frac{\partial B}{\partial L_j} - \frac{A}{BC^2}\frac{\partial C}{\partial L_j},$$
with $\partial B/\partial L_j = e^{\beta(L_j-m)}/Z$ the softmax weights, and the piecewise-linear $A$, $C$ contributing the simple local terms $\partial A$ and $\partial C$ to each consecutive node pair. Since $L_j = c_{j-1}$ with $c = v*v$, each $L_j$ depends on the heights through the self-convolution, so the derivative with respect to the heights is itself a convolution of the node-gradient with the (reversed) height vector — computable with FFTs in $O(N\log N)$. Even a few thousand gradient steps at $N = 500$ is seconds.

The optimizer on top of this gradient is *Adam*, chosen for a specific reason. The heights span a wide dynamic range — a spike roughly $28\times$ the smallest shoulder values in the eventual solution — and a plain fixed-step gradient ascent either crawls on the large coordinates or blows up the small ones. Adam's per-coordinate adaptive scaling, normalizing each height's step by its own recent gradient magnitude (with the standard bias-corrected first and second moments, $b_1 = 0.9$, $b_2 = 0.999$), advances the spike and the thin shoulder on comparable terms. After every step I clip to non-negative to stay legal, and I track the best *true* $R$ ever seen — not the surrogate — because the two differ slightly and I want to return the genuinely best vector.

One more ingredient is load-bearing: a small multiplicative perturbation $v \mapsto |v(1 + \varepsilon\,\xi)|$, $\xi \sim \mathcal{N}(0,I)$, applied *right after each upscale*. The upscaled point has zero gradient within a block along the symmetric directions, so Adam started exactly there can sit still. A tiny kick breaks the block symmetry and gives the gradient something to grab; I keep it small ($\varepsilon \sim 0.03$–$0.06$) so it unsticks the plateau without destroying the good coarse shape.

So the rung is a ladder of lifts: take the optimized $20$-piece profile, upscale $\times 5$ to $100$ and refine with two $\beta$-annealed Adam passes, upscale $\times 5$ again to $500$ and refine. Each lift is free; each refinement adds fine structure the previous resolution could not hold. The first lift to $100$ already clears the $20$-piece value, because $100$ pieces flatten the cap further; the second to $500$ adds more, and the construction lands at $0.894706$ — into the band of the published $50$–$575$-step results, just under AlphaEvolve's $50$-step $0.89628$. I do not reach the $\sim\!0.9016$ of Boyer–Li / Jaech–Joseph from a few thousand Adam steps per level: those spent vastly more compute (Boyer–Li on the order of $10^6$ gradient trajectories), and the last fraction of a percent is dominated by the irregular fine structure of the true optimum. The gradient is still paying at $500$ pieces, which is the opening for the endpoint — lift once more, to thousands of pieces, and spend a long, kicked, sharpening Adam schedule pushing toward that frontier.

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
