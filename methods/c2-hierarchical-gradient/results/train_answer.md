I want to push the autoconvolution Hölder ratio as high as I can. For a non-negative $f$ the quantity
$$R(f) = \frac{\|f*f\|_2^2}{\|f*f\|_\infty \cdot \|f*f\|_1} \le 1$$
is never attained, and $C_2 := \sup_f R(f)$ (Barnard–Steinerberger) is only ever approached by explicit constructions; the flat floor sits at $0.6667$ and the published step-function records climb from Matolcsi–Vinuesa's 20-step $0.88922$ through AlphaEvolve's 50-step $0.89628$ to the Boyer–Li / Jaech–Joseph $\sim 575$-step values near $0.9016$. I restrict to non-negative step functions $f = \sum_n v_n\,\mathbf 1_{[n,n+1)}$, which is legitimate because $R$ is invariant under translation and dilation, so the problem is finite-dimensional in the height vector $v \ge 0$. The autoconvolution is then piecewise linear with node values $L_j = (v*v)_{j-1}$ and $L_0 = L_{2N} = 0$, and the three norms have closed forms: $\|f*f\|_\infty = \max_j L_j$, $\|f*f\|_1 = \tfrac12\sum_j (L_j + L_{j+1})$, and $\|f*f\|_2^2 = \tfrac13\sum_j (L_j^2 + L_j L_{j+1} + L_{j+1}^2)$, with the self-convolution computed by FFT in $O(N\log N)$. When I run a coarse $\sim 20$-piece annealing search this way I get a profile around $0.8848$ — a tall spike with a tapering shoulder and several heights pinned at zero — just under the $0.88922$ benchmark. The shape is right but it is stuck, and the reason is resolution: $20$ pieces simply cannot render an autoconvolution whose cap is flat enough, and the flatness of that cap is what caps $R$. Searching a long vector from scratch throws away the good coarse shape, and coordinate-wise annealing on hundreds of heights is hopelessly slow, so I need a way to keep the shape and give it more resolution.

I propose a **hierarchical lift with $\beta$-annealed Adam gradient ascent**. The lift is the key observation: if I replace each of the $20$ heights by $k$ identical copies, I get a $20k$-piece step function that is *the same function* — same graph, same autoconvolution, same $R$. Upscaling is therefore exactly ratio-preserving; it costs nothing, risks nothing, and simply hands the optimizer a finer canvas on which the autoconvolution can grow fine structure a coarse vector cannot hold. The subtlety is that the upscaled point is degenerate: every block of $k$ copies is flat, so it is a fixed point of any symmetric refinement and the gradient within a block vanishes. To unstick it I apply a small multiplicative kick, $v \leftarrow |v\,(1 + \varepsilon\,\xi)|$ with $\xi$ standard normal and $\varepsilon \approx 0.03$–$0.06$, which breaks the block symmetry and gives the gradient something to grab while keeping the good coarse shape intact. Then I refine with a real gradient method on all heights at once. The obstruction to differentiating $R$ is the $\max$ in the denominator, $\|f*f\|_\infty = \max_j L_j$, which is non-smooth at the peak. I replace it by the smooth softmax surrogate $B = m + \tfrac1\beta \log \sum_j e^{\beta(L_j - m)}$ with $m = \max_j L_j$, and *anneal* the sharpness $\beta$ upward over the run — soft early, where the surrogate is smooth and the gradient points broadly toward better shapes, sharp late, where $B \to \max_j L_j$ so that the optimum of the surrogate is the optimum of the true $R$. The gradient of the surrogate ratio $Q = A/(BC)$, with $A = \|f*f\|_2^2$ and $C = \|f*f\|_1$, factors cleanly: I differentiate $Q$ with respect to each node value $L_j$ via $\partial Q/\partial L = \mathrm dA/(BC) - A\,\mathrm dB/(B^2 C) - A\,\mathrm dC/(BC^2)$ where $\mathrm dB = e^{\beta(L-m)}/Z$, and then each $L_j$ depends on the heights through the self-convolution, so the chain rule back to the heights is itself a convolution of the node-gradient with the (reversed) height vector — everything still $O(N\log N)$ with FFTs, so a few thousand steps at $N = 500$ take seconds.

The optimizer on top of this gradient is **Adam**, and the choice is deliberate. The heights span a wide dynamic range — the spike is about $28\times$ the smallest shoulder values — and a plain fixed-step gradient ascent either crawls on the large coordinates or blows up the small ones. Adam's per-coordinate adaptive scaling, $v \leftarrow |v + \alpha\,\hat m_t / (\sqrt{\hat v_t} + \epsilon)|$ with the usual bias-corrected first and second moments $\hat m_t, \hat v_t$ at $\beta_1 = 0.9$, $\beta_2 = 0.999$, normalizes each height's step by its own recent gradient magnitude, so the spike and the thin shoulder advance on comparable terms. After every step I clip to non-negative to stay legal, and crucially I track the best *true* $R$ ever seen rather than the surrogate $Q$, because the two differ slightly and I want to return the genuinely best vector. The whole rung is then a ladder of free lifts and cheap refinements: take the optimized $20$-piece seed, upscale $\times 5$ to $100$ and refine with two $\beta$-annealed Adam passes, upscale $\times 5$ again to $500$ and refine. Each lift flattens the autoconvolution's cap a little more than the previous resolution could, and the measured climb bears this out — the run reaches $R \approx 0.894706$, into the AlphaEvolve $50$-step band. I do not expect to reach the $0.9016$ frontier from a few thousand Adam steps per level: those constructions spent vastly more compute (Boyer–Li on the order of $10^6$ gradient trajectories) and the last fraction of a percent is notoriously expensive, so the honest ceiling of this rung is the high $0.89$s, with the final push left to a longer, more carefully annealed run at higher resolution.

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
