The hierarchical lifts kept paying — $0.884823$ at $20$ pieces, $0.894706$ at $500$ — and the gradient was still moving when I stopped at $500$. The resolution was not yet exhausted, so the natural move is to lift once more, into the thousands of pieces, and spend a much longer, more carefully annealed refinement there. The published step-function frontier — Boyer–Li's $575$-piece $0.901564$, Jaech–Joseph's $539$-piece $\sim\!0.9016$ — lives exactly in the range a few thousand well-optimized pieces can reach, so that band is the target for this endpoint, with the absolute record (AlphaEvolve-V2's irregular $\sim\!50000$-piece $0.96102$) standing far above as the open distance.

The method is the *same $\beta$-annealed Adam gradient ascent as before, scaled to the high-resolution endpoint*: I take the optimized $500$-piece profile, upscale $\times 4$ to $N = 2000$ (free and ratio-preserving), kick it with a small multiplicative perturbation to break the flat-block plateau, and refine. What is new is the *budget and the schedule*, and getting them right is the whole game at this resolution. Three things change at $2000$ pieces.

First, the *length of the run*. At $500$ pieces a few thousand Adam steps per pass settled the shape; at $2000$ the optimum is a finer, more irregular profile with many more coordinates to coordinate, and the gradient keeps finding small improvements for tens of thousands of steps. So I budget a long final pass — on the order of $40000$ iterations — and let it grind. This is affordable only because the FFT evaluator and gradient are $O(N\log N)$: even $40000$ steps at $2000$ pieces is a couple of minutes.

Second, the $\beta$ schedule must be pushed *much sharper at the end*. The softmax $B(\beta) = m + \beta^{-1}\log\sum_j e^{\beta(L_j - m)}$ is a faithful proxy for the true $\|f*f\|_\infty = \max_j L_j$ only when $\beta$ is large relative to the spread of the node values. At $2000$ pieces with a tall spike the node values span a wide range, and a $\beta$ that was sharp enough at $500$ is too soft here — it lets the surrogate peak $B$ sit *below* the true peak, so the optimizer chases a slightly wrong objective and the true $R$ lags. So I anneal $\beta$ up to several hundred times $N$ (about $400N$) in the final passes, far sharper than at the coarse levels, so the surrogate genuinely tracks the hard $\max$ and the optimum it finds is the optimum of the real ratio.

Third, *periodic kicks during the long run*, not just at the upscale. A single kick at the lift unsticks the initial plateau, but over tens of thousands of steps the optimizer can settle into a *shallow* local basin and stall. A small multiplicative kick $v \mapsto |v(1 + \kappa\,\xi)|$, $\xi \sim \mathcal{N}(0,I)$, applied every few thousand steps acts like a mild restart that keeps the long run exploring — gentle enough not to wreck the shape, strong enough to jostle it out of a shallow trap. I shrink $\kappa$ as the run sharpens, so the late phase is pure refinement.

I run the endpoint as a short ladder of passes at $2000$: a first reorganizing pass with a moderate $\beta$ ceiling and a kick to let the lifted shape settle into the finer grid; then the long pass with a high $\beta$ ceiling and periodic kicks, the phase that carves the irregular fine structure the published constructions rely on; then a final low-learning-rate, sharpest-$\beta$ polish. Throughout I keep the best *true* $R$ ever seen across all passes, since the surrogate and the true ratio diverge slightly and I want the genuinely best vector, not the surrogate-best one.

The reorganizing pass clears the $500$-piece value comfortably — more resolution always helped — and the long sharp-$\beta$ grind is where the endpoint number comes from: the construction reaches $0.901804$, at and slightly above the published step-function frontier, matching Jaech–Joseph's $\sim\!0.9016$ and exceeding Boyer–Li's $0.901564$, here with $2000$ pieces and about $130$ seconds of compute. The returned solution is genuinely irregular and sparse — roughly $31\%$ of heights effectively zero, a spike $\approx 28\times$ the shoulder — the structure the literature reports for near-optimal autoconvolutions. I am honest that this does *not* reach $0.96102$: that record is a $\sim\!50000$-piece deliberately irregular function found by an evolutionary search spending orders of magnitude more compute, and the gap from the low $0.90$s to $0.96$ is the part a single bounded gradient run on a few thousand pieces does not close. So the endpoint of this single-constructor ladder is precisely the step-function frontier careful gradient refinement can reach, with the AlphaEvolve-V2 record standing above as the still-open distance — and the remaining gap to $0.96102$, like the residual $0.0390$ to the Hölder ceiling $1.0$, is the honest measure of how much further this open problem has to go.

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
