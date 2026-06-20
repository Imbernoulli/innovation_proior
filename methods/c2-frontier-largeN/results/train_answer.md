For a non-negative function $f$, the autoconvolution Hölder ratio
$$R(f) = \frac{\|f*f\|_2^2}{\|f*f\|_\infty \cdot \|f*f\|_1} \le 1$$
is never attained, and the constant $C_2 := \sup_f R(f)$ is approached only from below by explicit constructions. The natural construction class is the non-negative step function $f = \sum_n v_n \mathbf{1}_{[n,n+1)}$ with $v_n \ge 0$, which is translation- and dilation-invariant and whose autoconvolution is piecewise linear: writing $L_j = (v*v)_{j-1}$ for the node values with $L_0 = L_{2N} = 0$, the three norms have closed forms $\|f*f\|_\infty = \max_j L_j$, $\|f*f\|_1 = \tfrac12\sum_j (L_j + L_{j+1})$, and $\|f*f\|_2^2 = \tfrac13\sum_j (L_j^2 + L_j L_{j+1} + L_{j+1}^2)$. The flat profile gives only $0.6667$; the best small hand-built constructions reach $0.88922$ (Matolcsi–Vinuesa, 20 steps); and a hierarchical $\beta$-annealed gradient refinement climbs into the high $0.89$s by a few hundred pieces — my own $N=500$ rung settled at $0.894706$ with the gradient still visibly moving. The published step-function frontier sits just above this, at Boyer–Li's $575$-piece $0.901564$ and Jaech–Joseph's $539$-piece $\sim 0.9016$, and far above everything stands the absolute record $0.96102$, a $\sim 50000$-piece irregular function found by an evolutionary search with orders of magnitude more compute. Because the $500$-piece gradient had not stalled, the resolution was not exhausted, and the honest next move is to lift once more, into the thousands of pieces, and spend a long, sharply-annealed refinement to reach that step-function frontier.

The method I propose is the endpoint of that hierarchical ladder: lift the optimized $500$-piece profile by a factor of $4$ to $N=2000$ and then run a long $\beta$-annealed kicked-Adam ladder whose *budget and schedule* are tuned specifically for high resolution. The lift itself is free and ratio-preserving — repeating each height four times reproduces the same function and the same $R$ — but it lands on a degenerate plateau of flat blocks, so I immediately kick the vector with a small multiplicative perturbation to break the block symmetry and give the gradient traction. The optimizer is Adam ascent on a smooth surrogate of $R$: since $\|f*f\|_\infty = \max_j L_j$ is non-differentiable, I replace it with the softmax temperature stand-in $B = m + \beta^{-1}\log\sum_j e^{\beta(L_j - m)}$ (with $m = \max_j L_j$), which approaches the true max as $\beta \to \infty$, and ascend $Q = A/(B\,C)$ where $A = \|f*f\|_2^2$ and $C = \|f*f\|_1$. The gradient with respect to the heights is assembled by the chain rule — $dL = dA/(BC) - A\,dB/(B^2C) - A\,dC/(BC^2)$, with $dB = w/Z$ the softmax weights — and then pushed back through the autoconvolution by a single correlation $g = 2\,(dc \star v)$, all of it evaluated with FFTs so each step costs $O(N\log N)$.

Three things change at $2000$ pieces, and getting them right is the whole game. First, the run must be much longer. At $500$ pieces a few thousand Adam steps per pass sufficed; at $2000$ the optimum is a finer, more irregular shape with many more coordinates to coordinate, and the gradient keeps finding small improvements for tens of thousands of steps. So I budget a single grind of $40000$ iterations and let it run — affordable precisely because the FFT evaluator keeps each step at $O(N\log N)$, so $40000$ steps is a couple of minutes rather than hours. Second, $\beta$ must be pushed *far* sharper at the end. The softmax is a faithful proxy for the hard max only when $\beta$ is large relative to the spread of the node values, and at $2000$ pieces a tall spike makes those values span a wide range; a $\beta$ that was sharp enough at $500$ is too soft here, letting the surrogate's peak sit below the true peak so the optimizer chases a slightly wrong objective and the true $R$ lags. I therefore anneal $\beta$ up to several hundred times $N$ — to $\sim 400N$ in the grind and $\sim 800N$ in the polish — so the surrogate genuinely tracks the hard max and the optimum it finds is the optimum of the real ratio. Third, the kicks become periodic rather than one-off. A single kick at the lift unsticks the initial plateau, but over tens of thousands of steps the optimizer can settle into a shallow basin and stop improving; a small multiplicative kick every few thousand steps acts as a mild restart that keeps the long run exploring, and I shrink it as the run sharpens so the late phase is pure refinement. Throughout, I track the best *true* $R$ ever seen rather than the surrogate-best vector, because the surrogate and the true ratio diverge slightly and I want the genuinely best returned point.

Concretely the $2000$-piece endpoint runs as a short ladder of passes: a moderate-$\beta$, kicked reorganize pass ($12000$ iterations, lr $0.006$, kick $0.03$) to let the lifted shape settle into the finer grid; then the long grind in two stages ($10000$ then $40000$ iterations, lr $0.003 \to 0.002$, $\beta$ annealed to $\sim 200N$ then $\sim 400N$, with periodic kicks of $0.006$–$0.008$) that carves the irregular fine structure the published constructions rely on; then a low-learning-rate, sharpest-$\beta$ polish ($20000$ iterations, lr $0.0008$, $\beta$ to $\sim 800N$). The reorganize pass clears the $500$-piece value comfortably, landing near $0.899$, and the long sharp-$\beta$ grind pushes through into the low $0.90$s. The measured result is $R \approx 0.901804$, which matches Jaech–Joseph's $\sim 0.9016$ and exceeds Boyer–Li's $0.901564$ — the published step-function frontier — in about $130$ seconds. I am honest that this does not approach the absolute record: $0.96102$ is a $\sim 50000$-piece deliberately irregular function bought by an evolutionary search with vastly more compute, and the gap from the low $0.90$s up to $0.96$ is exactly the open part of the problem that a single bounded gradient run on a few thousand pieces does not close. So this is the step-function frontier careful gradient refinement on thousands of pieces can reach, with the record standing above as the still-open distance.

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
