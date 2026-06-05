# Chinchilla (Compute-Optimal Scaling)

## Problem

Given a fixed training-compute budget $C$, how should you split it between model size $N$ and training tokens $D$ to minimize final pretraining loss? The cost model couples them: $C \approx 6ND$, so the choice is a point on the hyperbola $ND = C/6$. The deliverable is the pair of functions $N_{\text{opt}}(C)$, $D_{\text{opt}}(C)$.

## Key idea

Estimate the compute-optimal frontier three independent ways — all matching the cosine learning-rate cycle to each run's token horizon (failing to do so biases intermediate-horizon losses upward and tilts the answer toward big models on few tokens):

1. **Training-curve envelope (fix $N$, vary $D$).** Smooth/interpolate each run's loss-vs-FLOPs curve; at each compute level take the lower envelope; fit $N_{\text{opt}}\propto C^a$, $D_{\text{opt}}\propto C^b$. → $a=0.50$, $b=0.50$.
2. **IsoFLOP profiles (fix $C$, vary $N$).** At each fixed budget, loss vs $N$ is a valley; fit a parabola, take the vertex as the optimal $N$; power-law fit the vertices. → $a=0.49$, $b=0.51$.
3. **Parametric loss fit.** Model $\hat L(N,D)=E+\dfrac{A}{N^{\alpha}}+\dfrac{B}{D^{\beta}}$ (Bayes risk + an $N$-dependent approximation gap + a $D$-dependent stochastic gap). Fit $(A,B,E,\alpha,\beta)$ by minimizing the Huber loss between $\log\hat L$ and $\log L$ (computed via log-sum-exp), with L-BFGS from a grid of inits. Rounded: $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$; for projections use the unrounded fit, e.g. $\alpha=0.33917084$, $\beta=0.2849083$.

**Closed-form allocation.** Minimizing $\hat L$ subject to $6ND=C$ (substitute $D=C/(6N)$, set the derivative to zero):
$$N_{\text{opt}}(C)=G\Big(\tfrac{C}{6}\Big)^{a},\quad D_{\text{opt}}(C)=G^{-1}\Big(\tfrac{C}{6}\Big)^{b},\quad G=\Big(\tfrac{\alpha A}{\beta B}\Big)^{\frac{1}{\alpha+\beta}},\quad a=\tfrac{\beta}{\alpha+\beta},\ b=\tfrac{\alpha}{\alpha+\beta}.$$
Here $a+b=1$ identically. With the fit, $a=0.28/0.62=0.46$, $b=0.34/0.62=0.54$.

All three methods agree: $N$ and $D$ should be scaled in **roughly equal proportion** with compute ($a\approx b\approx 0.5$) — sharply against the prevailing $a\approx0.73$, $b\approx0.27$. Consequence: large models of the era are several times too big for their token budgets. At a budget of $\sim 5.76\times10^{23}$ FLOPs, the training-curve estimate is about 67B parameters and 1.5T tokens, the iso-FLOP estimate is similar, and the parametric fit is smaller (about 40B with the unrounded coefficients). A 70B, $\sim$1.4T-token allocation sits at the upper end of the predicted range and is much cheaper at inference than a 280B-parameter model.

## Code

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, huber


def log_loss_pred(theta, logN, logD):
    # log L_hat = LSE(a - alpha*logN, b - beta*logD, e) = log(E + A/N^alpha + B/D^beta)
    a, b, e, alpha, beta = theta
    terms = np.stack([a - alpha * logN, b - beta * logD, np.full_like(logN, e)], axis=0)
    return logsumexp(terms, axis=0)


def parametric_loss(N, D, params):
    A, B, E, alpha, beta = params
    return E + A / N ** alpha + B / D ** beta


def fit_parametric(runs, delta=1e-3):
    runs = np.asarray(runs, dtype=float)
    N, D, L = runs[:, 0], runs[:, 1], runs[:, 2]
    logN, logD, logL = np.log(N), np.log(D), np.log(L)

    def objective(theta):
        r = log_loss_pred(theta, logN, logD) - logL
        return np.sum(huber(delta, r))

    best = None
    for alpha0 in [0., 0.5, 1.0, 1.5, 2.0]:
        for beta0 in [0., 0.5, 1.0, 1.5, 2.0]:
            for e0 in [-1., -0.5, 0., 0.5, 1.]:
                for a0 in [0., 5., 10., 15., 20., 25.]:
                    for b0 in [0., 5., 10., 15., 20., 25.]:
                        res = minimize(objective, [a0, b0, e0, alpha0, beta0],
                                       method="L-BFGS-B")
                        if best is None or res.fun < best.fun:
                            best = res
    a, b, e, alpha, beta = best.x
    return (np.exp(a), np.exp(b), np.exp(e), alpha, beta)


def optimal_allocation(C, params):
    A, B, E, alpha, beta = params
    G = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    a = beta / (alpha + beta)
    b = alpha / (alpha + beta)
    N_opt = G * (C / 6.0) ** a
    D_opt = (C / 6.0) ** b / G          # == (C/6) / N_opt  ->  6*N_opt*D_opt == C
    return N_opt, D_opt


def envelope_optimum(C, run_curves):
    # run_curves: iterable of {"N": scalar, "flops": array, "loss": smoothed array}
    best = None
    logC = np.log(C)
    for curve in run_curves:
        flops = np.asarray(curve["flops"], dtype=float)
        if C < flops[0] or C > flops[-1]:
            continue
        N = float(curve["N"])
        loss = float(np.interp(logC, np.log(flops), curve["loss"]))
        D = C / (6.0 * N)
        if best is None or loss < best[2]:
            best = (N, D, loss)
    if best is None:
        raise ValueError("no curve covers the requested compute budget")
    return best


def isoflop_optimum(runs_at_fixed_C):
    runs_at_fixed_C = np.asarray(runs_at_fixed_C, dtype=float)
    logN = np.log(runs_at_fixed_C[:, 0])
    loss = runs_at_fixed_C[:, 2]
    c2, c1, c0 = np.polyfit(logN, loss, 2)
    return np.exp(-c1 / (2 * c2))


def fit_power_law(Cs, values):
    slope, intercept = np.polyfit(np.log(Cs), np.log(values), 1)
    return np.exp(intercept), slope     # (coeff, exponent)
```
