Training a large language model is essentially a one-shot decision: given a fixed compute budget C, you must commit to a model size N and a token count D before the run, because a full-scale training run is too expensive to repeat. The two choices are coupled by the dense-Transformer cost model C ≈ 6ND, so the problem is really choosing the best point on the hyperbola ND = C/6 to minimize the final pretraining loss L(N, D). The prevailing view, from Kaplan et al. (2020), held that compute should mostly buy parameters, with N growing as C^0.73 and D only as C^0.27. That conclusion was partly a measurement artifact: many intermediate token-count losses were read off long training runs whose cosine learning-rate schedule had not been matched to the actual stopping point, biasing losses upward at larger D and making extra tokens look less valuable than they are.

The method I propose is Chinchilla, also called compute-optimal scaling. Its core contribution is a corrected, multiply-checked estimate of the optimal allocation functions N_opt(C) and D_opt(C). The correction starts with a simple experimental discipline: each run's cosine decay cycle is matched to its own token horizon, so the learning rate reaches its floor exactly when training stops. With that fixed, the frontier is estimated through three independent procedures. The first takes many model sizes, trains each over a wide range of token horizons, smooths each loss-versus-FLOPs curve, and reads the lower envelope of all curves; this gives N_opt ∝ C^0.50 and D_opt ∝ C^0.50. The second fixes nine FLOP budgets and, for each, trains several model sizes whose token counts are set by the budget; loss versus log N is a valley, and the vertex gives the optimal N for that C, yielding exponents around 0.49 and 0.51. The third fits a parametric loss surface L_hat(N, D) = E + A/N^α + B/D^β, motivated by a risk decomposition into irreducible Bayes risk, a model-capacity gap, and a stochastic-optimization gap. The parameters are fit by minimizing Huber loss on log-predicted versus log-observed loss, using a numerically stable log-sum-exp formulation and L-BFGS from multiple initializations. Once fitted, minimizing L_hat subject to 6ND = C has the closed form N_opt(C) = G(C/6)^a and D_opt(C) = G^{-1}(C/6)^b, where a = β/(α+β), b = α/(α+β), and G = (αA/(βB))^{1/(α+β)}. With the fitted α ≈ 0.34 and β ≈ 0.28 this gives a ≈ 0.46 and b ≈ 0.54. All three methods agree: model size and training data should grow in roughly equal proportion with compute, not in the lopsided 0.73/0.27 ratio.

The practical implication is that contemporary large models were substantially over-parameterized for their token budgets. At a budget around 5.76 × 10^23 FLOPs, the estimates point to tens of billions of parameters trained on well over a trillion tokens, rather than hundreds of billions of parameters on roughly 300B tokens. Such a model is predicted to reach lower loss for the same training budget and is also cheaper to serve afterward, since inference cost scales with parameter count. A conservative single choice under extrapolation uncertainty is about 70B parameters and roughly 1.4T tokens.

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, huber


def log_loss_pred(theta, logN, logD):
    """Stable log of E + A/N**alpha + B/D**beta."""
    a, b, e, alpha, beta = theta
    terms = np.stack(
        [a - alpha * logN, b - beta * logD, np.full_like(logN, e)], axis=0
    )
    return logsumexp(terms, axis=0)


def parametric_loss(N, D, params):
    A, B, E, alpha, beta = params
    return E + A / N ** alpha + B / D ** beta


def fit_parametric(runs, delta=1e-3):
    """Fit L_hat(N,D) = E + A/N^alpha + B/D^beta robustly in log space."""
    runs = np.asarray(runs, dtype=float)
    N, D, L = runs[:, 0], runs[:, 1], runs[:, 2]
    logN, logD, logL = np.log(N), np.log(D), np.log(L)

    def objective(theta):
        r = log_loss_pred(theta, logN, logD) - logL
        return np.sum(huber(delta, r))

    best = None
    for alpha0 in [0.0, 0.5, 1.0, 1.5, 2.0]:
        for beta0 in [0.0, 0.5, 1.0, 1.5, 2.0]:
            for e0 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                for a0 in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]:
                    for b0 in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]:
                        res = minimize(
                            objective,
                            [a0, b0, e0, alpha0, beta0],
                            method="L-BFGS-B",
                        )
                        if best is None or res.fun < best.fun:
                            best = res
    a, b, e, alpha, beta = best.x
    return (np.exp(a), np.exp(b), np.exp(e), alpha, beta)


def optimal_allocation(C, params):
    """Closed-form optimum of the fitted parametric loss under 6*N*D = C."""
    A, B, E, alpha, beta = params
    G = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    a = beta / (alpha + beta)
    b = alpha / (alpha + beta)
    N_opt = G * (C / 6.0) ** a
    D_opt = (C / 6.0) ** b / G
    return N_opt, D_opt


def envelope_optimum(C, run_curves):
    """Read the lowest loss across interpolated training curves at compute C."""
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
    """Vertex of the parabolic loss valley in log N at one FLOP budget."""
    runs = np.asarray(runs_at_fixed_C, dtype=float)
    logN = np.log(runs[:, 0])
    loss = runs[:, 2]
    c2, c1, c0 = np.polyfit(logN, loss, 2)
    return np.exp(-c1 / (2 * c2))


def fit_power_law(Cs, values):
    """Fit value = coeff * C**exponent in log-log space."""
    slope, intercept = np.polyfit(np.log(Cs), np.log(values), 1)
    return np.exp(intercept), slope
```
