# Mean–variance portfolio selection and the efficient frontier

## The problem

Choose how to split a fixed sum across N securities whose future returns are uncertain. The
decision rule must be consistent with diversification (sensible investors spread their money) and
must be computable from estimable beliefs about the securities.

## The key idea

Maximizing expected return alone fails: portfolio expected return is a weighted average of the
securities' expected returns, so it is maximized by concentrating everything in the single
highest-mean security — it never recommends diversification. The fix is to make the *dispersion* of
portfolio return a second criterion. Use **variance**. The variance of a portfolio is a weighted
sum that includes all pairwise **covariances**; relative to a perfect-correlation mix, imperfectly
correlated securities reduce the variance of the mix, with the equal-variance two-asset check
giving V = (1/2)σ²(1 + ρ). Diversification, specifically into low-covariance securities, is built
into variance as a risk measure. With two opposed criteria, expected return (maximize) and variance
(minimize), and no assumed risk appetite, the rule is to keep only the **efficient** portfolios —
minimum variance for each level of expected return — and let the investor pick a point on that
**efficient frontier**.

## The model

For weights w = (X₁,…,X_N) with wᵀ1 = 1 (and Xᵢ ≥ 0 if short sales are barred), expected returns
μ = (μ₁,…,μ_N), and covariance matrix Σ (Σᵢⱼ = σᵢⱼ = ρᵢⱼ σᵢ σⱼ):

    portfolio expected return   E = wᵀμ = Σᵢ Xᵢ μᵢ
    portfolio variance          V = wᵀΣw = Σᵢ Σⱼ Xᵢ Xⱼ σᵢⱼ

A portfolio is **efficient** if no other gives at least as much E with less V (or as little V with
more E). The efficient set solves, for each target E,

    minimize  V = wᵀΣw    subject to   wᵀμ = E,   wᵀ1 = 1   (and Xᵢ ≥ 0 if no short sales).

**Geometry / closed form.** With only the two equality constraints, the Lagrangian
wᵀΣw − 2λ(wᵀ1 − 1) − 2γ(wᵀμ − E) gives Σw = λ1 + γμ, i.e. w = Σ⁻¹(λ1 + γμ). With

    A = 1ᵀΣ⁻¹1,   B = 1ᵀΣ⁻¹μ,   C = μᵀΣ⁻¹μ,   D = AC − B²,

the constraints give λ = (C − BE)/D, γ = (AE − B)/D, so the minimum-variance weights are an affine
function of the target E (this affine locus is the **critical line**), and the minimum variance is a
parabola in E:

    V(E) = (A E² − 2B E + C) / D.

The equality-constrained solution is a single parabola in the (E, V) plane, and the **global
minimum-variance portfolio** is w_mv = Σ⁻¹1 / (1ᵀΣ⁻¹1). With the no-short-sale constraint Xᵢ ≥ 0,
the attainable set is the simplex; the iso-mean lines are parallel, the iso-variance curves are
concentric ellipses in the non-degenerate case, and the efficient set becomes a connected series of
straight-line critical-line segments from the minimum available variance point to a
maximum-expected-return attainable portfolio — equivalently connected parabola segments in the
(E, V) plane. Plotting σ = √V against E gives the bullet-shaped efficient frontier.

## Worked frontier (three securities)

```python
import numpy as np

def portfolio_mean(weights, mu):
    return weights @ mu                       # E = w^T mu

def portfolio_variance(weights, Sigma):
    return weights @ Sigma @ weights          # V = w^T Sigma w (all covariances)

def select_portfolios(mu, Sigma, targets):
    # min-variance portfolio for each target E (Lagrange / critical line)
    one = np.ones(len(mu)); Si = np.linalg.inv(Sigma)
    A = one @ Si @ one
    B = one @ Si @ mu
    C = mu  @ Si @ mu
    D = A * C - B * B
    rows = []
    for E in targets:
        lam = (C - B * E) / D
        gam = (A * E - B) / D
        w = Si @ (lam * one + gam * mu)
        rows.append((E, portfolio_variance(w, Sigma), w))
    return rows, (A, B, C, D)

mu = np.array([0.06, 0.10, 0.14])
sd = np.array([0.10, 0.18, 0.28])
rho = np.array([[1.0, 0.30, 0.20],
                [0.30, 1.0, 0.50],
                [0.20, 0.50, 1.0]])
Sigma = np.outer(sd, sd) * rho

Si = np.linalg.inv(Sigma); one = np.ones(3)
w_mv = Si @ one / (one @ Si @ one)            # global minimum-variance portfolio
print("min-variance:  E=%.4f  sigma=%.4f"
      % (portfolio_mean(w_mv, mu), np.sqrt(portfolio_variance(w_mv, Sigma))))
# -> E=0.0666  sigma=0.0965   (vs sigma=0.28 for all-in the max-mean security)

w_eq = one / 3
print("equal mix sigma=%.4f  vs weighted-avg stand-alone sigma=%.4f"
      % (np.sqrt(portfolio_variance(w_eq, Sigma)), w_eq @ sd))
# -> 0.1465 vs 0.1867

targets = np.linspace(portfolio_mean(w_mv, mu), 0.14, 7)
for E, V, w in select_portfolios(mu, Sigma, targets)[0]:
    print("E=%.4f  sigma=%.4f  w=%s" % (E, np.sqrt(V), np.round(w, 3)))
```

The global minimum-variance portfolio (σ ≈ 0.0965) carries far less risk than concentrating in the
highest-mean security (σ = 0.28); an equal-weight mix has σ ≈ 0.147 against a weighted-average
stand-alone σ of ≈ 0.187, the gap being the covariance benefit of diversification; and sweeping the
target return traces the frontier V(E) = (A E² − 2B E + C)/D — the curve the investor chooses from.
