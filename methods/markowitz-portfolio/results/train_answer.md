An investor with a fixed sum and a list of securities has to decide how to split the money among them, and the future returns are uncertain. I want a rule that turns beliefs about the securities into a concrete split, that is computable for a realistic number of securities, and — this is the binding constraint — that is consistent with the one investment behavior everyone already agrees is sensible: diversification, spreading money across several securities rather than betting it all on one. The natural starting point is the discounted-cash-flow idea in its sharpest form, due to John Burr Williams: the value of a security is the present value of its future dividend stream. Lifted to a portfolio under uncertainty, this says: assign each security $i$ its anticipated discounted return $R_i$, choose weights $X_i \ge 0$ with $\sum_i X_i = 1$, and maximize the portfolio's expected return $R = \sum_i X_i R_i$. But $R$ is a weighted average of fixed numbers $R_i$ with nonnegative weights summing to one, and a weighted average can never exceed its largest entry, attaining it only when all the weight sits on that entry. So this rule sets $X_i = 1$ for the single security with the largest $R_i$ and zero everywhere else — it always sends the investor to one security and never prefers a diversified portfolio. That is descriptively and normatively wrong about the behavior we are most sure of, and no cleverness about discount rates can rescue it, because the defect is in the *form*: one number per security, maximized as a weighted sum, has no room for diversification. The reason becomes clear once I ask why people diversify at all. Not to raise expected return — concentration does that. They diversify to reduce the uncertainty of outcomes, the spread, and the expected return $R_i$ is only the *center* of security $i$'s distribution. By keeping the center and deleting the spread, the rule threw away exactly the quantity that generates the behavior I am trying to explain. The obvious patch — spread among the top-mean securities and lean on the law of large numbers to make the realized yield converge to the expected one — also fails, because that law needs near-independent draws, and security returns are heavily intercorrelated: when the market falls most stocks fall together, so averaging cannot drive the dispersion to zero, and the maximum-expected-return portfolio is in general *not* the minimum-dispersion one. There is a genuine trade-off rate between return and dispersion, so dispersion has to enter the objective itself, as a thing the investor will pay expected return to reduce. Von Neumann–Morgenstern expected-utility maximization is the right ideal in principle but is not operational: evaluating $E[U]$ for many securities needs the full joint return distribution, far beyond what can be estimated and computed.

I propose mean–variance portfolio selection, whose output is the efficient frontier. The investor cares about two opposed quantities: the expected return of the portfolio, which they want high, and the dispersion of its return, which they want low. For dispersion I use the variance. The decisive question is whether the variance of a *portfolio* rewards spreading, and the answer comes from writing it out. With $\mu_i = E(R_i)$ the portfolio's expected return is the linear form $E = \sum_i X_i \mu_i$, while the variance is the quadratic form
$$V = E\!\left[\Big(\sum_i X_i (R_i - \mu_i)\Big)^2\right] = \sum_i \sum_j X_i X_j\, \sigma_{ij}, \qquad \sigma_{ij} = E[(R_i-\mu_i)(R_j-\mu_j)] = \rho_{ij}\,\sigma_i \sigma_j.$$
The variance of a weighted sum is emphatically *not* the weighted sum of the variances: the diagonal terms $\sum_i X_i^2 \sigma_{ii}$ are the naive guess, but there is a whole second batch of off-diagonal covariance terms $\sum_{i\ne j} X_i X_j \sigma_{ij}$, each carrying the correlation $\rho_{ij}$. Those cross terms are the mechanism of diversification. Whenever two securities are imperfectly correlated, $\rho_{ij} < 1$, so $\sigma_{ij} = \rho_{ij}\sigma_i\sigma_j$ is smaller than $\sigma_i\sigma_j$, and the variance of the mix falls below what the individual variances alone would give. The cleanest check is half in each of two securities of equal variance $\sigma^2$: $V = \tfrac14\sigma^2 + \tfrac14\sigma^2 + 2\cdot\tfrac14\rho\sigma^2 = \tfrac12\sigma^2(1+\rho)$, which equals $\sigma^2$ only at $\rho=1$ (lockstep, no benefit), drops to $\tfrac12\sigma^2$ at $\rho=0$, and is smaller still for $\rho<0$. So variance is the right risk measure precisely because it is a weighted sum that drags in all the covariances — diversification is built into it automatically. This also sharpens the proverb into a prescription the proverb never could give: holding *many* securities is not the point; sixty railway stocks barely diversify because their returns move together and the cross terms stay large, whereas the same number of names spread across railroads, utilities, mining, and manufacturing cuts the variance far more, because firms with different economic drivers have lower covariances. The right kind of diversification is into securities with low covariances among themselves, and the covariance terms are what name the baskets.

With two opposed criteria and no assumed risk appetite, there is no single best portfolio: more $E$ must be paid for with more $V$, at a rate that depends on a taste I do not want to assume. What I can do, as pure logic before any taste enters, is discard every portfolio that is beaten on both counts. A portfolio is efficient if no other gives at least as much $E$ with less $V$, or as little $V$ with more $E$; every sensible investor will pick one of these, and the rest are dominated. So the rule is to compute the efficient set — minimum variance for each level of expected return — and hand the investor that frontier to choose a point on. The efficient set solves, for each target $E$, $\min V = w^{\mathsf T}\Sigma w$ subject to $w^{\mathsf T}\mu = E$ and $w^{\mathsf T}\mathbf 1 = 1$ (and $X_i \ge 0$ under no short sales). Keeping just the two equality constraints, the Lagrangian $w^{\mathsf T}\Sigma w - 2\lambda(w^{\mathsf T}\mathbf 1 - 1) - 2\gamma(w^{\mathsf T}\mu - E)$ has gradient condition $\Sigma w = \lambda\mathbf 1 + \gamma\mu$, so $w = \Sigma^{-1}(\lambda\mathbf 1 + \gamma\mu)$. Defining the four scalars
$$A = \mathbf 1^{\mathsf T}\Sigma^{-1}\mathbf 1,\quad B = \mathbf 1^{\mathsf T}\Sigma^{-1}\mu,\quad C = \mu^{\mathsf T}\Sigma^{-1}\mu,\quad D = AC - B^2,$$
the two constraints give $\lambda = (C - BE)/D$ and $\gamma = (AE - B)/D$, so the minimum-variance weights are an affine function of the target $E$ — that affine locus is the critical line. The achieved variance follows without further work, since $V = w^{\mathsf T}\Sigma w = w^{\mathsf T}(\lambda\mathbf 1 + \gamma\mu) = \lambda + \gamma E$, which collapses to the parabola
$$V(E) = \frac{A E^2 - 2 B E + C}{D},$$
and dropping the return constraint entirely leaves the global minimum-variance portfolio $w_{mv} = \Sigma^{-1}\mathbf 1 / (\mathbf 1^{\mathsf T}\Sigma^{-1}\mathbf 1)$. Geometrically, in the weight plane (using $X_3 = 1 - X_1 - X_2$ for three securities) the iso-mean sets are parallel straight lines, because $E$ is affine in the weights, and the iso-variance sets are concentric ellipses around $w_{mv}$, because $V$ is a positive quadratic. The minimum-variance portfolio for a given $E$ is where an iso-mean line is tangent to the smallest ellipse it can touch; as $E$ sweeps, those tangency points trace the straight critical line. Under no short sales the attainable set is the simplex, and the efficient set becomes a connected series of critical-line segments running from the minimum available variance point up to the maximum-attainable-expected-return portfolio, turning each time a weight hits zero and a constraint binds; equivalently it is connected parabola segments in the $(E,V)$ plane, and plotting $\sigma = \sqrt V$ against $E$ gives the familiar bullet-shaped frontier. The concentrated all-in-the-best-mean portfolio I began by rejecting reappears only as the single aggressive endpoint of that frontier, not as the whole answer.

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
