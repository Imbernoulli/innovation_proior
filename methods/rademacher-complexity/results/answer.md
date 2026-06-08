# Rademacher complexity — data-dependent generalization bounds

## Problem

In statistical learning the test error of a predictor chosen from a class $\mathcal F$ after seeing the data is governed by the **uniform** gap $\sup_{f\in\mathcal F}(E_P f-\hat E_S f)$ between true and empirical means. Classical penalties bounding this gap (VC dimension, covering numbers, fat-shattering) are *distribution-free, worst-case* combinatorial numbers: identical for every distribution and every realized sample, hence loose for the problem at hand and unreliable for model selection. We want a capacity measure that is computable from the single training sample, adapts to the unknown distribution through it, provably controls the gap, and is never much worse than the worst-case penalties.

## Key idea

Measure the capacity of a function class by **how well it can correlate with random $\pm1$ noise on the actual sample**. Draw i.i.d. Rademacher signs $\sigma_i\in\{\pm1\}$ and ask how large the class can make the noise-correlation:

$$\hat R_S(\mathcal F)=E_\sigma\Bigl[\sup_{f\in\mathcal F}\frac1m\sum_{i=1}^m\sigma_i f(z_i)\Bigm| z_1,\dots,z_m\Bigr]\quad(\text{empirical}),\qquad R_m(\mathcal F)=E_S\,\hat R_S(\mathcal F).$$

Since $E\sigma_i=0$, any correlation a function achieves is pure overfitting to noise, so $\hat R_S$ quantifies the class's capacity to fit anything. It is **data-dependent** — computed on the realized points, with no reference to $P$ — and it bounds the uniform generalization gap directly.

## Main results

**Symmetrization.** For a class $\mathcal G$ into $[0,1]$,
$$E_S\sup_{g\in\mathcal G}(E_P g-\hat E_S g)\le 2R_m(\mathcal G).$$
The proof introduces a ghost sample $S'$ ($E_P g=E_{S'}\hat E_{S'}g$), pulls $E_{S'}$ out of the supremum by Jensen, and exploits the exchangeability of paired real/ghost points — swapping $z_i\leftrightarrow z_i'$ is the same as flipping a sign $\sigma_i$, leaving the expectation invariant — which produces the Rademacher signs; sub-additivity of $\sup$ over the two halves gives the factor $2$.

**Generalization bound (McDiarmid).** $\Phi(S)=\sup_g(E_P g-\hat E_S g)$ changes by at most $1/m$ when one sample point changes, so McDiarmid's bounded-differences inequality with $\sum_i c_i^2=1/m$ gives, with probability $\ge1-\delta$, for all $g\in\mathcal G$:
$$E_P g\;\le\;\hat E_S g\;+\;2R_m(\mathcal G)\;+\;\sqrt{\frac{\log(1/\delta)}{2m}}.$$
$\hat R_S$ itself has the same $1/m$ bounded-difference property, and the one-sided McDiarmid event $R_m\le\hat R_S+\sqrt{\log(2/\delta)/2m}$ gives the **fully data-dependent** bound
$$E_P g\;\le\;\hat E_S g\;+\;2\hat R_S(\mathcal G)\;+\;3\sqrt{\frac{\log(2/\delta)}{2m}}.$$

**Binary classification reduction.** For the zero-one loss class $\mathcal G=\{(x,y)\mapsto\mathbf 1[h(x)\neq y]\}$, using $\mathbf 1[h(x)\neq y]=(1-yh(x))/2$ and that $-\sigma_i y_i\sim\sigma_i$, one gets $\hat R_S(\mathcal G)=\tfrac12\hat R_{S_X}(\mathcal H)$, hence with probability $\ge1-\delta$, for all $h\in\mathcal H$:
$$P(Y\neq h(X))\;\le\;\hat P_m(Y\neq h(X))\;+\;R_m(\mathcal H)\;+\;\sqrt{\frac{\log(1/\delta)}{2m}}.$$
The data-dependent version is $P(Y\neq h(X))\le\hat P_m(Y\neq h(X))+\hat R_{S_X}(\mathcal H)+3\sqrt{\log(2/\delta)/(2m)}$.

**Never worse than VC (Massart's lemma).** For finite $A\subseteq\mathbb R^m$ with $r=\max_{u\in A}\|u\|_2$, $E_\sigma\sup_{u\in A}\frac1m\sum_i\sigma_i u_i\le r\sqrt{2\log|A|}/m$ (Jensen + Hoeffding's lemma + optimizing $t=\sqrt{2\log|A|}/r$). Applied to $\mathcal H_{|S}$ ($|A|\le\Pi_{\mathcal H}(m)$, $r=\sqrt m$): $R_m(\mathcal H)\le\sqrt{2\log\Pi_{\mathcal H}(m)/m}$, and via Sauer $\le\sqrt{2d\log(em/d)/m}$ — the VC rate as a corollary. Because the chain runs through the realized sign-patterns / empirical entropy, the Rademacher bound can be far tighter.

**Contraction for Lipschitz losses (Ledoux–Talagrand).** In the absolute $2/m$ structural norm $R_m^{\mathrm{str}}(\mathcal F)=E\,E_\sigma\sup_{f\in\mathcal F}|(2/m)\sum_i\sigma_i f(X_i)|$, if $\phi$ is $L$-Lipschitz with $\phi(0)=0$, then $R_m^{\mathrm{str}}(\phi\circ\mathcal F)\le 2L\,R_m^{\mathrm{str}}(\mathcal F)$. Hence for a margin cost $\phi$ ($L$-Lipschitz, dominating the step), with probability $\ge1-\delta$:
$$P(Yf(X)\le0)\;\le\;\hat E_m\phi(Yf(X))\;+\;2L\,R_m^{\mathrm{str}}(\mathcal F)\;+\;\sqrt{\frac{\log(2/\delta)}{2m}}.$$

**Structural calculus.** The same norm obeys: monotonicity ($\mathcal F\subseteq\mathcal H\Rightarrow R_m^{\mathrm{str}}(\mathcal F)\le R_m^{\mathrm{str}}(\mathcal H)$); $R_m^{\mathrm{str}}(c\mathcal F)=|c|R_m^{\mathrm{str}}(\mathcal F)$; $R_m^{\mathrm{str}}(\mathrm{conv}\,\mathcal F)=R_m^{\mathrm{str}}(\mathcal F)$ (so voting/boosting reduces to the base class); $R_m^{\mathrm{str}}(\sum_j\mathcal F_j)\le\sum_j R_m^{\mathrm{str}}(\mathcal F_j)$; $R_m^{\mathrm{str}}(\mathcal F+h)\le R_m^{\mathrm{str}}(\mathcal F)+\|h\|_\infty/\sqrt m$. These plus contraction bound decision trees, neural nets, and kernel machines via their pieces; e.g. for $\{x\mapsto\langle w,\Phi(x)\rangle:\|w\|\le B\}$, the one-sided $1/m$ empirical complexity is at most $\frac{B}{m}(\sum_i k(X_i,X_i))^{1/2}$, while the absolute $2/m$ structural version has $\frac{2B}{m}(\sum_i k(X_i,X_i))^{1/2}$.

**Localization for fast rates.** A global $R_m$ gives only $1/\sqrt m$. Restricting the noise-correlation to a variance ball $\mathcal F\cap B(r)$, $B(r)=\{f:Pf^2\le r\}$, and iterating $r_{k+1}=$ (local Rademacher norm)$(r_k)$ + variance-sensitive slack (Talagrand/Massart concentration) drives the bound to the fixed point $\delta$ of $\delta=m^{-1/2}\psi(\sqrt\delta)$ in $\approx\log\log(1/\varepsilon)$ steps; for a VC class this is $O(V(\mathcal H)\log m/m)$, the near-minimax fast rate in the realizable case.

## Final form

Let $\mathcal F$ map into $[0,1]$, $S=(z_1,\dots,z_m)$ be i.i.d. from $P$. With probability at least $1-\delta$, simultaneously for all $f\in\mathcal F$,

$$\boxed{\;E_P f\;\le\;\frac1m\sum_{i=1}^m f(z_i)\;+\;2\,\hat R_S(\mathcal F)\;+\;3\sqrt{\frac{\log(2/\delta)}{2m}}\;,\qquad \hat R_S(\mathcal F)=E_\sigma\sup_{f\in\mathcal F}\frac1m\sum_{i=1}^m\sigma_i f(z_i).}$$

## Computing the penalty

$\hat R_S$ is an expectation over Rademacher draws of a maximization over the class on the fixed sample — directly estimable by Monte Carlo:

```python
import numpy as np

def empirical_rademacher(corr_oracle, S, n_draws=1000, rng=None):
    """R_hat_S(F) = E_sigma[ sup_{f in F} (1/m) sum_i sigma_i f(z_i) ].
    corr_oracle(sigma, S) returns sup_{f in F} (1/m) sum_i sigma_i f(z_i)
    for a given sign vector sigma (an ERM-style maximization on this sample)."""
    if rng is None:
        rng = np.random.default_rng()
    m = len(S)
    vals = [corr_oracle(rng.choice([-1.0, 1.0], size=m), S) for _ in range(n_draws)]
    return float(np.mean(vals))

def generalization_bound(emp_risk, corr_oracle, S, delta, n_draws=1000):
    m = len(S)
    R_hat = empirical_rademacher(corr_oracle, S, n_draws)
    conf  = 3.0 * np.sqrt(np.log(2.0 / delta) / (2.0 * m))
    return emp_risk + 2.0 * R_hat + conf

# Linear class {x -> <w,x> : ||w||_2 <= B}:
#   sup_w (1/m) sum_i sigma_i <w,x_i> = (B/m) || sum_i sigma_i x_i ||_2
def linear_l2_corr_oracle(B):
    def oracle(sigma, X):                 # X: (m, d)
        return (B / X.shape[0]) * np.linalg.norm(sigma @ X)
    return oracle
```

For an $\ell_2$-ball linear class this Monte-Carlo estimate is bounded by $\hat R_S(\mathcal F)\le \tfrac{B}{m}(\sum_i\|x_i\|_2^2)^{1/2}$ (Cauchy–Schwarz + Jensen), and by $BR/\sqrt m$ when $\|x_i\|_2\le R$; for $\ell_1$/simplex constraints with coordinate-bounded features, Massart's lemma gives $\sqrt{\log d}$ dependence rather than the $\sqrt d$ dependence of a coordinatewise Euclidean bound.
