# The Gallager random-coding exponent

## Problem

Shannon's coding theorem says reliable communication is possible at every rate
$R < C$ and impossible above capacity, but it gives only a vanishing limit
$P_e \to 0$ — no rate of decay. The random-coding exponent quantifies *how fast*:
it produces a positive function $E_r(R)$, computable from the channel transition
probabilities about as easily as capacity, such that the best block codes achieve
$P_e \le e^{-N E_r(R)}$ for all $R < C$, on any discrete memoryless channel (and,
with an input-tilt, on power-constrained continuous channels).

## Key idea

Bound the maximum-likelihood error of a *random* codebook, but replace the loose
union bound by a one-parameter family: overbound the error indicator $\phi_m(y)$
(some competitor ties or beats the sent word) by the competitor likelihood-sum
raised to a free power $\rho \in (0,1]$, with the symmetric tilt $1/(1+\rho)$:
$$\phi_m(y) \le \left[\sum_{m'\ne m} \big(P(y\mid x_{m'})/P(y\mid x_m)\big)^{1/(1+\rho)}\right]^{\rho}.$$
Averaging over an i.i.d. ensemble factorizes (independent codewords), and Jensen's
inequality — valid because $\xi^\rho$ is concave for $0<\rho\le1$ — pulls the
ensemble average inside the bracket, $\overline{\xi^\rho}\le(\bar\xi)^\rho$. For a
memoryless channel everything collapses to a single-letter functional, and
optimizing $\rho$ and the input distribution $p$ gives the exponent. The union
bound is the frozen case $\rho=1$ (the Bhattacharyya/cutoff-rate line); letting
$\rho\to0$ gives gentler supporting lines whose slope at the origin is $I(p)$, which
keeps the exponent alive up to capacity.

## Main theorem (random-coding bound)

For a discrete memoryless channel with transition probabilities
$P_{jk}=\Pr(b_j\mid a_k)$, any block length $N$, $M=e^{NR}$ codewords, and any
input distribution $p=(p_1,\dots,p_K)$, there exists a code whose ML decoding-error
probability satisfies, for every $\rho\in[0,1]$,
$$P_e \le \exp\!\big[-N\,(E_0(\rho,p)-\rho R)\big],\qquad
E_0(\rho,p) = -\ln \sum_{j=1}^{J}\left(\sum_{k=1}^{K} p_k\,P_{jk}^{1/(1+\rho)}\right)^{1+\rho}.$$
Optimizing the free parameters defines the **random-coding exponent**
$$\boxed{\,E_r(R) = \max_{0\le\rho\le1}\ \max_{p}\ \big[E_0(\rho,p)-\rho R\big]\,},
\qquad P_e \le e^{-N E_r(R)}.$$

## Proof sketch of the bound

1. Exact ML error: $P_{e,m}=\sum_y P(y\mid x_m)\phi_m(y)$.
2. Tilt the indicator (eq. above): valid since the right side is $\ge0$ always and
   $\ge1$ when $\phi_m=1$ (some numerator term exceeds the denominator, and raising
   $\ge1$ to $\rho>0$ keeps it $\ge1$). Substituting and using the $1/(1+\rho)$
   symmetry,
   $P_{e,m}\le\sum_y P(y\mid x_m)^{1/(1+\rho)}\big[\sum_{m'\ne m}P(y\mid x_{m'})^{1/(1+\rho)}\big]^{\rho}$.
3. Average over the ensemble; independence splits the bar across the $m$-factor and
   the competitor bracket. Jensen with $0<\rho\le1$ pulls the bar inside:
   $\overline{P_{e,m}}\le(M-1)^{\rho}\sum_y\big[\sum_x P(x)P(y\mid x)^{1/(1+\rho)}\big]^{1+\rho}$.
4. Memoryless + i.i.d. letters factorize the sum into the $N$-th power of the
   single-letter quantity; with $M=e^{NR}$ this is $\exp[-N(E_0(\rho,p)-\rho R)]$.
   At least one code beats the average ⇒ existence.

## Why $E_r(R)>0$ for $R<C$

At $\rho=0$ the inner sum is the output marginal and $E_0(0,p)=-\ln 1=0$.
Differentiating $E_0=-\ln F(\rho)$ at $\rho=0$ (where $F(0)=1$) gives
$$\left.\frac{\partial E_0}{\partial\rho}\right|_{\rho=0}
= \sum_{j,k} p_k P_{jk}\ln\frac{P_{jk}}{\sum_{k'}p_{k'}P_{jk'}} = I(p),$$
the mutual information. Hence for small $\rho>0$,
$E_0(\rho,p)-\rho R \approx \rho\,(I(p)-R)>0$ whenever $R<I(p)$. Choosing the
capacity-achieving $p$ gives $I(p)=C$, so $E_r(R)>0$ for every $R<C$. Moreover
$E_0(\rho,p)$ is increasing and concave in $\rho$ (from monotonicity of weighted
power means and Hölder), so $E_r(R)$ is positive, continuous, and convex downward on
$0<R<C$, with $\partial E_r/\partial R=-\rho$ — the optimal tilt is the negative
slope.

## Structure of the curve

- **High rates** $R_{\text{crit}}<R<C$: interior $\rho^\star\in(0,1)$ solves
  $R=\partial E_0/\partial\rho$; parametrically $R=\partial E_0/\partial\rho$,
  $E_r=E_0-\rho\,\partial E_0/\partial\rho$.
- **Low rates** $R<R_{\text{crit}}=\partial E_0/\partial\rho|_{\rho=1}$: optimum at
  $\rho=1$, $E_r(R)=E_0(1,p)-R$, a slope-$-1$ line; its intercept is the cutoff
  rate $R_0$.

## Expurgated bound (low rates, $\rho\ge1$)

The averaged bound is loose at low $R$ because rare codes that assign two messages
nearly the same word dominate the average. Discard them: regard $P_{e,m}$ as random
over the ensemble, bound $\Pr(P_{e,m}\ge B)$ by tilting, and expurgate the worst
half of the codewords (rate cost $(\ln2)/N\to0$). The survivors satisfy
$P_e\le\exp[-N(E_x(\rho,p)-\rho R)]$ with
$$E_x(\rho,p) = -\rho\,\ln\sum_{k,i} p_k p_i\left[\sum_j\sqrt{P_{jk}P_{ji}}\right]^{1/\rho},\qquad \rho\ge1,$$
the pairwise-distance family continued past $\rho=1$ (where Jensen would have
reversed). At $\rho=1$, $E_x(1,p)=E_0(1,p)$, so the $\rho=1$ line is the same
$E_0(1,p)-R$ line used by the random-coding bound. The optimized expurgated envelope
improves on that line once its maximizing $\rho$ moves above $1$.

## Sphere-packing converse and exactness

Fano's lower bound is $P_e\ge\exp[-N(E_L(R)+o(1))]$ with
$E_L(R)=\sup_{0<\rho<\infty}\max_p[E_0(\rho,p)-\rho R]$ — the *same* $E_0$,
with $\rho$ ranging over $(0,\infty)$ instead of $[0,1]$. The two upper envelopes coincide
wherever the optimal $\rho\le1$, i.e. for $R_{\text{crit}}<R<C$; there $E_r(R)$ is
the **exact reliability function** of the channel.

## Input constraints / AWGN

For a constraint $\sum_n f(x_n)\le0$ (Gaussian: $f(x)=x^2-A$, average power $A$),
tilt the ensemble by $e^{rf(x)}$, $r\ge0$. The exponent becomes
$E_0(\rho,p,r)=-\ln\sum_j\big(\sum_k p_k P_{jk}^{1/(1+\rho)}e^{rf_k}\big)^{1+\rho}$
with a CLT-controlled sub-exponential coefficient; the stationary input for the
additive Gaussian channel is $p(x)=\mathcal N(0,A)$, reproducing Shannon's exact
Gaussian exponent in its known band.

## Reference computation

```python
import numpy as np

def E0(rho, p, P):
    # E0(rho,p) = -ln sum_j ( sum_k p_k P_jk^{1/(1+rho)} )^{1+rho}
    inner = (p[None, :] * P ** (1.0 / (1.0 + rho))).sum(axis=1)   # sum over inputs k
    return -np.log((inner ** (1.0 + rho)).sum())                 # sum over outputs j

def E_r(R, P, p_grid, rho_grid):
    # E_r(R) = max_{rho in [0,1], p} [ E0(rho,p) - rho*R ]
    return max(E0(rho, p, P) - rho * R
               for p in p_grid for rho in rho_grid)              # rho_grid subset of [0,1]

def E_x(rho, p, P):
    # Expurgated single-letter function, rho >= 1.
    pair = np.sqrt(P[:, :, None] * P[:, None, :]).sum(axis=0)     # pair[k, i]
    return -rho * np.log((p[:, None] * p[None, :] * pair ** (1.0 / rho)).sum())

def E_ex(R, P, p_grid, rho_grid):
    # Expurgated exponent: max over rho >= 1 of E_x(rho,p) - rho*R.
    return max(E_x(rho, p, P) - rho * R
               for p in p_grid for rho in rho_grid)

def E_L(R, P, p_grid, rho_grid):
    # Fano/sphere-packing exponent: same E0, rho over positive values.
    return max(E0(rho, p, P) - rho * R
               for p in p_grid for rho in rho_grid)

def error_probability_bound(N, R, P, p_grid, rho_grid):
    return np.exp(-N * E_r(R, P, p_grid, rho_grid))

def mutual_information(p, P):
    # I(p) = E0'(0): the slope at rho=0 that keeps E_r(R)>0 for R<C
    q = P @ p
    return sum(p[k] * P[j, k] * np.log(P[j, k] / q[j])
               for k in range(len(p)) for j in range(P.shape[0])
               if p[k] > 0 and P[j, k] > 0)

# Binary symmetric channel, crossover q: closed form via p=(1/2,1/2).
def E0_bsc(rho, q):
    return rho * np.log(2) - (1 + rho) * np.log(
        q ** (1 / (1 + rho)) + (1 - q) ** (1 / (1 + rho)))
```
