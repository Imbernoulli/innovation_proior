# Rate–Distortion Theory

## Problem

Lossless source coding gives one number — the entropy `H(X)` — for the bits per symbol needed to reproduce a discrete source *exactly*. But exact reproduction is impossible for a continuous source (it needs infinite capacity) and usually unwanted for a discrete one. The real question is: to reproduce a source within a stated average distortion `D`, what is the fundamental minimum rate? Rate–distortion theory answers it and supplies the lossy counterpart of the source-coding theorem.

## Key idea

Measure fidelity by a single-letter distortion `d(x, x̂) ≥ 0`, averaged over a block, `d(x^n,x̂^n) = (1/n) Σ_i d(x_i,x̂_i)`. A code is free to choose the joint law of source and reproduction, i.e. a **test channel** `q(x̂|x)`, subject only to the budget `E[d] ≤ D`. The bits a reproduction must carry *about the source* are measured by mutual information `I(X;X̂)`. The fundamental rate is the cheapest test channel meeting the budget:

```
R(D) = min_{ q(x̂|x) : E[d(X,X̂)] ≤ D }  I(X; X̂).
```

This is the dual of channel capacity `C = max_{p(x)} I(X;Y)`: capacity **maximizes** information over inputs for a fixed channel (concave curve); the rate–distortion function **minimizes** information over channels for a fixed source (convex curve).

## The rate–distortion theorem

For an i.i.d. source `X ~ P` with distortion `d`, `R(D)` is exactly the least achievable rate at average distortion `D`.

**Properties.** `R(D)` is non-increasing and convex in `D`, with `R(0) = H(X)` when zero distortion means exact reproduction for a discrete source, and `R(D) = 0` for `D ≥ d_max = min_{x̂} E[d(X,x̂)]`. Convexity: distortion is linear in the channel and `I` is convex in the channel, so mixing channels gives `R(λD_1+(1−λ)D_2) ≤ λR(D_1)+(1−λ)R(D_2)`.

**Converse (no code beats `R(D)`).** Any `(2^{nR}, n)` code with `E d(X^n,X̂^n) ≤ D` has `R ≥ R(D)`:

```
nR ≥ H(X̂^n) ≥ H(X̂^n) − H(X̂^n|X^n) = I(X^n; X̂^n)
   = Σ_i H(X_i) − Σ_i H(X_i | X̂^n, X_{i−1},…,X_1)      (X_i i.i.d.; chain rule)
   ≥ Σ_i [H(X_i) − H(X_i | X̂_i)]                        (conditioning reduces entropy)
   = Σ_i I(X_i; X̂_i) ≥ Σ_i R(D_i)                       (def. of R as min MI at distortion D_i)
   ≥ n R( (1/n) Σ_i D_i ) = n R(D).                     (convexity + Jensen)
```

**Achievability (codes approach `R(D)`).** Fix the optimal test channel `q*` achieving `R(D)` and its output marginal `Q(x̂)=Σ_x P(x)q*(x̂|x)`. For any `R > R(D)`, draw `2^{nR}` reproduction words i.i.d. `~ Q`; encode `x^n` by any codeword that is **distortion-typical** with it (jointly typical and empirical distortion within `ε` of `D`). The probability that no codeword distortion-covers a typical source word is bounded, via `p(x̂^n) ≥ p(x̂^n|x^n) 2^{−n(I(X;X̂)+3ε)}` and `(1−ab)^M ≤ 1−a+e^{−bM}`, by

```
exp( − 2^{n(R − I(X;X̂) − 3ε)} ) → 0     whenever R > I(X;X̂)+3ε,
```

and with the optimal test channel `I(X;X̂)=R(D)`, any `R>R(D)` leaves room to choose `ε` small enough. The expected distortion over random codebooks is `≤ D + δ`; hence some codebook achieves it. Channel coding is sphere **packing** (`R<C`); rate–distortion coding is sphere **covering** (`R>R(D)`).

## Worked examples

**Gaussian source, squared error.** `X ~ N(0,σ²)`, `d(x,x̂)=(x−x̂)²`:

```
R(D) = ½ log₂(σ²/D),  0 < D < σ²;    R(0)=+∞;    R(D) = 0,  D ≥ σ².
```

Lower bound: `I = h(X) − h(X−X̂|X̂) ≥ ½log₂(2πeσ²) − ½log₂(2πeD) = ½log₂(σ²/D)` (conditioning reduces entropy; Gaussian maximizes entropy for fixed variance). Achieved by the **backward** test channel `X = X̂ + Z` with `X̂ ~ N(0, σ²−D)` independent of `Z ~ N(0, D)`, giving `E(X−X̂)²=D` and equality for `0<D≤σ²`. Equivalently `D(R)=σ² 2^{−2R}` — each bit cuts distortion 4× (6.02 dB/bit). For comparison, 1-bit scalar quantization of `N(0,σ²)` yields `≈0.363σ²`, worse than the `D=σ²/4` permitted at `R=1`; long-block joint coding closes the gap.

**Binary source, Hamming distortion.** `X ~ Bernoulli(p)`, `p ≤ ½`:

```
R(D) = H(p) − H(D),  0 ≤ D ≤ min(p,1−p);    R(D) = 0,  D ≥ min(p,1−p),
```

with `H(·)` the binary entropy. Lower bound `I = H(p) − H(X⊕X̂|X̂) ≥ H(p) − H(D)`; achieved by a backward BSC test channel of crossover `D` with input `X̂ ~ Bernoulli(r)`, `r=(p−D)/(1−2D)`, so the output `X` has marginal `Bernoulli(p)`. For `p=½`: `R(D)=1+D log₂D+(1−D)log₂(1−D)`, the capacity of a BSC with crossover `D`.

**Difference-distortion lower bound.** For `d(x,x̂)=ρ(x−x̂)`, with `φ(D)` the maximum entropy at mean distortion `D`: `R(D) ≥ h(X) − φ(D)`, tight for the Gaussian/squared-error pair. The Gaussian curve is the fixed-variance worst-case benchmark for squared error.

**Parallel Gaussian (reverse water-filling).** Independent `X_i ~ N(0,σ_i²)`, summed squared error: minimizing `Σ ½log₂(σ_i²/D_i)` at `Σ D_i = D` gives `D_i = min(θ, σ_i²)` with `θ` chosen so `Σ D_i = D` — pour equal distortion `θ` into every component and drop those with variance below the water level.

## Finite-source computation

`R(D)` for a finite source/distortion is the convex program `min_q I(X;X̂)` s.t. `E[d]≤D`; the closed forms above are the canonical checks.

```python
import numpy as np

def mutual_information(P_x, Q):                 # I(X; X_hat), bits
    P_x = np.asarray(P_x, float); Q = np.asarray(Q, float)
    P_xhat = P_x @ Q
    I = 0.0
    for i, pi in enumerate(P_x):
        for j, qij in enumerate(Q[i]):
            if qij > 0 and P_xhat[j] > 0:
                I += pi * qij * np.log2(qij / P_xhat[j])
    return I

def expected_distortion(P_x, Q, D_matrix):      # E[d] = sum_ij P_i q_i(j) d_ij
    P_x = np.asarray(P_x, float); Q = np.asarray(Q, float)
    return float(sum(P_x[i]*Q[i,j]*D_matrix[i,j]
                     for i in range(Q.shape[0]) for j in range(Q.shape[1])))

def entropy_bits(p):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p*np.log2(p)).sum())

# Closed-form R(D) for the canonical sources -------------------------------

def rate_distortion_gaussian(sigma2, D):
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")
    if D < 0:
        raise ValueError("D must be non-negative")
    if D == 0:
        return np.inf
    if D >= sigma2:
        return 0.0
    return 0.5*np.log2(sigma2/D)

def rate_distortion_bernoulli(p, D):
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    if D < 0:
        raise ValueError("D must be non-negative")
    p = min(p, 1-p)
    return entropy_bits([p, 1-p]) - entropy_bits([D, 1-D]) if D < p else 0.0

# Blahut-Arimoto: minimize I(X;X_hat) over test channels at a Lagrange slope s<=0.
# Solves R(D) for an arbitrary finite source/distortion by sweeping s.
def blahut_arimoto(P_x, D_matrix, s, iters=200):
    P_x = np.asarray(P_x, float)
    D_matrix = np.asarray(D_matrix, float)
    if s > 0:
        raise ValueError("s must be non-positive for rate-distortion")
    a, b = D_matrix.shape
    q = np.ones(b)/b                                   # output marginal q(x_hat)
    for _ in range(iters):
        W = q[None, :] * np.exp(s * D_matrix)          # unnormalized q(x_hat|x)
        W /= W.sum(axis=1, keepdims=True)              # normalize rows
        q = P_x @ W                                    # update output marginal
    R = mutual_information(P_x, W)                      # bits/symbol
    D = expected_distortion(P_x, W, D_matrix)
    return R, D                                        # a point (R(D), D) on the curve
```

## One-line takeaway

To reproduce a source within average distortion `D`, you need at least `R(D)=min_{q:E[d]≤D} I(X;X̂)` bits per symbol — convex, equal to `H(X)` at `D=0` for discrete exact-reproduction distortion, computable in closed form (`½log₂(σ²/D)` for the Gaussian at `D>0`, `H(p)−H(D)` for the binary source) — achievable above and impossible below, the lossy mirror of channel capacity and of the source-coding theorem.
