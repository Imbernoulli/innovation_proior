# signSGD / Signum, distilled

signSGD compresses distributed training by transmitting only the **sign of each stochastic
gradient coordinate** — one bit per parameter — and stepping `x_{k+1} = x_k - delta * sign(g~_k)`.
Its momentum variant **Signum** takes the sign of the momentum, and its distributed form aggregates
worker signs by **majority vote**, so communication is 1-bit in *both* directions. The headline is
that this biased, magnitude-blind, 1-bit scheme provably converges on non-convex objectives at the
same `O(1/sqrt N)` rate as full-precision SGD, with no dimension-dependent variance blowup.

## Problem it solves

In data-parallel distributed training, gradient communication (`64 M d` bits/iteration for `d`
parameters, `M` workers) is the bottleneck, and worsens as workers are added. The goal: cut
per-coordinate communication to ~1 bit in both directions while keeping an SGD-class non-convex
convergence guarantee. Unbiased quantizers (QSGD, TernGrad) bootstrap SGD theory but their
unbiasing randomization inflates variance by `~sqrt(d)`, making the bound vacuous at `d > 10^8`.

## Key idea

Use the **biased** `sign(g~)` directly and bound the damage of the bias. If `g_i = 0`, a wrong sign
has zero descent cost. Otherwise a coordinate's sign is wrong only when noise exceeds signal, so
`P[sign(g~_i) != sign(g_i)] <= P[|g~_i - g_i| >= |g_i|] <= E|g~_i-g_i|/|g_i| <= sigma_i/|g_i|`
(Markov then Jensen). Multiplying by the error cost `|g_i|`, the **magnitude cancels**:
`|g_i| P[wrong] <= sigma_i`. The bias's cost is capped by the noise scale alone, regardless of the
gradient size. The natural geometry is `l_inf` (the step is a `±delta` box; non-stochastic signSGD
is exactly `l_inf` steepest descent, `argmin{g^T v : ||v||_inf <= 1} = -sign(g)`), with progress
measured in `l_1`.

## Assumptions

1. Lower bound: `f(x) >= f*`.
2. Coordinate-wise smoothness: `|f(y) - [f(x)+g(x)^T(y-x)]| <= (1/2) sum_i L_i (y_i-x_i)^2`, vector
   `L >= 0`. (Recovers standard `L`-smoothness with `L := ||L||_inf`; implies `-diag(L) ≺ H ≺ diag(L)`.)
3. Per-coordinate variance: `E[g~] = g`, `E[(g~_i - g_i)^2] <= sigma_i^2`. (Recovers total variance
   `sigma^2 := ||sigma||_2^2`.) A minibatch of size `n_k` squashes coordinate variance to `sigma_i^2/n_k`.

## Algorithms

```
signSGD:   x_{k+1} = x_k - delta * sign(g~_k)
Signum:    m_{k+1} = beta*m_k + (1-beta)*g~_k ;   x_{k+1} = x_k - delta * sign(m_{k+1})
Majority:  server pushes  sign[ sum_m sign(g~_m) ]  to all workers;
           x_{k+1} = x_k - delta * sign[ sum_{m=1}^M sign(g~_m) ]
```

The implementation convention is `sign(0) = +1` via `>= 0`; for an even-worker vote tie, the server
broadcasts `+1`. The no-loss repetition-code proof uses strict majority or a neutral tie rule.

## Convergence

**signSGD (large batch).** `delta_k = 1/sqrt(||L||_1 K)`, `n_k = K`, `N = O(K^2)`:

```
E[ (1/K) sum_k ||g_k||_1 ]^2  <=  (1/sqrt N) [ sqrt(||L||_1)(f_0 - f* + 1/2) + 2||sigma||_1 ]^2.
```

Proof skeleton: coordinate smoothness gives `f_{k+1}-f_k <= -delta_k g_k^T sign(g~_k) +
(delta_k^2/2)||L||_1`; the identity `g_k^T sign(g~_k) = ||g_k||_1 - 2 sum_i |g_{k,i}| I[wrong]`
plus the key bound `|g_{k,i}| P[wrong] <= sigma_{k,i} <= sigma_i/sqrt(n_k)` yields
`E[f_{k+1}-f_k|x_k] <= -delta_k||g_k||_1 + 2(delta_k/sqrt n_k)||sigma||_1 + (delta_k^2/2)||L||_1`;
telescope and substitute. The growing batch is a *systems* win: `N` calls in `O(sqrt N)` iterations,
hence `O(sqrt N)` communication rounds.

**SGD comparison via density.** With `phi(v) := ||v||_1^2/(d||v||_2^2)` (1 = dense, `~1/d` = sparse),
the translated signSGD bound controls the square of the expected average `l_2` norm:
`E[(1/K)sum||g_k||_2]^2 <= (2/sqrt N)[ R_1 L (f_0-f*+1/2)^2 + 4 R_2 sigma^2 ]`, vs SGD's
`(1/sqrt N)[2L(f_0-f*)+sigma^2]`, with density ratios `R_1 = sqrt(phi(L))/phi(g)`,
`R_2 = phi(sigma)/phi(g)`. The sign bound can match or improve on the SGD bound when noise is no denser than the gradient
(`R_2` not `>> 1`); it loses when curvature is much denser than the gradient (`R_1 >> 1`).
**Measured fact:** in real networks (Welford's algorithm, Resnet-20/CIFAR-10), gradient and noise
densities are of the same order and both dense — the favorable regime.

**Small batch (`n_k=1`), unimodal symmetric noise.** Gauss's inequality gives the sharper flip bound
`P[wrong] <= (2/9)/S_i^2` if `S_i := |g_i|/sigma_i > 2/sqrt3`, else `1/2 - S_i/(2 sqrt3)` (strictly
`< 1/2` for nonzero signal; zero signal has zero descent cost). With `B_k = {i : S_i > 2/sqrt3}`:

```
E[ min_k ( sum_{i in B_k} |g_{k,i}| + sum_{i not in B_k} g_{k,i}^2/sigma_i ) ]
        <=  sqrt(3 ||L||_1 / N) (f_0 - f* + 1/2).
```

A mixed norm: high-SNR coords converge in `l_1`, low-SNR coords in variance-weighted `l_2`.

**Majority vote.** (a) Under Assumptions 1-3, strict-majority/no-tie decoding is at least as fast as
single-worker signSGD: for `S>1`, single-worker failure `q <= 1/(1+S^2) < 1/2` (Cantelli), so the
server decodes a repetition code and strict majority (the ML decoder) only lowers error, preserving
`|g_i| P[majority wrong] <= sigma_i`. A deterministic `>= 0` tie is the implementation convention,
not the neutral tie rule used for this no-loss statement. (b) Under unimodal symmetry, variance reduces by `sqrt M`:

```
E[(1/K) sum_k ||g_k||_1]^2  <=  (1/sqrt N)[ sqrt(||L||_1)(f_0-f*+1/2) + (2/sqrt M)||sigma||_1 ]^2,
```

via `P[Z <= M/2] <= 1/(sqrt M S)` for `Z ~ Binom(M,p)` (Cantelli with margin `eps = p - 1/2`), with
`N` counted per worker. Cost:
`2 M d` bits/iteration (vs SGD's `64 M d`). Symmetry is needed because skewed/bimodal noise (e.g.
`P[X=50]=0.1, P[X=-1]=0.9`, mean `+0.1` but `P[sign=correct]=0.1`) makes more workers *worse*; the
CLT makes symmetry mild for any non-tiny batch.

**Signum.** Master lemma: for `x_{k+1}=x_k - delta_k sign(v_k)`, if `E[sum_i |g_{k,i}|
P[sign(v_{k,i}) != sign(g_{k,i})|x_k]] <= xi(k) -> 0`, then `(1/(K-C)) sum E||g_k||_1 <= [f_C-f* +
2 sum delta_k xi(k) + sum delta_k^2 ||L||_1/2]/[(K-C) min delta_k]` (sufficient: `sum_i E|v_k[i] -
g_k[i]| <= xi(k)`). For Signum, take `v_k` as the normalized stochastic momentum, which has the same
sign as the recursive state; a bias-variance split bounds
`sum_i E|m~_k[i] - g_k[i]| <= (2/sqrt(k+1))(8||L||_1 delta * beta/(1-beta) + sqrt3 ||sigma||_1 sqrt(1-beta))`.
The variance uses a coordinate-wise martingale bound `E[(sum alpha_l Z_l)^2] <= sum alpha_l^2 sigma_l^2`
(with `sigma_l^2` reduced by that step's minibatch size); the bias uses `||g(x+eps s)-g(x)||_1 <= 2 eps ||L||_1` (Taylor +
Hessian psd/nsd split) chained over steps. With `delta_k=delta/sqrt(k+1)`, `n_k=k+1`, `N=O(K^2)`:

```
E[(1/(K-C)) sum_{k=C}^{K-1} ||g_k||_1]^2
   = O( (1/sqrt N)[ (f_C-f*)/delta + (1+log N)( delta||L||_1/(1-beta) + ||sigma||_1 sqrt(1-beta) ) ]^2 ).
```

`beta` is an explicit **bias-variance knob**: `beta -> 1` kills the `||sigma||_1 sqrt(1-beta)`
variance term but inflates the `delta||L||_1/(1-beta)` bias. Warmup `C` (`= 54` for `beta=0.9`) lets
the stale-momentum tail decay; during warmup, accumulate momentum but step with `sign(g~)`.

## Relation to prior methods

- **signSGD = zero-memory Adam.** Adam's step `~ <g~>_{beta1}/sqrt(<g~^2>_{beta2})`; sending both EMA
  timescales `beta1, beta2 -> 0` gives `g~/sqrt(g~^2) = sign(g~)`. This places signSGD/Signum in the
  Rprop/RMSprop/Adam family.
- **vs QSGD/TernGrad:** those keep the gradient *unbiased* via randomization (variance `~sqrt(d)`,
  vacuous at scale, log factors on the return path). signSGD embraces the *biased* sign, bounding its
  cost by the noise; majority vote keeps both directions 1-bit.
- **vs Rprop:** Rprop is sign-based but magnitude-adaptive and breaks minibatch averaging; signSGD
  confronts the stochastic biased sign with a convergence theory.

## Working code

Parameter-server codec (signSGD + majority vote), filling the `encode`/`aggregate`/`decode` slots;
`sign(0) -> +1` via `>= 0` so every coordinate emits a binary message and vote ties map to `+1`. The
update direction is `±1` per coordinate, scaled by `delta` in the loop.

```python
import torch


class SignSGDCodec:
    """signSGD with majority-vote aggregation: 1 bit per coordinate each direction."""

    def __init__(self):
        self.state = {}

    def encode(self, grad, name):
        shape = grad.shape
        bits = (grad.flatten() >= 0).to(torch.uint8)         # sign(g~), stored as uint8
        return [bits], shape

    def aggregate(self, messages):
        votes = sum(b.to(torch.float32) * 2 - 1 for [b] in messages)  # sum_m sign(g~_m)
        return [(votes >= 0).to(torch.uint8)]                # majority vote -> 1 bit back

    def decode(self, received, ctx):
        [bits] = received
        return (bits.to(torch.float32) * 2 - 1).view(ctx)    # bits -> +-1, step = -delta*this


class SignumCodec(SignSGDCodec):
    """Signum: sign of the per-parameter momentum m = beta*m + (1-beta)*g~."""

    def __init__(self, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.buf = {}

    def encode(self, grad, name):
        shape = grad.shape
        g, beta = grad.flatten(), self.momentum
        m = self.buf.get(name)
        if m is None:
            m = torch.zeros_like(g)
        m = beta * m + (1 - beta) * g                        # m_{k+1}=beta*m_k+(1-beta)*g~_k
        self.buf[name] = m
        return [(m >= 0).to(torch.uint8)], shape             # sign(m_{k+1})
```

Local Signum optimizer (same rule as an update; `momentum=0` recovers signSGD). Mirrors the
canonical Signum update `state = momentum*state + (1-momentum)*grad ; w = w - lr*sign(state)`,
with `sign(0) -> +1` through `>= 0`:

```python
import torch


class Signum:
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.params = list(params)
        self.lr, self.momentum, self.wd = lr, momentum, weight_decay
        self.state = {id(p): None for p in self.params}

    @torch.no_grad()
    def step(self):
        beta = self.momentum
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            if self.wd != 0:
                g = g.add(p, alpha=self.wd)                   # weight decay
            if beta != 0:
                m = self.state[id(p)]
                if m is None:
                    m = torch.zeros_like(g)
                m.mul_(beta).add_(g, alpha=1 - beta)          # state=beta*state+(1-beta)*grad
                self.state[id(p)] = m
                direction = torch.where(m >= 0, torch.ones_like(m), -torch.ones_like(m))
                p.add_(direction, alpha=-self.lr)             # x <- x - delta*sign(m)
            else:
                direction = torch.where(g >= 0, torch.ones_like(g), -torch.ones_like(g))
                p.add_(direction, alpha=-self.lr)             # signSGD: x <- x - delta*sign(g~)
```
