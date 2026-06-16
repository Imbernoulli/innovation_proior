STORM+ (STochastic Recursive Momentum, "+") is a fully **parameter-free** stochastic optimizer for
non-convex objectives `F(x) = E_xi[f(x;xi)]`. It keeps STORM's corrected-momentum gradient estimate
(variance reduction with no checkpoint and no mega-batch), and sets both the step size `eta_t` and
the momentum `a_t` adaptively from observed gradient and estimate norms — using **no knowledge of the
smoothness `L`, the gradient-norm bound `G`, or the noise variance `sigma`**. It attains the optimal
`O(1/T^{1/3})` stationarity rate, and `O(1/sqrt(T))` automatically when the problem is noiseless.

## Problem it solves

First-order stochastic non-convex optimization: minimize `F(x) = E_xi[f(x;xi)]` with each sampled
loss `f(.,xi)` being `L`-smooth (expectation over smooth losses), accessing only stochastic
gradients. Goal: output `xbar` after `T` oracle calls with `E||grad F(xbar)||` small. Reach the
optimal rate **without** any checkpoint/mega-batch (the cost of SVRG/SARAH/SPIDER) and **without**
needing the problem constants `L`, `G`, `sigma` to set the step size and momentum (the cost of STORM).

## Key idea

Keep STORM's corrected-momentum estimate, computed with two gradient calls on the same fresh sample
`xi_t` at the current and previous iterate:

    d_t      = grad f(x_t; xi_t) + (1 - a_t) ( d_{t-1} - grad f(x_{t-1}; xi_t) )
    x_{t+1}  = x_t - eta_t d_t

The within-sample difference `grad f(x_t;xi_t) - grad f(x_{t-1};xi_t)` plus `L`-smoothness makes the
estimate error `eps_t = d_t - grad F(x_t)` shrink with the step, giving variance reduction with no
anchor point. The contribution of STORM+ is the **fully adaptive, parameter-free schedule**:

    a_{t+1} = 1 / ( 1 + sum_{i<=t} ||g_i||^2 ) ^ (2/3) ,   g_i = grad f(x_i; xi_i)
    eta_t   = 1 / ( sum_{i<=t} ||d_i||^2 / a_{i+1} ) ^ (1/3)

- `a_t` is set from the cumulative **raw gradient** norms (AdaGrad-style), exponent `2/3`. This
  mimics STORM's `a_t propto L^2 eta_t^2 propto t^{-2/3}` decay without any `L`; the `+1` makes
  `a_1 = 1` and keeps `a_t in (0,1]`; the `2/3` exponent makes the reciprocal increments
  `1/a_{t+1} - 1/a_t` bounded (by `2/3` once `a_t` is small), which the proof needs.
- `eta_t` is set from the cumulative **estimate** norms `||d_i||^2`, exponent `1/3` (not AdaGrad's
  `1/2` — the cube root is what lands the `1/T^{1/3}` rate). Using `||d_i||` (not `||g_i||`) is what
  lets the smoothness/curvature terms `eta_{t-1}^2 ||d_{t-1}||^2` telescope.
- The **`1/a_{i+1}` reweighting inside `eta_t`** is the headline novelty — an adaptive interplay
  between learning rate and momentum. STORM coupled them as `a propto L^2 eta^2` (needs `L`); STORM+
  reverses the coupling so the **step depends on the momentum**, with the extra `1/a ~ t^{2/3}` of
  shrinkage doing `L`'s old job parameter-free.

Initialize `d_1 = g_1 = grad f(x_1; xi_1)` (single sample, no batch). Output `xbar_T` chosen
uniformly at random from `{x_1, ..., x_T}` (standard for non-convex stationarity; in practice use the
last iterate).

## Final algorithm

```
Input: x_1, number of iterations T
Sample xi_1; set d_1 = g_1 = grad f(x_1; xi_1)
for t = 1, ..., T:
    a_{t+1} = 1 / (1 + sum_{i=1}^{t} ||g_i||^2)^{2/3}
    eta_t   = 1 / (sum_{i=1}^{t} ||d_i||^2 / a_{i+1})^{1/3}
    x_{t+1} = x_t - eta_t d_t
    Sample xi_{t+1}; set g_{t+1} = grad f(x_{t+1}; xi_{t+1}),  gtilde_t = grad f(x_t; xi_{t+1})
    d_{t+1} = g_{t+1} + (1 - a_{t+1}) (d_t - gtilde_t)
Return xbar_T ~ Uniform{x_1, ..., x_T}
```

Two oracle calls per iteration (`g_{t+1}` and `gtilde_t` share the sample `xi_{t+1}`); one
full-vector (scalar) step size and momentum per iteration; no checkpoint, no mega-batch.

## Guarantee

Under bounded function range (`max_{x,y}|F(x)-F(y)| <= B`), bounded gradient (`||grad f(x;xi)||^2 <=
G^2`), bounded variance (`E||grad f(x;xi) - grad F(x)||^2 <= sigma^2`), and per-sample `L`-smoothness:

    E||grad F(xbar_T)|| <= O( M / sqrt(T) + kappa * sigma^{1/3} / T^{1/3} ),

with `kappa = O(B^{3/4} + L^{3/2})` and `M = O(1 + L^{9/4} + B^{9/8} + G^5 + (L G^4)^{3/2})`. So the
rate is the optimal `O(1/T^{1/3})` in the stochastic case, and the second term vanishes when
`sigma = 0`, recovering `O(1/sqrt(T))` — i.e. it **adapts to the variance** without knowing it. The
constants `L, G, B, sigma` appear only in `M, kappa`, never in the computed `eta_t, a_t`.

## Why the rate holds (proof shape)

- **Telescoping identity** (McMahan-Streeter, any `p in (0,1)`): `sum_i b_i/(sum_{j<=i}b_j)^p <=
  (1/(1-p))(sum_i b_i)^{1-p}`. Proved by induction; `h(x) = (1/(1-p))(Z-x)^{1-p} + x/Z^p` is concave
  and maximized at `x=0`. Used with `p=2/3` (cube root on `d`-energy) and `p=1/3` (on `gbar`-energy).
- **Offline (`sigma=0`):** `d_t = gbar_t` exactly; smoothness + the telescope give `sum||gbar||^2 <=
  O(1 + L^3 + B^{9/4})`, hence `O(1/sqrt T)`.
- **Error dynamics:** `eps_t = (1-a_t)eps_{t-1} + a_t(g_t - gbar_t) + (1-a_t)Z_t`, with `Z_t =
  (g_t - gtilde_{t-1}) - (gbar_t - gbar_{t-1})` and `||Z_t|| <= 2L eta_{t-1}||d_{t-1}||` by
  smoothness. Bounding `E sum||eps_t||^2` then a two-case split (`E sum||eps||^2` vs `(1/2)E
  sum||gbar||^2`) converts it into `E sum||gbar_t||^2 <= O(M^2 + kappa^2 sigma^{2/3} T^{1/3})`.
- **Adaptive (random) `a_t`:** the reciprocal increments `1/a_{t+1} - 1/a_t` are no longer bounded by
  `2/3`. Define `beta = min{1,1/G^4}`, stopping time `tau* = max{t: a_t >= beta}`. After `tau*`,
  concavity of `y^{2/3}` restores `1/a_{t+1} - 1/a_t <= 2/3`; before `tau*`, `1/a_{t+1} <= 1/betatilde
  = (1/beta^{3/2}+G^2)^{2/3}`. Cross terms are martingale differences killed by Doob's optional
  stopping. The variance term uses the **exponent-`4/3` lemma** `sum a_i/(1+sum_{j<=i}a_j)^{4/3} <=
  12` (dyadic blocking; the `>1` exponent is what makes it a constant, not a log), and Young's
  inequality with `rho = (512L^2)^{1/3}` (exponents `3/2, 3`) splits the mixed `g`-energy/`d`-energy
  products.

## Relation to prior methods

- **STORM (Cutkosky & Orabona 2019):** same corrected-momentum template, but `eta_t = k/(w+sum
  ||g||^2)^{1/3}`, `a_{t+1} = c eta_t^2` with `k, c, w` depending on `L` and `G`; proof breaks without
  them. STORM+ replaces these with the parameter-free `eta_t, a_t` above, shaves a `(log T)^{3/4}`
  factor off the `1/sqrt T` term, and keeps variance adaptivity.
- **SVRG / SARAH / SPIDER / PAGE:** reach `O(1/T^{1/3})` via periodic full/large-batch checkpoint
  gradients and an `L`-tied step cap; STORM+ removes all checkpoints and all problem-constant
  knobs.
- **AdaGrad:** the `sum-of-squared-norms^{-power}` step-size shape is borrowed, but with exponent
  `1/3` (for the VR rate) and applied to the estimate energy `||d||^2`, plus the `1/a` coupling.

## Working code

Filling the `_schedule` slot of the corrected-momentum harness with the parameter-free `eta_t, a_t`:

```python
import torch


def grad(model, loss_fn, x, batch):
    """One stochastic gradient grad f(x; xi), flattened to a single vector."""
    set_flat_params(model, x)
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model, batch)
    loss.backward()
    return flat_grad(model)                       # 1-D tensor over all parameters


class StormPlus:
    """STORM+ : fully parameter-free, no-checkpoint variance reduction.

        a_{t+1} = 1 / (1 + sum_{i<=t} ||g_i||^2) ^ (2/3)
        eta_t   = 1 / (sum_{i<=t} ||d_i||^2 / a_{i+1}) ^ (1/3)
        x_{t+1} = x_t - eta_t d_t
        d_{t+1} = g_{t+1} + (1 - a_{t+1}) (d_t - grad f(x_t; xi_{t+1}))
    """

    def __init__(self, model, loss_fn, x_init):
        self.model = model
        self.loss_fn = loss_fn
        self.x = x_init.clone()              # current iterate x_t
        self.g = None                        # current stochastic gradient g_t
        self.d = None                        # gradient estimate d_t
        self.sum_g2 = 0.0                    # sum ||g_i||^2  (gradient energy)
        self.sum_d2_over_a = 0.0             # sum ||d_i||^2 / a_{i+1}  (estimate energy)

    def step(self, batch=None):
        if self.d is None:                                   # warm start: d_1 = g_1 (one sample)
            if batch is None:
                batch = draw_sample()
            self.g = grad(self.model, self.loss_fn, self.x, batch)
            self.d = self.g.clone()

        g_t = self.g                                         # grad f(x_t; xi_t)

        # momentum a_{t+1} from cumulative gradient norms -> ~ t^{-2/3}, in (0, 1]
        self.sum_g2 += g_t.pow(2).sum().item()
        a_next = 1.0 / (1.0 + self.sum_g2) ** (2.0 / 3.0)

        # step eta_t from cumulative estimate norms reweighted by 1/a (lr <-> momentum coupling)
        self.sum_d2_over_a += self.d.pow(2).sum().item() / a_next
        eta_t = 1.0 / (self.sum_d2_over_a) ** (1.0 / 3.0)

        x_old = self.x.clone()
        with torch.no_grad():
            self.x = self.x - eta_t * self.d                 # x_{t+1} = x_t - eta_t d_t

        # refresh on a FRESH sample: two oracle calls (current and previous iterate), no checkpoint
        next_batch = draw_sample()
        g_next = grad(self.model, self.loss_fn, self.x, next_batch)    # grad f(x_{t+1}; xi_{t+1})
        g_tilde = grad(self.model, self.loss_fn, x_old, next_batch)    # grad f(x_t;     xi_{t+1})
        self.d = g_next + (1.0 - a_next) * (self.d - g_tilde)          # corrected momentum
        self.g = g_next                                                # g_{t+1} for next iteration
        return eta_t, a_next
```
