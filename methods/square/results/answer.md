# Square Attack, distilled

Square Attack is a score-based black-box adversarial attack via **random search**. It needs
only the model's output scores (no gradients, no weights), so it is immune to gradient masking,
and it is query-efficient. At each iteration it proposes a localized **square-shaped** update at
a random position, with values chosen so the perturbation stays on the boundary of the `Lp`-ball,
and accepts the update only if it lowers the loss or has already crossed the decision boundary.
This is the `Linf` version.

## Problem it solves

Untargeted score-based black-box attack under an `Linf` budget: for a correctly classified
`(x, y)`, find `x_hat` with `argmax_k f_k(x_hat) != y`, `||x_hat - x||_inf <= eps`,
`x_hat in [0,1]^d`, using as few forward queries as possible. Primary metric: success rate
within a per-sample query budget `N`; tie-break: average queries (lower better).

## Key idea

Greedy random search on the margin loss `L = f_y - max_{k!=y} f_k`, with a carefully designed
proposal distribution:

- **Random search, not gradient estimation.** Uses only forward scores (one query per
  candidate), so it bypasses the `O(d)`-per-step query cost of NES/SPSA finite differences and
  is unaffected by gradient masking.
- **Stay on the boundary (corners).** Successful `Linf` perturbations usually sit at corners
  `max(0, x_i - eps)` or `min(1, x_i + eps)`. Updates take value `+/- 2*eps` so that projection
  onto the `Linf`-ball followed by `[0,1]` clipping sends each touched component to one of
  those extremes — the full budget is spent every step, and a later square can overwrite an
  earlier region (unlike SimBA's non-undoable orthogonal moves).
- **Square-shaped updates.** For a fixed number `k` of changed pixels, the square is the shape
  that maximizes the number of first-layer convolutional receptive fields fully covered by the
  update (Prop. below), i.e. that maximizes the worst-case change in first-layer activations.
  Unlike fixed-grid/tiling attacks, the position is sampled freely each iteration and the size
  follows a coarse-to-fine schedule.
- **Single spatial sign per color channel.** Within each color channel, one `+/- 2eps` sign is
  shared across the whole square; different channels get their own signs. Against a
  piecewise-constant gradient direction `v`, the spatially shared sign correlates at the
  `||v||_1` scale vs `||v||_2` for independent pixel signs, a factor `h` larger inner product.
- **Coarse-to-fine size schedule.** `p` = fraction of pixels changed, square side
  `h = round(sqrt(p * n_features / c))`; `p` halves at fixed iteration breakpoints (rescaled to
  the budget) — the analogue of step-size decay.
- **Vertical-stripe initialization.** Start at the `Linf` boundary with random width-1 vertical
  stripes (`+/- eps` per column per channel), a structured high-frequency pattern to which CNNs
  are known to be sensitive.

## Algorithm (Linf)

```
x_best <- clip(x + eps * vertical_stripe_signs, 0, 1)        # boundary init
margin_min, loss_min <- margin_and_loss(x_best, y)          # 1 query
for it in 0..N-1:
    if all samples fooled (margin_min <= 0): break
    on the still-unfooled samples only:
        p <- schedule(it, N);   s <- max(round(sqrt(p * n_features / c)), 1)
        (vh, vw) <- uniform random top-left with the s x s square inside the image
        delta <- zeros;  delta[:, vh:vh+s, vw:vw+s] <- 2*eps * sign(rand[c,1,1])
        x_new <- clip( Linf_project(x_best + delta, x, eps), 0, 1 )
        margin, loss <- margin_and_loss(x_new, y)           # 1 query
        loss_improved <- (loss < loss_min)
        update loss_min only on loss_improved
        accepted <- loss_improved OR (margin <= 0)          # force-accept a flip
        update margin_min and x_best on accepted
```

Defaults: `p_init = 0.8`, `loss = "margin"`, `n_restarts = 1`, schedule rescaled to `N`.

## Why squares (proposition)

Let `delta` have `||delta||_0 = k >= s^2` nonzero entries and `||delta||_inf <= eps`, and let
`w in R^{s x s}` be an unknown first-layer filter, `z = delta * w`. Then
`|z_{u,v}| <= eps * sum_{i,j} |w_{i,j}| * 1[index in support of delta]`, maximized when the
`s x s` receptive window `C(u,v)` is fully covered by nonzero entries. So the update's support
should maximize the number of `s x s` sub-squares it contains. Among shapes of area `k`, the
maximum number of contained `s x s` squares is
`N* = (a - s + 1)(b - s + 1) + (r - s + 1)^+` with `a = floor(sqrt(k))`, `b = floor(k/a)`,
`r = k - ab`. The maximizer is the (near-)square shape; if `k = l^2` then `a = b = l` exactly —
the support is a square.

## Why single sign (L1 vs L2)

For a piecewise-constant gradient direction `v` and one random sign shared spatially over a
block, `<delta^single, v> = rho * sum_block v = rho * ||v_block||_1`, so
`E|<delta^single, v>| = Theta(||v||_1)`. For independent signs per pixel, Khintchine gives
`E|<delta^multiple, v>| = Theta(||v||_2)`. Since `||v||_1 >> ||v||_2` for a constant block
(`h^2` vs `h`), the spatially shared-sign update is `Theta(h)` more aligned with the gradient.
The implemented Linf update applies this argument per color channel, sampling `[c,1,1]` signs
rather than collapsing all channels to one sign.

## Convergence (smooth, unconstrained variant)

For `L`-smooth `g`, update `x_{t+1} = x_t + delta_t` if it lowers `g` else `x_t`. Smoothness gives
`g(x_{t+1}) <= g(x_t) + min{0, <grad g(x_t), delta_t> + (L/2)||delta_t||^2}`. Applying
`2 min{a,b} = a+b-|a-b|` with `a=0`, then `|A+B| >= |A|-|B|`, gives
`g(x_{t+1}) <= g(x_t) + (1/2)<grad g,delta_t> + (L/2)||delta_t||^2
- (1/2)|<grad g,delta_t>|`. If conditionally on `x_t`,
`E[delta_t]=0`, `E||delta_t||^2 <= gamma_t^2 C`, and
`E|<delta_t, v>| >= Ctilde gamma_t ||v||` for every `v`, then with `v=grad g(x_t)`:

```
E g(x_{t+1}) <= E g(x_t) - (Ctilde gamma_t / 2) E||grad g(x_t)|| + (L C gamma_t^2 / 2).
```

Summing `t = 0..T` with `gamma_t = gamma/sqrt(T)`:

```
min_{t<=T} E||grad g(x_t)|| <= (2 / (gamma Ctilde sqrt(T))) ( g(x_0) - E g(x_{T+1}) + gamma^2 C L / 2 ).
```

i.e. `O(1/sqrt(T))` to a critical point. The correlation assumption holds for the
independent-sign square (`E||delta||^2 = 4 c eps^2 h^2`,
`E|<delta,v>| >= (sqrt2 eps h^2 / w^2)||v||_2` via Khintchine plus convexity of the norm) but
fails for the per-channel shared-sign square (counterexample `h=2`, `v^i_{k,l} = (-1)^{kl}`
gives `<v,delta> = 0` in every channel). So the guarantee covers the multi-sign variant; the
per-channel shared-sign version is the image-structured heuristic, and the same `L1`-alignment
that motivates it is why it falls outside the worst-case `L2` bound.

## Working code (Linf)

```python
import math
import torch
import torch.nn.functional as F


class SquareLinf:
    def __init__(self, model, eps=8 / 255, n_queries=5000, p_init=0.8,
                 loss="margin", resc_schedule=True, seed=0):
        self.model = model
        self.eps = eps
        self.n_queries = n_queries
        self.p_init = p_init
        self.loss = loss
        self.rescale_schedule = resc_schedule
        self.seed = seed

    def margin_and_loss(self, x, y):
        logits = self.model(x)                              # one query per row
        xent = F.cross_entropy(logits, y, reduction="none")
        u = torch.arange(x.shape[0], device=x.device)
        y_corr = logits[u, y].clone()
        logits[u, y] = -float("inf")
        y_other = logits.max(dim=-1)[0]
        margin = y_corr - y_other
        if self.loss == "margin":
            return margin, margin
        return margin, -1.0 * xent

    def p_selection(self, it):
        if self.rescale_schedule:
            it = int(it / self.n_queries * 10000)
        if   10 < it <= 50:    p = self.p_init / 2
        elif 50 < it <= 200:   p = self.p_init / 4
        elif 200 < it <= 500:  p = self.p_init / 8
        elif 500 < it <= 1000: p = self.p_init / 16
        elif 1000 < it <= 2000: p = self.p_init / 32
        elif 2000 < it <= 4000: p = self.p_init / 64
        elif 4000 < it <= 6000: p = self.p_init / 128
        elif 6000 < it <= 8000: p = self.p_init / 256
        elif 8000 < it:         p = self.p_init / 512
        else:                   p = self.p_init
        return p

    def random_sign(self, shape, device):
        return torch.sign(2 * torch.rand(shape, device=device) - 1)

    def random_int(self, low, high, shape, device):
        t = low + (high - low) * torch.rand(shape, device=device)
        return t.long()

    @torch.no_grad()
    def perturb(self, x, y):
        torch.manual_seed(self.seed)
        if x.is_cuda:
            torch.cuda.random.manual_seed(self.seed)
        c, h, w = x.shape[1:]
        n_features = c * h * w

        # vertical-stripe boundary init
        adv = torch.clamp(
            x + self.eps * self.random_sign([x.shape[0], c, 1, w], x.device), 0.0, 1.0)
        margin_min, loss_min = self.margin_and_loss(adv, y)        # 1 query / sample

        for it in range(self.n_queries):
            idx = (margin_min > 0.0).nonzero().flatten()           # unfooled only
            if len(idx) == 0:
                break
            x_curr, x_best_curr, y_curr = x[idx], adv[idx], y[idx]

            p = self.p_selection(it)
            s = max(int(round(math.sqrt(p * n_features / c))), 1)
            vh = int(self.random_int(0, h - s, [1], x.device).item())
            vw = int(self.random_int(0, w - s, [1], x.device).item())

            deltas = torch.zeros([c, h, w], device=x.device)
            deltas[:, vh:vh + s, vw:vw + s] = (
                2.0 * self.eps * self.random_sign([c, 1, 1], x.device))  # one spatial sign/channel

            x_new = x_best_curr + deltas
            x_new = torch.min(torch.max(x_new, x_curr - self.eps),
                              x_curr + self.eps)                   # Linf projection
            x_new = torch.clamp(x_new, 0.0, 1.0)                   # image box

            margin, loss = self.margin_and_loss(x_new, y_curr)     # 1 query / sample

            improved = (loss < loss_min[idx]).float()
            loss_min[idx] = improved * loss + (1.0 - improved) * loss_min[idx]
            miscl = (margin <= 0.0).float()                        # force-accept a flip
            improved = torch.max(improved, miscl)
            margin_min[idx] = improved * margin + (1.0 - improved) * margin_min[idx]
            sel = improved.reshape([-1, *([1] * (x.ndim - 1))])
            adv[idx] = sel * x_new + (1.0 - sel) * x_best_curr

            if (margin_min <= 0.0).all():
                break

        return adv
```
