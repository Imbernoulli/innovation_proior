# Sparse-RS L0 Branch

The `L0` branch is a score-only random search over sparse supports. It attacks by maintaining exactly `k` perturbed spatial pixels, assigning those pixels binary corner colors in `{0,1}^c`, swapping perturbed and clean pixels while preserving support size, and accepting candidates whose loss improves or whose margin is already negative.

## Method

- Objective, untargeted: `L(z) = f_y(z) - max_{r != y} f_r(z)`. Negative margin means misclassification.
- State: `M`, a set of `k` perturbed pixels, plus colors on `M`.
- Initialization: sample `k` pixels uniformly from the `h*w` spatial positions; set their colors to random corners of `[0,1]^c`.
- Proposal: choose `A subset M` and `B subset U\M` with equal size; restore `A` to the clean image and set `B` to fresh binary corner colors, so `M' = (M\A) union B` and `|M'| = k`.
- Schedule: in the reference implementation, the loop starts at `it=1`; after optional rescaling to `N=10000`, the `L0` branch uses divisors `{2,4,5,6,8,10,12,15,20}` on intervals `(0,50]`, `(50,200]`, `(200,500]`, `(500,1000]`, `(1000,2000]`, `(2000,4000]`, `(4000,6000]`, `(6000,8000]`, `(8000,N]`.
- Swap size: `eps_it = max(int(alpha_i * k), 1)`, matching the GitHub code’s floor through `int`, not rounding.
- Accept: update `loss_min` if the loss improves; update the best candidate/support if the loss improves or `margin < -1e-6`.

## Query Bound

For binary `x in {0,1}^d`, a linear model, and one-pixel swaps `alpha_i = 1/k`, the white-box optimum is to choose the `k` smallest entries of

```text
w_hat_x = y*w_x ⊙ (1 - 2x).
```

The score-only analysis relaxes exact recovery to finding `k` coordinates among the `m` smallest. If the current support contains `i` such coordinates, the probability that one swap increases this count is

```text
P[i -> i+1] = (k-i)/k * (m-i)/(d-k).
```

The geometric waiting time is therefore

```text
E[t_{i+1} - t_i] = (d-k)k / ((k-i)(m-i)).
```

Summing and reindexing with `j = k-i`,

```text
E[t_k] = (d-k)k * sum_{i=0}^{k-1} 1/((k-i)(m-i))
       = (d-k)k * sum_{j=1}^k 1/(j(j+m-k)).
```

For `m > k`, partial fractions give

```text
sum_{j=1}^k 1/(j(j+m-k))
  = 1/(m-k) * sum_{j=1}^k (1/j - 1/(j+m-k))
  = 1/(m-k) * (H_k - H_m + H_{m-k}).
```

Using `ln k < H_k <= ln k + 1`,

```text
H_k - H_m + H_{m-k}
  < ln k - ln m + ln(m-k) + 2
  < ln k + 2,
```

so

```text
E[t_k] < (d-k)k(ln k + 2)/(m-k).
```

When `m-k` grows with `d`, this beats the naive `O(d)` weight-estimation route. If `m=k`, the partial-fraction form is not valid and the sum is `sum_{j=1}^k 1/j^2`, giving the linear-in-`d` worst-case bound `< (pi^2/6)(d-k)k`.

## Reference-Faithful Code

```python
def run_attack(model, images, labels, pixels, device, n_classes):
    """L0 random-search attack faithful to the canonical rs_attacks.py L0 branch."""
    import torch
    import torch.nn.functional as F

    _ = (n_classes,)
    model.eval()
    n_queries = 10000
    p_init = 0.8
    eps = int(pixels)

    x = images.detach().clone().to(device)
    y = labels.detach().clone().to(device)
    batch, channels, height, width = x.shape
    n_pixels = height * width

    def margin_and_loss(xb, yb):
        with torch.no_grad():
            logits = model(xb)
        xent = F.cross_entropy(logits, yb, reduction="none")
        rows = torch.arange(xb.shape[0], device=xb.device)
        y_corr = logits[rows, yb].clone()
        logits[rows, yb] = -float("inf")
        y_others = logits.max(dim=-1)[0]
        margin = y_corr - y_others
        loss = margin
        _ = xent
        return margin, loss

    def random_choice(shape):
        return torch.sign(2 * torch.rand(shape, device=device, dtype=x.dtype) - 1.0)

    def p_selection(it):
        it = int(it / n_queries * 10000)
        if 0 < it <= 50:
            p = p_init / 2
        elif 50 < it <= 200:
            p = p_init / 4
        elif 200 < it <= 500:
            p = p_init / 5
        elif 500 < it <= 1000:
            p = p_init / 6
        elif 1000 < it <= 2000:
            p = p_init / 8
        elif 2000 < it <= 4000:
            p = p_init / 10
        elif 4000 < it <= 6000:
            p = p_init / 12
        elif 6000 < it <= 8000:
            p = p_init / 15
        elif 8000 < it:
            p = p_init / 20
        else:
            p = p_init
        return p

    x_best = x.clone()
    b_all = torch.zeros(batch, eps, dtype=torch.long, device=device)
    be_all = torch.zeros(batch, n_pixels - eps, dtype=torch.long, device=device)

    for img in range(batch):
        ind_all = torch.randperm(n_pixels, device=device)
        ind_p = ind_all[:eps]
        ind_np = ind_all[eps:]
        x_best[img, :, ind_p // width, ind_p % width] = random_choice(
            (channels, eps)
        ).clamp(0.0, 1.0)
        b_all[img] = ind_p.clone()
        be_all[img] = ind_np.clone()

    margin_min, loss_min = margin_and_loss(x_best, y)

    for it in range(1, n_queries):
        idx_to_fool = (margin_min > 0.0).nonzero().squeeze(-1)
        if idx_to_fool.numel() == 0:
            break

        x_curr = x[idx_to_fool].clone()
        x_best_curr = x_best[idx_to_fool].clone()
        y_curr = y[idx_to_fool]
        loss_min_curr = loss_min[idx_to_fool]
        b_curr = b_all[idx_to_fool].clone()
        be_curr = be_all[idx_to_fool].clone()

        x_new = x_best_curr.clone()
        eps_it = max(int(p_selection(it) * eps), 1)
        ind_p = torch.randperm(eps, device=device)[:eps_it]
        ind_np = torch.randperm(n_pixels - eps, device=device)[:eps_it]

        for img in range(x_new.shape[0]):
            p_set = b_curr[img, ind_p]
            np_set = be_curr[img, ind_np]
            x_new[img, :, p_set // width, p_set % width] = x_curr[
                img, :, p_set // width, p_set % width
            ].clone()

            if eps_it > 1:
                x_new[img, :, np_set // width, np_set % width] = random_choice(
                    (channels, eps_it)
                ).clamp(0.0, 1.0)
            else:
                old_clr = x_new[img, :, np_set // width, np_set % width].clone()
                new_clr = old_clr.clone()
                while (new_clr == old_clr).all().item():
                    new_clr = random_choice((channels, 1)).clone().clamp(0.0, 1.0)
                x_new[img, :, np_set // width, np_set % width] = new_clr.clone()

        margin, loss = margin_and_loss(x_new, y_curr)

        idx_improved = loss < loss_min_curr
        idx_to_update = idx_improved.nonzero().squeeze(-1)
        if idx_to_update.numel() > 0:
            loss_min[idx_to_fool[idx_to_update]] = loss[idx_to_update]

        idx_miscl = margin < -1e-6
        keep = idx_improved | idx_miscl
        idx_keep = keep.nonzero().squeeze(-1)

        if idx_keep.numel() > 0:
            margin_min[idx_to_fool[idx_keep]] = margin[idx_keep].clone()
            x_best[idx_to_fool[idx_keep]] = x_new[idx_keep].clone()

            t = b_curr[idx_keep].clone()
            te = be_curr[idx_keep].clone()
            t[:, ind_p] = be_curr[idx_keep][:, ind_np].clone()
            te[:, ind_np] = b_curr[idx_keep][:, ind_p].clone()
            b_all[idx_to_fool[idx_keep]] = t.clone()
            be_all[idx_to_fool[idx_keep]] = te.clone()

    return x_best.detach()
```

The implementation keeps the `L0` constraint in spatial-pixel space, uses binary corner colors through `sign(rand).clamp(0,1)`, follows the divisor schedule from `rs_attacks.py`, spends queries only on examples with positive current margin, and preserves the reference accept rule.
