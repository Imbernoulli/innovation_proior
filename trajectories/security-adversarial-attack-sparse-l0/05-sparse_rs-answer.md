**Problem (from step 4).** Every prior rung was limited by something other than the budget itself — Pixle
guessed locations, JSMA committed greedily, OnePixel ran out of generations, SparseFool leaned on a
brittle local-linear boundary (mean ASR ≈ 0.187, swinging 0.113–0.280 across models with boundary
linearity). None is *built for the discrete `L0` set*.

**Key idea (Sparse-RS, L0 random search).** The `L0` budget is a *combinatorial support* `M` of `k`
pixels, not a norm ball, so continuous projection lurches and stalls. Use random search: keep the best
candidate, sample a *feasible sparse* candidate, query the margin `L = f_y - max_{r!=y} f_r`, keep it iff
the loss improves or the margin is already negative. Proposal: colors are cube corners `{0,1}^c` (use each
pixel maximally); the support move is a *swap* of equal-size sets between `M` and its complement (always
feasible, `|M|=k` preserved); the swap fraction *decays* on a piecewise-constant schedule — broad
exploration early, single-pixel refinement late.

**Why it is the strongest.** The proposal is *derived from* `L0` structure, so every candidate is feasible
with no projection. A coupon-collector analysis of single-pixel swaps gives
`E[t_k] < (d-k)k(ln k + 2)/(m-k)`, sublinear in `d` — beating the `O(d)` of any black-box gradient
estimate. And it gets `10000` queries per image versus the prior rungs' ~15–120.

**Scaffold edit / hyperparameters.** No `torchattacks` wrapper exists, so the full L0 attack is *inlined*.
`n_queries=10000`, `p_init=0.8`, `eps=pixels=24`; per-image index sets `b` (perturbed) / `be` (clean),
pixel `p`→`(p//W, p%W)` matching the harness's channel-wise `L0` count; swap
`eps_it=max(int(p_selection(it)*eps),1)`; schedule divisors `{2,4,5,6,8,10,12,15,20}`; spend queries only
on still-positive-margin images; single-pixel entrant resampled until its color differs.

**What to watch.** A *step change*: mean ASR clears 0.187 decisively, above 0.85 on every model, with
`Engstrom` near the top — and a *narrower* per-model spread than SparseFool, since random search over
feasible supports does not hinge on any one boundary's linearity.

```python
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    pixels: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    """Sparse-RS L0 black-box attack (Croce et al., AAAI 2022)."""
    import torch
    import torch.nn.functional as F

    _ = (n_classes,)
    model.eval()

    n_queries = 10000
    p_init = 0.8
    eps = int(pixels)

    x = images.detach().clone().to(device)
    y = labels.detach().clone().to(device)
    B, C, H, W = x.shape
    n_pixels = H * W

    def _margin_and_loss(xb, yb):
        with torch.no_grad():
            logits = model(xb)
        u = torch.arange(xb.shape[0], device=xb.device)
        y_corr = logits[u, yb].clone()
        logits[u, yb] = -float("inf")
        y_others = logits.max(dim=-1)[0]
        margin = y_corr - y_others
        return margin, margin  # 'margin' loss variant

    def _p_selection(it):
        # Rescaled schedule (see Sparse-RS paper Fig. 3 / rs_attacks.py).
        it = int(it / n_queries * 10000)
        if 0 < it <= 50:
            return p_init / 2
        if 50 < it <= 200:
            return p_init / 4
        if 200 < it <= 500:
            return p_init / 5
        if 500 < it <= 1000:
            return p_init / 6
        if 1000 < it <= 2000:
            return p_init / 8
        if 2000 < it <= 4000:
            return p_init / 10
        if 4000 < it <= 6000:
            return p_init / 12
        if 6000 < it <= 8000:
            return p_init / 15
        if 8000 < it:
            return p_init / 20
        return p_init

    def _rand_colors(shape):
        # Binary {0,1} random colors, as in Sparse-RS default.
        return torch.randint(0, 2, shape, device=device, dtype=x.dtype)

    # ---- Initialise: random eps pixels per image with random binary colors.
    x_best = x.clone()
    b_all = torch.zeros(B, eps, dtype=torch.long, device=device)
    be_all = torch.zeros(B, n_pixels - eps, dtype=torch.long, device=device)
    for i in range(B):
        perm = torch.randperm(n_pixels, device=device)
        ind_p = perm[:eps]
        ind_np = perm[eps:]
        x_best[i, :, ind_p // W, ind_p % W] = _rand_colors((C, eps)).clamp(0.0, 1.0)
        b_all[i] = ind_p
        be_all[i] = ind_np

    margin_min, loss_min = _margin_and_loss(x_best, y)

    for it in range(1, n_queries):
        idx_to_fool = (margin_min > 0.0).nonzero().squeeze(-1)
        if idx_to_fool.numel() == 0:
            break

        x_curr = x[idx_to_fool].clone()
        x_best_curr = x_best[idx_to_fool].clone()
        y_curr = y[idx_to_fool]
        margin_curr = margin_min[idx_to_fool].clone()
        loss_curr = loss_min[idx_to_fool].clone()
        b_curr = b_all[idx_to_fool].clone()
        be_curr = be_all[idx_to_fool].clone()

        x_new = x_best_curr.clone()
        eps_it = max(int(_p_selection(it) * eps), 1)
        ind_p = torch.randperm(eps, device=device)[:eps_it]
        ind_np = torch.randperm(n_pixels - eps, device=device)[:eps_it]

        for i in range(x_new.shape[0]):
            p_set = b_curr[i, ind_p]
            np_set = be_curr[i, ind_np]
            # Restore previously-perturbed positions to clean.
            x_new[i, :, p_set // W, p_set % W] = x_curr[i, :, p_set // W, p_set % W]
            # Perturb newly-selected positions with random binary colors.
            if eps_it > 1:
                x_new[i, :, np_set // W, np_set % W] = _rand_colors((C, eps_it)).clamp(0.0, 1.0)
            else:
                old = x_new[i, :, np_set // W, np_set % W].clone()
                new = old.clone()
                tries = 0
                while (new == old).all() and tries < 16:
                    new = _rand_colors((C, 1)).clamp(0.0, 1.0)
                    tries += 1
                x_new[i, :, np_set // W, np_set % W] = new

        margin, loss = _margin_and_loss(x_new, y_curr)

        idx_improved = (loss < loss_curr).float()
        idx_miscl = (margin < -1e-6).float()
        idx_keep = torch.max(idx_improved, idx_miscl)
        nkeep = int(idx_keep.sum().item())

        # Update loss whenever loss improves.
        upd_loss = (idx_improved > 0).nonzero().squeeze(-1)
        if upd_loss.numel() > 0:
            loss_min[idx_to_fool[upd_loss]] = loss[upd_loss]

        if nkeep > 0:
            upd = (idx_keep > 0).nonzero().squeeze(-1)
            margin_min[idx_to_fool[upd]] = margin[upd]
            x_best[idx_to_fool[upd]] = x_new[upd]

            # Swap mask indices for the accepted moves.
            # `upd` comes from .squeeze(-1), so the batch dim is preserved
            # (shape [K] with K>=1); always use the 2-D batched form.
            t = b_curr[upd].clone()
            te = be_curr[upd].clone()
            t[:, ind_p] = be_curr[upd][:, ind_np]
            te[:, ind_np] = b_curr[upd][:, ind_p]
            b_all[idx_to_fool[upd]] = t
            be_all[idx_to_fool[upd]] = te

    return x_best.detach()
```
