SparseFool came back with the mixed signature I predicted: mean ASR $\approx 0.187$ — `0.167` on `Rebuffi-R18-L2`, `0.113` on `Augustin-L2`, `0.280` on `Engstrom-L2`. It beat OnePixel's $0.153$ mean but *trailed* it on `Augustin` ($0.113$ vs $0.18$) while leading clearly on `Engstrom` ($0.280$ vs $0.18$). That non-uniform result is the diagnosis in numbers: the engine is local-linear, so where the boundary best fits a hyperplane it wins decisively, but on the least-linear model twenty relinearizations chase a boundary the linear model keeps mis-estimating. Stepping back across all four rungs, every one is limited by *something other than the budget itself* — Pixle guessed locations, JSMA committed greedily to a local first-order signal, OnePixel ran out of generations, SparseFool leaned on a brittle local-linear boundary. None is *built for the discrete $L0$ set*. That is the gap to close.

The realization that reframes everything is that the $L0$ budget is not a norm radius — it is a *combinatorial support*. It says only a selected set $M$ of $k$ pixels may move at all; once a pixel is in $M$, its channels can go anywhere in $[0,1]$. So the object to optimize is not a dense vector in a ball — it is a *pair*, a support $M$ and the colors on it. This is why every continuous method stalled: in an $L_\infty$ or $L_2$ attack you step and project back into a smooth convex ball, but here projection means *choosing* a support, a discrete top-$k$, so a small continuous change before projection causes a large support change after, the iteration lurches, and the support settles poorly. And any gradient-based form has the budget fighting the gradient, which wants to spread signal over many pixels even though only $k$ survive. So I drop continuous optimization altogether and use the one primitive that needs nothing but feasible candidates and scalar comparisons: *random search*. Keep the best candidate, sample a new legal sparse candidate, query its loss, keep it if it improves. The loop is trivial; the entire problem is the *distribution over legal sparse candidates*, and getting that distribution right is what makes this the strongest rung.

I propose **Sparse-RS**, the $L0$ random search. I split a candidate into support and color and design the proposal for each. The objective is the untargeted margin

$$ L(z) = f_y(z) - \max_{r\neq y} f_r(z), $$

minimized; its *sign* is the misclassification certificate, $L(z) < 0$ is exactly a flip, so I never need a separate success test. For colors: because the budget penalizes magnitude not at all — a pixel in $M$ costs the same whether nudged or slammed to an extreme — I use *corners* of the color cube, $\{0,1\}^c$, spending each precious pixel maximally. For the support, the move that never leaves the budget is a *swap*: pick $A$ from the current perturbed support $M$ and $B$ from the clean complement of equal size, restore $A$ to clean, give $B$ fresh corner colors, so $M' = (M\setminus A)\cup B$ and $|M'| = k$ always. Every candidate is feasible by construction — no projection, no clipping, no rejected samples, the structural cleanliness all the budget-bookkeeping pain on earlier rungs was groping toward.

The swap *size* must change over the run, and this is the heart of the method. Swap many pixels early and I escape a poor random initial support fast; keep swapping many late and I destroy the good support I have found; swap one from the start and I refine gently but spend far too many queries finding the right region. So the swap fraction *decays* — large for exploration, small for refinement — which neither a constant-large nor a constant-small schedule can do. There is a real reason this is query-efficient, and tracing it tells me what the decay buys. Analyze single-pixel swaps on a binary-linear model as a coupon-collector chain. In the white-box limit the optimal $k$-sparse attack picks the $k$ coordinates with the smallest entries of an effective weight vector; the black-box obstacle is that this vector is hidden and reading it coordinate by coordinate costs $O(d)$ queries — hopeless at $d\sim1000$ on CIFAR. But I do not need the *exact* top $k$: a relaxed goal is to collect $k$ coordinates among the $m$ smallest, for some $m > k$. Model the support's progress toward that goal as a Markov chain on the count of "good" coordinates in the support; a single swap improves the count when it drops a bad coordinate and adds a good one, with a probability I can write down exactly, and the expected time to fill the support is a sum of geometric waiting times that works out to

$$ \mathbb{E}[t_k] < \frac{(d-k)\,k\,(\ln k + 2)}{m-k}, $$

which is *sublinear* in $d$ once $m-k$ grows with $d$ — beating the $O(d)$ cost of coordinate-wise weight estimation that sank any black-box gradient route. The relaxation from "exact top $k$" to "$k$ among the $m$ smallest" is the whole trick: it converts a prohibitive $O(d)$ identification problem into a $k\log k$-style hitting time, and the decaying schedule is exactly what runs the broad-support exploration early and approaches the one-pixel refinement regime whose analysis gives that bound. So the strongest rung is the one whose proposal distribution is *derived from* the $L0$ structure, not adapted to it from a continuous method.

`torchattacks` has no Sparse-RS, so the fill *inlines* the full attack, and the configuration is `n_queries = 10000`, `p_init = 0.8`, `eps = pixels = 24`. The proposal is exactly the swap above: maintain per image an index set `b` of the $24$ perturbed pixels and its complement `be`, map a pixel index $p$ to coordinates $(p\,/\!/\,W,\ p\bmod W)$ — matching the harness's channel-wise $L0$ count, a pixel changes if *any* channel differs — and swap $\text{eps\_it} = \max(\lfloor \text{p\_selection}(it)\cdot\text{eps}\rfloor, 1)$ entries between `b` and `be` each step, redrawing binary corner colors on the entrants. The schedule `p_selection` is the piecewise-constant decay with divisors $\{2,4,5,6,8,10,12,15,20\}$ on the reference intervals, rescaled to `n_queries`. The accept rule keeps a move if the loss improves *or* the margin is already negative (locking in a flip), and only spends queries on images whose current margin is still positive — so easy samples stop early and the budget concentrates on the stubborn ones. The single-pixel case resamples the entering color until it differs from the current one, so a one-pixel refinement step is never a wasted no-op query. This is $10000$ directed, *always-feasible* proposals per image with a schedule that provably explores-then-refines — a different order of search entirely from SparseFool's twenty relinearizations.

Sparse-RS attacks the *one* thing every prior rung got wrong — it is built natively for the discrete $L0$ support, with a feasible proposal, a provably query-efficient decaying schedule, and corner colors that use the budget maximally — and it gets two-to-three orders of magnitude more queries to do it ($10000$ versus $\sim15$–$120$). So I expect not a modest gain but a *step change*: from the high-teens of the best continuous method into the high-eighties-to-mid-nineties of percent across all three models, above $0.85$ on every one. The per-model spread should also *flatten* relative to SparseFool, because random search over feasible supports does not depend on the local linearity of any one model's boundary — the property that made SparseFool swing from $0.113$ to $0.280$. If it failed to dominate — if it landed merely in the SparseFool range — that would mean $10000$ queries are not enough on these robust models, and the only thing stronger would be a method that uses the gradient the harness permits to *guide* the support choice rather than searching it blind.

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
