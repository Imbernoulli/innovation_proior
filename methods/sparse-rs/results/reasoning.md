I start from the constraint rather than from an optimizer. I have an image `x`, a correct label `y`, and a budget of `k` spatial pixels. The budget does not say that every coordinate can move a little; it says that only a selected support can move at all. Once a pixel is in that support, its color channels can move anywhere in `[0,1]`. So the object I need to optimize is not a dense vector in a norm ball. It is a pair: a set `M` of selected pixels and a set of values placed on those pixels.

This immediately makes the usual gradient picture look wrong. In an `L_inf` or `L_2` attack, I can take a step and project back into a continuous ball. In the `L0` case, projection chooses a support. That choice is discrete, and a small continuous change before projection can produce a large support change after projection. A gradient can also spread useful signal across many pixels even though the final attack keeps only `k` of them. In a white-box setting this already makes the iteration unstable; in a score-only setting I also have to pay many queries to estimate any gradient at all. At ImageNet scale, reading a full direction has cost proportional to `d = 224*224*3`, so I need a different primitive.

Random search gives me the right primitive because it only asks for feasible candidates and scalar comparisons. I can keep the best candidate so far, sample a new legal candidate, query the loss, and keep the new one when it improves. The hard part is not the accept-if-improve loop. The hard part is the distribution over legal sparse candidates.

I split the candidate into support and color. I initialize `M` by sampling `k` pixels uniformly from the `h*w` spatial positions because I have no support information yet. For the values on `M`, I use corners of the color cube, `{0,1}^c`. This follows from the budget: after I spend one of the `k` pixels, there is no extra penalty for a large color change. A gray or intermediate color uses the same support slot while using less of the available box. So I set selected pixels to binary corners.

For the scalar objective, I use the untargeted margin `L(z) = f_y(z) - max_{r != y} f_r(z)`. I minimize it. Its sign gives the stopping condition because `L(z) < 0` is exactly misclassification. For targeted attacks, the corresponding target-class cross-entropy takes over, but this branch is untargeted and margin-based.

Now I need to move through supports without ever leaving the budget. The natural move is a swap. At iteration `i`, I select `A` from the current perturbed support `M` and `B` from the clean complement `U \ M`, with `|A| = |B|`. Then I form `M' = (M \ A) union B`. Pixels in `A` return to their original clean values. Pixels in `B` receive fresh binary corner colors. The support size remains exactly `k`, so every query is feasible by construction.

The swap size has to change over time. If I swap many pixels early, I can escape a poor random initial support quickly. If I keep swapping many pixels late, I destroy useful pixels that I already found. If I swap one pixel from the start, I refine gently but spend too many queries finding the right region. So I make the swap fraction decay: large at the beginning for exploration, smaller later for refinement.

The implementation uses one parameter `p_init` and a piecewise constant schedule. The loop starts at iteration `1`; after rescaling to a reference budget of `10000`, the `L0` branch uses divisors `2, 4, 5, 6, 8, 10, 12, 15, 20` on the intervals `(0,50]`, `(50,200]`, `(200,500]`, `(500,1000]`, `(1000,2000]`, `(2000,4000]`, `(4000,6000]`, `(6000,8000]`, and `(8000,N]`. The code computes the number of swapped pixels as `max(int(alpha_i*k), 1)`, so it floors the scheduled product and never allows a zero-swap proposal.

I also need to avoid wasting a query in the one-pixel case. If the entering pixel receives exactly the same color it already has in the candidate, the proposal can be a no-op at that location. The implementation resamples the one-pixel color until it differs from the current value. For larger swaps, the probability of redrawing the whole same update is negligible, so the special guard is only needed at single-pixel refinement.

The accept rule has two pieces because the code tracks both the optimization loss and the margin certificate. It updates `loss_min` when the new loss improves. It updates the best image and swaps the support bookkeeping when either the loss improves or the new margin is already negative. In the untargeted margin branch these usually coincide for still-unfooled images, but the explicit misclassification branch preserves the success certificate logic used across attack variants.

I now check whether this simple support swapping can be query efficient rather than merely plausible. I analyze the binary-linear case. Let `x in {0,1}^d`, `y in {-1,1}`, and let the linear model have gradient `w_x`. If I encode a coordinate flip by `delta_i in {0,1}` and fold in the label and current binary value, the objective becomes minimizing `<w_hat_x, delta>` under `||delta||_0 <= k`, where `w_hat_x = y*w_x ⊙ (1 - 2x)`. In white-box form, the solution is to choose the `k` smallest entries of `w_hat_x`.

The score-only version cannot inspect `w_hat_x`; naive coordinate estimation costs `O(d)` queries. I relax the goal: I only need to collect `k` coordinates among the `m` smallest, with `m > k`. I set the swap fraction to `1/k`, so each step drops one coordinate `p` from `M` and adds one coordinate `q` from `U \ M`. The score comparison tells me whether the swap improves the linear objective, which means whether the entering coordinate is better than the leaving one.

I track `i`, the number of current support coordinates that already lie among the `m` smallest weights. To increase `i` to `i+1`, I must drop a coordinate outside the smallest `m` and add a coordinate inside the smallest `m` that is not already in `M`. The current support has `k-i` bad coordinates among its `k` elements, so the drop probability is `(k-i)/k`. The complement has `d-k` coordinates, and `m-i` of the good coordinates remain outside `M`, so the add probability is `(m-i)/(d-k)`. The transition probability is therefore `(k-i)/k * (m-i)/(d-k)`.

The waiting time for that transition is geometric with mean `(d-k)k/((k-i)(m-i))`. Summing from `i=0` to `k-1` gives `E[t_k] = (d-k)k * sum_{i=0}^{k-1} 1/((k-i)(m-i))`. I reindex with `j = k-i`, so `m-i = j + (m-k)`, and the sum becomes `sum_{j=1}^k 1/(j(j+m-k))`.

For `m > k`, I write `a = m-k` and use `1/(j(j+a)) = (1/a)(1/j - 1/(j+a))`. The sum is `(1/(m-k)) * sum_{j=1}^k (1/j - 1/(j+m-k))`. The first term is `H_k`. The second term is the harmonic tail from `m-k+1` through `m`, which is `H_m - H_{m-k}`. So the exact sum is `(1/(m-k))(H_k - H_m + H_{m-k})`.

I bound that bracket with the standard harmonic inequalities. Since `H_k <= ln k + 1`, `H_m > ln m`, and `H_{m-k} <= ln(m-k) + 1`, I get `H_k - H_m + H_{m-k} < ln k - ln m + ln(m-k) + 2`. The middle two terms combine to `ln((m-k)/m)`, which is negative. Therefore the bracket is less than `ln k + 2`, and `E[t_k] < (d-k)k(ln k + 2)/(m-k)`.

This is the key reason the method can beat coordinate-wise weight estimation. If the acceptable gap `m-k` grows with `d`, the ratio `(d-k)/(m-k)` can stay bounded, and the expected query count scales like `k log k` rather than `d`. If `m = k`, the partial fraction step is invalid; the sum becomes `sum_{j=1}^k 1/j^2`, so `E[t_k] < (pi^2/6)(d-k)k`, which is linear in `d`. The sublinear gain comes from accepting an approximate set of good coordinates instead of identifying the exact top `k`.

The proof only covers one-pixel swaps in a binary-linear model. For real neural networks I keep the same late-stage refinement behavior but use larger swaps early. That is exactly what the decaying schedule does: it starts with broad support exploration, then approaches the one-pixel regime whose coupon-collector analysis explains the query efficiency.

So let me write the thing I would actually run, filling the one empty slot in the random-search harness — the proposal, the initialization, and the keep-if-better update. I keep it batched, because I want to attack many images at once and only spend queries on the ones not yet fooled. Per image I track the index set of perturbed pixels (`b`: the `k` perturbed indices) and its complement (`be`: the `n-k` clean indices), so a swap is just exchanging entries between those two lists; a pixel index `p` maps to spatial coordinates `(p // W, p % W)`, and a pixel counts as modified if any channel differs, matching the benchmark's `L0` over channels. Colors are binary corners. The accept rule keeps a move if the loss improves and also locks in any already-misclassified candidate (`margin < 0`).

```python
def run_attack(model, images, labels, pixels, device, n_classes):
    """L0 random-search attack: maintain a size-`pixels` mask of perturbed spatial
    pixels with binary {0,1} corner colors; each step swap a decaying fraction of
    perturbed<->clean pixels and redraw colors, keep the move iff the margin loss
    improves (or the candidate is already misclassified). Margin loss is the only
    oracle; every candidate respects the L0 budget and stays in [0,1]."""
    import torch

    _ = (n_classes,)
    model.eval()

    n_queries = 10000        # query budget / reference schedule length
    p_init = 0.8             # the single knob driving the swap schedule
    eps = int(pixels)        # L0 budget = number of perturbed spatial pixels

    x = images.detach().clone().to(device)
    y = labels.detach().clone().to(device)
    B, C, H, W = x.shape
    n_pixels = H * W

    def margin_and_loss(xb, yb):
        # f_y - max_{r!=y} f_r ; < 0  <=>  misclassified. One forward pass = one query.
        with torch.no_grad():
            logits = model(xb)
        u = torch.arange(xb.shape[0], device=xb.device)
        y_corr = logits[u, yb].clone()
        logits[u, yb] = -float("inf")
        y_others = logits.max(dim=-1)[0]
        margin = y_corr - y_others
        return margin, margin                            # margin variant: loss == margin

    def random_corners(shape):
        # Corners of the color cube {0,1}: use the budget maximally per perturbed pixel.
        return torch.randint(0, 2, shape, device=device, dtype=x.dtype)

    def p_selection(it):
        # Piecewise-constant DECAYING schedule, large->small: explore then refine.
        it = int(it / n_queries * 10000)                 # rescale to the reference N=10000
        if   0    < it <= 50:   return p_init / 2
        elif 50   < it <= 200:  return p_init / 4
        elif 200  < it <= 500:  return p_init / 5
        elif 500  < it <= 1000: return p_init / 6
        elif 1000 < it <= 2000: return p_init / 8
        elif 2000 < it <= 4000: return p_init / 10
        elif 4000 < it <= 6000: return p_init / 12
        elif 6000 < it <= 8000: return p_init / 15
        elif 8000 < it:         return p_init / 20
        return p_init

    # ---- Initialize: k uniformly-random pixels per image, random corner colors. ----
    x_best = x.clone()
    b_all  = torch.zeros(B, eps, dtype=torch.long, device=device)            # perturbed idxs (M)
    be_all = torch.zeros(B, n_pixels - eps, dtype=torch.long, device=device) # clean idxs (U\M)
    for i in range(B):
        perm = torch.randperm(n_pixels, device=device)
        ind_p, ind_np = perm[:eps], perm[eps:]
        x_best[i, :, ind_p // W, ind_p % W] = random_corners((C, eps)).clamp(0.0, 1.0)
        b_all[i], be_all[i] = ind_p, ind_np

    margin_min, loss_min = margin_and_loss(x_best, y)

    for it in range(1, n_queries):
        idx = (margin_min > 0.0).nonzero().squeeze(-1)   # only spend queries on not-yet-fooled
        if idx.numel() == 0:
            break

        x_curr        = x[idx].clone()                   # clean originals (for restoring)
        y_curr        = y[idx]
        loss_min_curr = loss_min[idx]
        b_curr, be_curr = b_all[idx].clone(), be_all[idx].clone()

        x_new = x_best[idx].clone()
        eps_it = max(int(p_selection(it) * eps), 1)      # |A| = |B|, floored, at least one
        ind_p  = torch.randperm(eps, device=device)[:eps_it]              # A: drop from M
        ind_np = torch.randperm(n_pixels - eps, device=device)[:eps_it]   # B: add to M

        for i in range(x_new.shape[0]):
            p_set  = b_curr[i, ind_p]                     # pixels leaving M -> restore to clean
            np_set = be_curr[i, ind_np]                   # pixels entering M -> fresh corners
            x_new[i, :, p_set // W, p_set % W] = x_curr[i, :, p_set // W, p_set % W]
            if eps_it > 1:
                x_new[i, :, np_set // W, np_set % W] = random_corners((C, eps_it)).clamp(0.0, 1.0)
            else:
                # single-pixel refinement: ensure the new color differs from the current one
                old = x_new[i, :, np_set // W, np_set % W].clone()
                new = old.clone()
                while (new == old).all():
                    new = random_corners((C, 1)).clamp(0.0, 1.0)
                x_new[i, :, np_set // W, np_set % W] = new

        margin, loss = margin_and_loss(x_new, y_curr)

        improved = loss < loss_min_curr
        upd_loss = improved.nonzero().squeeze(-1)        # track loss improvement separately
        if upd_loss.numel() > 0:
            loss_min[idx[upd_loss]] = loss[upd_loss]

        keep = improved | (margin < -1e-6)               # accept: improve OR already fooled
        upd = keep.nonzero().squeeze(-1)
        if upd.numel() > 0:
            margin_min[idx[upd]] = margin[upd]
            x_best[idx[upd]] = x_new[upd]
            # Commit the support swap: A and B trade places between M and U\M.
            t  = b_curr[upd].clone()
            te = be_curr[upd].clone()
            t[:, ind_p]   = be_curr[upd][:, ind_np]
            te[:, ind_np] = b_curr[upd][:, ind_p]
            b_all[idx[upd]], be_all[idx[upd]] = t, te

    return x_best.detach()
```

Retrace the causal chain. The `L0` budget is a combinatorial support, not a norm radius, so continuous gradient methods stall and mis-estimate robustness on it, and black-box gradient estimation costs `O(d)` queries, hopeless at `d ~ 150k`. Random search needs only feasible candidates and one score query per step, so it sidesteps both walls — but it is only as good as its sampling distribution. The perturbation splits into a support `M` of size `k` and colors, so the proposal is "swap an equal number of pixels in and out of `M`" (always feasible) plus "redraw colors." Colors should be corners of the cube because the model penalizes magnitude not at all, so extremal values use each precious pixel maximally. The objective is the margin, whose sign is the misclassification certificate. The swap size must be large early to explore which pixels matter from a random start, and shrink to a single pixel late to refine without destroying a good support — a constant schedule fails one way or the other — so I decay it on a piecewise-constant schedule driven by one knob. Analyzing single-pixel swaps on a binary-linear model as a coupon-collector chain gives `E[t_k] < (d-k)k(ln k + 2)/(m-k)`, sublinear in `d` once I accept "`k` among the `m` smallest" rather than the exact `k` — exactly the relaxation that lets me dodge the `O(d)` estimation cost. The whole thing drops into the generic accept-if-improve loop as an initialization, a decaying two-part proposal over index sets, and a keep-if-better update — no gradients, every query feasible.
