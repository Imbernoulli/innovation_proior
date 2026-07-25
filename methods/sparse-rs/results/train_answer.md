I solve the query-efficient black-box sparse adversarial attack problem by treating the L0 budget as a combinatorial support rather than a norm radius. The method I propose is Sparse-RS, a score-only random-search attack originally introduced by Croce, Andriushchenko, and Hein for crafting adversarial examples under a strict L0 pixel budget. My goal is to change at most k spatial pixels of a correctly classified image so that the classifier outputs a different label, using only the model's score vector and as few forward queries as possible. The key observation is that the L0 feasible set is the union of all k-dimensional coordinate subspaces indexed by the chosen support; projection onto this set is a discrete top-k selection, which makes continuous gradient methods unstable and black-box gradient estimation prohibitively expensive at ImageNet scale. Random search avoids both problems because it only needs feasible candidates and scalar comparisons.

A sparse perturbation naturally decomposes into two parts: a support M of perturbed spatial pixels and the color values assigned to those pixels. The budget constrains only the cardinality of M, not the magnitude of the changes. Once a pixel is selected, its channels may be set to any value in [0,1]^c without additional L0 cost. I therefore initialize the attack by sampling k spatial pixels uniformly at random and setting each selected pixel to a random corner of the color cube {0,1}^c. Corners use each precious pixel maximally, because intermediate colors consume the same support slot while exerting less influence on the model.

The search maintains a current best candidate x_best and a margin loss L(z) = f_y(z) - max_{r != y} f_r(z). The sign of this margin is the misclassification certificate: L(z) < 0 means the candidate is already adversarial. At each iteration I propose a new candidate by swapping equal-size sets between the current perturbed support M and the clean complement, restoring the leaving pixels to their original values and drawing fresh corner colors for the entering pixels. This preserves |M| = k exactly, so every query respects the budget by construction and no projection is needed.

The swap size must evolve over the run. Large swaps early escape a poor random initial support quickly; small swaps late refine a good support without destroying it. I use a piecewise-constant decaying schedule driven by a single parameter p_init. After rescaling the iteration to a reference budget of 10000 queries, the swap fraction is divided by 2, 4, 5, 6, 8, 10, 12, 15, and 20 on successive intervals. The number of swapped pixels is max(int(alpha_i * k), 1), so the proposal always changes at least one pixel.

For the one-pixel refinement case, I resample the entering color until it differs from the current value, preventing wasted no-op queries. The accept rule keeps a candidate either when the margin loss improves or when the margin is already negative, which locks in an adversarial example. Queries are spent only on images whose current margin is still positive, so easy examples stop early and the budget concentrates on stubborn ones.

The query efficiency of this scheme can be understood through a binary-linear coupon-collector analysis. Consider a binary input x in {0,1}^d and a linear model with gradient w_x. Folding in the label and current value gives an effective weight vector w_hat_x = y * w_x ⊙ (1 - 2x). The optimal white-box k-sparse attack picks the k smallest entries of w_hat_x. In the black-box setting I cannot read w_hat_x directly; coordinate-wise estimation costs O(d) queries, which is impractical for d around 150000. I therefore relax exact recovery to finding k coordinates among the m smallest for some m > k. Modeling single-pixel swaps as a Markov chain on the number of good coordinates currently in the support gives an expected hitting time E[t_k] < (d - k) k (ln k + 2) / (m - k). When m - k grows with d, this bound is sublinear in the input dimension and beats the O(d) cost of black-box gradient estimation. The relaxation from exact top-k to k among the m smallest is what converts a prohibitive identification problem into a k log k-style hitting time. Real networks are piecewise-linear, so the same broad-then-fine behavior remains effective when larger swaps are used early and single-pixel swaps dominate late.

The function below, `run_attack`, is the complete implementation. It takes the model, a batch of images, their labels, the pixel budget, and the device; it also accepts a class-count argument, kept only for interface parity with the other attack branches and unused here. It runs exactly the loop just described — initialize the support and corner colors, then propose, query, and accept — for up to 10000 queries, restricting every query to the images that are not yet fooled, and returns the best adversarial batch it found.

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

This implementation keeps the entire L0 constraint in spatial-pixel space, uses binary corner colors to maximize the effect of each perturbed pixel, follows the decaying swap schedule from the Sparse-RS reference code, spends queries only on examples that are not yet misclassified, and preserves the accept rule that locks in any candidate whose margin is already negative. Every candidate is feasible by construction, so the method scales from small CIFAR-10 images up to high-resolution ImageNet inputs without ever leaving the L0 budget.
