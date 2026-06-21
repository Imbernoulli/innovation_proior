I solve the query-efficient black-box sparse adversarial attack problem by treating the L0 budget as a combinatorial support rather than a norm radius. The method I propose is Sparse-RS, a score-only random-search attack originally introduced by Croce, Andriushchenko, and Hein for crafting adversarial examples under a strict L0 pixel budget. My goal is to change at most k spatial pixels of a correctly classified image so that the classifier outputs a different label, using only the model's score vector and as few forward queries as possible. The key observation is that the L0 feasible set is the union of all k-dimensional coordinate subspaces indexed by the chosen support; projection onto this set is a discrete top-k selection, which makes continuous gradient methods unstable and black-box gradient estimation prohibitively expensive at ImageNet scale. Random search avoids both problems because it only needs feasible candidates and scalar comparisons.

A sparse perturbation naturally decomposes into two parts: a support M of perturbed spatial pixels and the color values assigned to those pixels. The budget constrains only the cardinality of M, not the magnitude of the changes. Once a pixel is selected, its channels may be set to any value in [0,1]^c without additional L0 cost. I therefore initialize the attack by sampling k spatial pixels uniformly at random and setting each selected pixel to a random corner of the color cube {0,1}^c. Corners use each precious pixel maximally, because intermediate colors consume the same support slot while exerting less influence on the model.

The search maintains a current best candidate x_best and a margin loss L(z) = f_y(z) - max_{r != y} f_r(z). The sign of this margin is the misclassification certificate: L(z) < 0 means the candidate is already adversarial. At each iteration I propose a new candidate by swapping equal-size sets between the current perturbed support M and the clean complement, restoring the leaving pixels to their original values and drawing fresh corner colors for the entering pixels. This preserves |M| = k exactly, so every query respects the budget by construction and no projection is needed.

The swap size must evolve over the run. Large swaps early escape a poor random initial support quickly; small swaps late refine a good support without destroying it. I use a piecewise-constant decaying schedule driven by a single parameter p_init. After rescaling the iteration to a reference budget of 10000 queries, the swap fraction is divided by 2, 4, 5, 6, 8, 10, 12, 15, and 20 on successive intervals. The number of swapped pixels is max(int(alpha_i * k), 1), so the proposal always changes at least one pixel.

For the one-pixel refinement case, I resample the entering color until it differs from the current value, preventing wasted no-op queries. The accept rule keeps a candidate either when the margin loss improves or when the margin is already negative, which locks in an adversarial example. Queries are spent only on images whose current margin is still positive, so easy examples stop early and the budget concentrates on stubborn ones.

The query efficiency of this scheme can be understood through a binary-linear coupon-collector analysis. Consider a binary input x in {0,1}^d and a linear model with gradient w_x. Folding in the label and current value gives an effective weight vector w_hat_x = y * w_x ⊙ (1 - 2x). The optimal white-box k-sparse attack picks the k smallest entries of w_hat_x. In the black-box setting I cannot read w_hat_x directly; coordinate-wise estimation costs O(d) queries, which is impractical for d around 150000. I therefore relax exact recovery to finding k coordinates among the m smallest for some m > k. Modeling single-pixel swaps as a Markov chain on the number of good coordinates currently in the support gives an expected hitting time E[t_k] < (d - k) k (ln k + 2) / (m - k). When m - k grows with d, this bound is sublinear in the input dimension and beats the O(d) cost of black-box gradient estimation. The relaxation from exact top-k to k among the m smallest is what converts a prohibitive identification problem into a k log k-style hitting time. Real networks are piecewise-linear, so the same broad-then-fine behavior remains effective when larger swaps are used early and single-pixel swaps dominate late.

The following Python script is a complete runnable illustration of the Sparse-RS L0 attack. It includes a small synthetic model, generates images that are initially correctly classified, runs the attack, and verifies that the final perturbation respects the L0 budget and flips the predicted class.

```python
import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """A small linear classifier used only to demonstrate the attack."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.n_classes = n_classes
        self.fc = nn.Linear(3 * 32 * 32, n_classes, bias=True)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def sparse_rs_l0_attack(model, images, labels, pixels, n_queries=10000, p_init=0.8):
    """Sparse-RS L0 black-box attack (Croce et al., AAAI 2022)."""
    model.eval()
    eps = int(pixels)
    x = images.detach().clone()
    y = labels.detach().clone()
    B, C, H, W = x.shape
    n_pixels = H * W
    device = x.device

    def margin_and_loss(xb, yb):
        with torch.no_grad():
            logits = model(xb)
        u = torch.arange(xb.shape[0], device=device)
        y_corr = logits[u, yb].clone()
        logits[u, yb] = -float("inf")
        y_others = logits.max(dim=-1)[0]
        margin = y_corr - y_others
        return margin, margin

    def p_selection(it):
        it = int(it / n_queries * 10000)
        if 0 < it <= 50:
            return p_init / 2
        elif 50 < it <= 200:
            return p_init / 4
        elif 200 < it <= 500:
            return p_init / 5
        elif 500 < it <= 1000:
            return p_init / 6
        elif 1000 < it <= 2000:
            return p_init / 8
        elif 2000 < it <= 4000:
            return p_init / 10
        elif 4000 < it <= 6000:
            return p_init / 12
        elif 6000 < it <= 8000:
            return p_init / 15
        elif 8000 < it:
            return p_init / 20
        return p_init

    def rand_colors(shape):
        return torch.randint(0, 2, shape, device=device, dtype=x.dtype)

    x_best = x.clone()
    b_all = torch.zeros(B, eps, dtype=torch.long, device=device)
    be_all = torch.zeros(B, n_pixels - eps, dtype=torch.long, device=device)
    for i in range(B):
        perm = torch.randperm(n_pixels, device=device)
        ind_p, ind_np = perm[:eps], perm[eps:]
        x_best[i, :, ind_p // W, ind_p % W] = rand_colors((C, eps)).clamp(0.0, 1.0)
        b_all[i] = ind_p
        be_all[i] = ind_np

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

        for i in range(x_new.shape[0]):
            p_set = b_curr[i, ind_p]
            np_set = be_curr[i, ind_np]
            x_new[i, :, p_set // W, p_set % W] = x_curr[i, :, p_set // W, p_set % W]
            if eps_it > 1:
                x_new[i, :, np_set // W, np_set % W] = rand_colors((C, eps_it)).clamp(0.0, 1.0)
            else:
                old = x_new[i, :, np_set // W, np_set % W].clone()
                new = old.clone()
                while (new == old).all():
                    new = rand_colors((C, 1)).clamp(0.0, 1.0)
                x_new[i, :, np_set // W, np_set % W] = new

        margin, loss = margin_and_loss(x_new, y_curr)
        idx_improved = loss < loss_min_curr
        idx_miscl = margin < -1e-6
        idx_keep = idx_improved | idx_miscl

        if idx_improved.any():
            loss_min[idx_to_fool[idx_improved]] = loss[idx_improved]
        if idx_keep.any():
            margin_min[idx_to_fool[idx_keep]] = margin[idx_keep]
            x_best[idx_to_fool[idx_keep]] = x_new[idx_keep]
            t = b_curr[idx_keep].clone()
            te = be_curr[idx_keep].clone()
            t[:, ind_p] = be_curr[idx_keep][:, ind_np]
            te[:, ind_np] = b_curr[idx_keep][:, ind_p]
            b_all[idx_to_fool[idx_keep]] = t
            be_all[idx_to_fool[idx_keep]] = te

    return x_best.detach()


if __name__ == "__main__":
    torch.manual_seed(7)
    device = torch.device("cpu")
    n_classes = 10
    model = LinearModel(n_classes=n_classes).to(device).eval()

    # Generate images and assign each one a label that the model already predicts.
    B = 4
    images = torch.rand(B, 3, 32, 32, device=device)
    with torch.no_grad():
        init_logits = model(images)
    labels = init_logits.argmax(dim=1)
    print("initial predictions:", labels.tolist())
    print("true labels:       ", labels.tolist())

    adversarial = sparse_rs_l0_attack(model, images, labels, pixels=24, n_queries=5000)

    with torch.no_grad():
        final_preds = model(adversarial).argmax(dim=1)
    print("final predictions: ", final_preds.tolist())

    changed = (adversarial != images).any(dim=1).float()
    l0_per_image = changed.view(changed.size(0), -1).sum(dim=1)
    print("L0 changed pixels per image:", l0_per_image.int().tolist())
    print("budget respected:", (l0_per_image <= 24).all().item())
    print("success rate:", (final_preds != labels).float().mean().item())
```

This implementation keeps the entire L0 constraint in spatial-pixel space, uses binary corner colors to maximize the effect of each perturbed pixel, follows the decaying swap schedule from the Sparse-RS reference code, spends queries only on examples that are not yet misclassified, and preserves the accept rule that locks in any candidate whose margin is already negative. Every candidate is feasible by construction, so the method scales from small CIFAR-10 images up to high-resolution ImageNet inputs without ever leaving the L0 budget.
