**Problem (from step 5).** Sparse-RS is the strongest baseline (mean ASR ≈ 0.922, flat across models) but
uses *no gradient* despite full white-box access — leaving ~22 Rebuffi survivors whose rare 24-pixel
support blind random search cannot find in 10000 queries. Use the gradient to *choose the support*.

**Key idea (Sparse-PGD / sPGD).** Naive `L0`-PGD fails because the hard top-`k` projection lurches and
zeroes the gradient on unselected pixels. Instead decompose the perturbation as `delta = p ⊙ binarize(m)`:
a magnitude tensor `p` times the top-`k` binarization of a continuous mask `m` (one scalar per pixel). The
top-`k` enforces `||delta||_0 <= k` *by construction*; cross its non-differentiability with
**straight-through estimation** — hard top-`k` in the forward pass, *soft*-mask gradient in the backward
pass — so every pixel (selected or not) gets signal and the support can move. Restarts alternate routing
the perturbation gradient through the soft mask (explore) vs the hard mask (refine), which fall into
different local optima; keep the best.

**Why it should beat Sparse-RS.** It uses the gradient Sparse-RS ignores to steer the support directly, so
it should find the hard examples random search misses in budget — most of all on `Rebuffi-R18-L2`, the
model with the lowest baseline ASR (0.853). Native `[-x, 1-x]` clamps keep `x+delta in [0,1]` per pixel; a
margin objective gives the misclassification certificate without saturating.

**Scaffold edit / hyperparameters.** `eps=1` (full range under `L0`); `alpha=0.25*eps` (magnitude sign
step); `beta=0.25` mask step scaled by `sqrt(H*W)` (normalized — `m` is a selection variable); `t=300`
iterations × `n_restarts=2` alternating routings; mask `(B,1,H,W)` matching the harness's channel-collapsed
`L0` count; best `24`-sparse example kept across all iterations and restarts. The two routings
correspond to the explore (`MaskingA`) and refine (`MaskingB`) variants, and the clamp order is
`[-eps,eps]` then `[-x,1-x]`.

**Bar to clear (against step 5's real numbers).** Sparse-RS: `0.853 / 0.947 / 0.967`, mean `0.922`. sPGD
must beat the mean and, decisively, lift the per-model minimum above `0.853`. If it trails, the natural
remedy is an sPGD + Sparse-RS ensemble (best-of-both), how the strongest sparse-`L0` evaluations are built.

```python
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    pixels: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    """Sparse-PGD (sPGD) L0 white-box attack."""
    import torch
    import torch.nn.functional as F

    _ = (n_classes,)
    model.eval()

    eps = 1.0          # L0 lets a chosen pixel take any value in [0,1]
    t = 300            # iterations per restart
    n_restarts = 2     # alternate projected / unprojected gradient routing
    alpha = 0.25 * eps                       # magnitude sign-step size
    beta = 0.25                              # mask step (scaled by sqrt(H*W))

    x = images.detach().clone().to(device)
    y = labels.detach().clone().to(device)
    B, C, H, W = x.shape
    k = int(pixels)
    mask_step = beta * (H * W) ** 0.5

    class _TopKMask(torch.autograd.Function):
        """Forward: hard top-k binarization (exactly k pixels=1) -> L0 budget by
        construction. Backward: straight-through -- the mask gradient flows through the
        SOFT mask, so unselected pixels still get signal and the support can change.
        `unprojected=True` routes the perturbation gradient through the soft mask
        (MaskingA, explore); `False` through the hard projection (MaskingB, refine)."""

        @staticmethod
        def forward(ctx, x_perturb, mask, kk, unprojected):
            b = mask.shape[0]
            flat = mask.view(b, -1)
            idx = flat.argsort(dim=1, descending=True)
            hard = torch.zeros_like(flat).scatter_(1, idx[:, :kk], 1.0).view_as(mask)
            ctx.save_for_backward(x_perturb, mask if unprojected else hard)
            return x_perturb * hard, hard

        @staticmethod
        def backward(ctx, grad_out, _grad_mask):
            x_perturb, routed = ctx.saved_tensors
            grad_perturb = grad_out * routed
            grad_mask = (grad_out * x_perturb).sum(dim=1, keepdim=True)
            return grad_perturb, grad_mask, None, None

    def margin_loss(logits, yb):
        u = torch.arange(logits.shape[0], device=logits.device)
        correct = logits[u, yb].clone()
        tmp = logits.clone()
        tmp[u, yb] = -float("inf")
        other = tmp.max(dim=1)[0]
        return correct - other            # >0 correct; minimized

    x_adv_best = x.clone()
    loss_best = torch.full((B,), float("inf"), device=device)

    for r in range(n_restarts):
        unprojected = (r % 2 == 0)
        perturb = x.new(x.size()).uniform_(-eps, eps)
        perturb = torch.min(torch.max(perturb, -x), 1 - x)       # x+perturb in [0,1]
        mask = torch.randn(B, 1, H, W, device=device)            # one scalar per spatial pixel

        for it in range(t):
            perturb = perturb.detach().requires_grad_(True)
            mask = mask.detach().requires_grad_(True)
            proj_perturb, _hard = _TopKMask.apply(perturb, torch.sigmoid(mask), k, unprojected)
            logits = model(x + proj_perturb)
            loss = margin_loss(logits, y)
            loss.sum().backward()
            g_perturb = perturb.grad.detach()
            g_mask = mask.grad.detach()

            with torch.no_grad():
                cur = loss.detach()
                improve = cur < loss_best
                if improve.any():
                    loss_best[improve] = cur[improve]
                    x_adv_best[improve] = (x + proj_perturb)[improve].detach()

            # magnitude: PGD sign step (descent on margin) + native validity clamps
            perturb = perturb.detach() - alpha * g_perturb.sign()
            perturb = perturb.clamp(-eps, eps)
            perturb = torch.min(torch.max(perturb, -x), 1 - x)

            # mask: normalized gradient step (selection variable, not bounded perturbation)
            gnorm = g_mask.flatten(1).norm(dim=1).clamp_min(1e-10)
            mask = mask.detach() - mask_step * (g_mask / gnorm.view(B, 1, 1, 1))

    return x_adv_best.detach()
```
