# Sparse-PGD (sPGD)

Sparse-PGD is a white-box fixed-budget sparse adversarial attack. It represents a perturbation as

```text
delta = p * m
```

where `p` is a dense magnitude tensor and `m` is a hard top-`k` spatial mask. The forward pass is always feasible: `m` selects at most `k` pixels, and `p` is clamped so `0 <= x + p <= 1`. The backward pass replaces the nondifferentiable top-`k` derivative with a straight-through surrogate so the continuous mask logits receive gradients even for pixels that are not currently selected.

## Core Algorithm

Use a continuous mask logit tensor `u` and `s = sigmoid(u)`. Project `s` to a hard mask by keeping the `k` largest spatial entries:

```text
m = Pi_k(sigmoid(u))
delta = p * m
```

For an untargeted attack, maximize an attack loss `L(theta, x + delta, y)`. In the paper-level sparse-AutoAttack description this loss is cross-entropy. In the reference repository, `loss_fn` uses cross-entropy for high-confidence samples and a denominator-free DLR/margin fallback `max_{r != y} f_r - f_y` for low-confidence samples. In both cases the loss is maximized.

Magnitude update:

```text
projected gradient:    g_p = (dL / ddelta) * m
unprojected gradient:  g_p = (dL / ddelta) * sigmoid(u)
p <- Pi_Sp(p + alpha * sign(g_p))
```

The projection `Pi_Sp` first clamps to `[-eps_inf, eps_inf]` and then to `[-x, 1 - x]`.

Mask-logit update:

```text
g_u ~= sum_channels((dL / ddelta) * p) * sigmoid(u) * (1 - sigmoid(u))
u <- u + beta * sqrt(H * W) * g_u / ||g_u||_2
m <- Pi_k(sigmoid(u))
```

If the mask-gradient norm is effectively zero, skip the mask step. If a still-correct sample's projected mask is unchanged for `patience` consecutive iterations, randomly reinitialize that sample's mask logits.

## Constants And Cases

- `k`: spatial `L0` budget.
- `eps_inf = 1` (`255/255`): selected pixels may move across the full image range.
- `alpha = 0.25 * eps_inf`: magnitude sign-step size for unstructured attacks.
- `beta = 0.25`: code parameter; the actual mask-logit step is `0.25 * sqrt(H * W)`.
- `patience = 3`: unstructured attack mask-reinitialization tolerance.
- Paper small-gradient cutoff: `gamma = 2e-8`; cloned reference code zeros the mask step at `grad_norm < 2e-10` and divides by `grad_norm + 1e-10`.
- `sPGD_unproj` / `MaskingA`: forward uses the hard mask, but `p`'s backward route uses `sigmoid(u)` for denser exploration.
- `sPGD_proj` / `MaskingB`: forward and `p`'s backward route both use the hard mask for exploitation.
- Sparse-AutoAttack runs `sPGD_unproj`, then `sPGD_proj`, then Sparse-RS in cascade, skipping later attacks for samples already broken.

## Structured Extension

For structured sparse perturbations, replace the pixel mask by a group mask `v`. A projected group mask chooses rows, columns, patches, or custom patterns. A transposed convolution with a binary pattern kernel maps selected groups to a pixel mask, clipped to one. The backward pass maps pixel-mask gradients back to group logits by convolution and ignores the nondifferentiable clipping/projection. With a `1 x 1` kernel, the structured version reduces to ordinary unstructured spatial `L0`.

## Reference-Faithful Core

```python
import torch
import torch.nn.functional as F


class MaskingA(torch.autograd.Function):
    """Unprojected p-gradient: hard top-k forward, soft-mask backward for p."""

    @staticmethod
    def forward(ctx, perturb, soft_mask, k):
        b = soft_mask.shape[0]
        flat = soft_mask.view(b, -1)
        _, idx = torch.sort(flat, dim=1, descending=True)
        hard = torch.zeros_like(flat).scatter_(1, idx[:, :k], 1.0).view_as(soft_mask)
        ctx.save_for_backward(perturb, soft_mask)
        return perturb * hard, hard

    @staticmethod
    def backward(ctx, grad_output, _grad_hard):
        perturb, soft_mask = ctx.saved_tensors
        grad_perturb = grad_output * soft_mask
        grad_soft_mask = (grad_output * perturb).sum(dim=1, keepdim=True)
        return grad_perturb, grad_soft_mask, None


class MaskingB(torch.autograd.Function):
    """Projected p-gradient: hard top-k forward, hard-mask backward for p."""

    @staticmethod
    def forward(ctx, perturb, soft_mask, k):
        b = soft_mask.shape[0]
        flat = soft_mask.view(b, -1)
        _, idx = torch.sort(flat, dim=1, descending=True)
        hard = torch.zeros_like(flat).scatter_(1, idx[:, :k], 1.0).view_as(soft_mask)
        ctx.save_for_backward(perturb, hard)
        return perturb * hard, hard

    @staticmethod
    def backward(ctx, grad_output, _grad_hard):
        perturb, hard = ctx.saved_tensors
        grad_perturb = grad_output * hard
        grad_soft_mask = (grad_output * perturb).sum(dim=1, keepdim=True)
        return grad_perturb, grad_soft_mask, None


def ce_with_low_conf_margin(logits, y, threshold=0.5):
    b = logits.shape[0]
    u = torch.arange(b, device=logits.device)
    correct = logits[u, y]
    other_logits = logits.clone()
    other_logits[u, y] = -float("inf")
    other = other_logits.max(dim=1).values
    low_conf = (correct - other) < threshold

    loss = F.cross_entropy(logits, y, reduction="none")
    loss[low_conf] = other[low_conf] - correct[low_conf]
    return loss


def project_mask(mask_logits, k):
    soft = torch.sigmoid(mask_logits)
    b = soft.shape[0]
    flat = soft.view(b, -1)
    _, idx = torch.sort(flat, dim=1, descending=True)
    return torch.zeros_like(flat).scatter_(1, idx[:, :k], 1.0).view_as(soft)


def sparse_pgd_core(model, images, labels, k, steps=10000, eps_inf=1.0, alpha=0.25,
                    beta=0.25, patience=3, unprojected_gradient=True,
                    zero_grad_threshold=2e-10):
    model.eval()
    x = images.detach()
    y = labels.detach()
    b, c, h, w = x.shape
    alpha_step = alpha * eps_inf
    mask_step = beta * (h * w) ** 0.5
    masking = MaskingA.apply if unprojected_gradient else MaskingB.apply

    perturb = x.new_empty(x.shape).uniform_(-eps_inf, eps_inf)
    perturb = torch.min(torch.max(perturb, -x), 1 - x)
    mask_logits = torch.randn(b, 1, h, w, device=x.device, dtype=x.dtype)
    unchanged = torch.zeros(b, dtype=torch.long, device=x.device)

    best = x.clone()
    best_loss = torch.full((b,), -float("inf"), device=x.device)

    for _ in range(steps):
        prev_hard = project_mask(mask_logits.detach(), k)
        perturb = perturb.detach().requires_grad_(True)
        mask_logits = mask_logits.detach().requires_grad_(True)

        proj_perturb, _hard = masking(perturb, torch.sigmoid(mask_logits), k)
        logits = model(x + proj_perturb)
        loss = ce_with_low_conf_margin(logits, y)
        loss.sum().backward()

        with torch.no_grad():
            current = x + proj_perturb
            improve = loss >= best_loss
            success = logits.argmax(dim=1) != y
            save = improve | success
            best_loss[improve] = loss[improve]
            best[save] = current[save]

            perturb_next = perturb + alpha_step * perturb.grad.sign()
            perturb_next = perturb_next.clamp(-eps_inf, eps_inf)
            perturb_next = torch.min(torch.max(perturb_next, -x), 1 - x)

            grad_mask = mask_logits.grad
            grad_norm = grad_mask.flatten(1).norm(p=2, dim=1).view(b, 1, 1, 1)
            direction = grad_mask / (grad_norm + 1e-10)
            active_step = torch.full((b, 1, 1, 1), mask_step, device=x.device, dtype=x.dtype)
            active_step[grad_norm < zero_grad_threshold] = 0.0
            mask_next = mask_logits + active_step * direction

            hard_next = project_mask(mask_next, k)
            same = ((hard_next - prev_hard).abs().sum(dim=(1, 2, 3)) == 0) & (~success)
            unchanged[same] += 1
            unchanged[~same] = 0
            reinit = unchanged >= patience
            if reinit.any():
                mask_next[reinit] = torch.randn_like(mask_next[reinit])
                unchanged[reinit] = 0

            perturb = perturb_next
            mask_logits = mask_next

    return best.detach()
```

The sign convention is the main invariant: this code maximizes cross-entropy or `max_other - f_y`, so both updates use plus signs and the best candidate has the largest attack loss. If using the opposite margin `f_y - max_other`, all update signs and best-candidate comparisons must be reversed.
