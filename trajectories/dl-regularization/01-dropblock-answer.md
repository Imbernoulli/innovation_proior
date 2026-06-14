**Problem.** Dropout is the best injected noise we have, but it fails on convolutional layers: spatial
correlation lets a dropped unit's information leak around the hole through its neighbors. The
activation-masking DropBlock fixes this by zeroing contiguous `block_size × block_size` squares — but
that is a forward-graph layer with rescaling and an inference no-op, and this edit surface only allows a
differentiable scalar *added to the loss*. So the DropBlock insight must be re-expressed as a loss-side
penalty on the one thing the function can touch every step: the convolutional weights.

**Key idea (this task's baseline).** Discourage filters from being sharp *contiguous-region detectors* —
the weight-space echo of "don't let any contiguous block of the feature map carry everything." For each
`Conv2d` weight `w = [out_c, in_c, kH, kW]`, collapse the input-channel axis to a per-filter spatial
energy map `w_sq = mean_{in}(w^2)` of shape `[out_c, 1, kH, kW]`, then measure the energy inside local
`block_size × block_size` neighborhoods with a stride-1 average pool and penalize its mean. Filters that
pack energy into a tight contiguous sub-block are pushed toward smoother, more distributed responses.
This is *not* the paper's activation-masking layer — it never masks an activation, rescales, or changes
inference; it is a pure weight-shape penalty.

**Why it should help / why it is weak.** It targets the same reliance-on-contiguous-regions failure as
DropBlock, but indirectly (weights, not activations), on a pipeline that already runs L2 weight decay
and BatchNorm — leaving thin marginal room. A fixed strong penalty from step zero would shape
not-yet-learned filters and shock the BN-heavy early dynamics, so the penalty is held **off for the
first 20% of training** and then ramped linearly to its target — exactly when the cosine schedule has
nearly stopped the weights, giving it the least leverage at full strength. Expect a floor-of-the-ladder
result, weakest on the harder CIFAR-100 pairs.

**Hyperparameters.** `block_size = 3` (the full `3×3` kernel — the smallest genuine local neighborhood;
applied only to convs with kernel `>= 3`, so `1×1` pointwise convs are skipped). `lambda_max = 1e-4` (an
order of magnitude below the L2 coefficient — a light complement, not a second weight decay). Schedule:
zero penalty for `progress < 0.2`, then linear `lam = lambda_max · (progress − 0.2)/0.8` to full
strength at the end. Averaged over contributing layers so the scale is depth-independent across the
three architectures.

```python
# EDITABLE region of pytorch-vision/custom_reg.py (lines 246-273) — step 1: DropBlock-inspired
# spatial co-activation penalty on conv weights. torch / nn / F already imported at module scope.
def compute_regularization(model, inputs, outputs, targets, config):
    """Spatial co-activation penalty on convolutional weights.

    Applies a spatial co-activation penalty on convolutional weights.
    For each Conv2d layer with spatial kernels >= block_size, it
    penalizes the mean energy of local spatial blocks in the weight
    tensor, discouraging spatially correlated filter patterns.

    Uses conservative strength (lambda_max=1e-4) with linear warm-up
    and only activates after 20% of training to avoid destabilizing
    early learning, particularly for BatchNorm-heavy architectures.

    block_size=3, lambda_max=1e-4, linear warm-up with delayed start.
    """
    block_size = 3
    lambda_max = 1e-4
    progress = config['epoch'] / max(config['total_epochs'] - 1, 1)

    # Delay activation: no penalty for first 20% of training
    if progress < 0.2:
        return torch.tensor(0.0, device=outputs.device)

    # Linear schedule from 20% to 100% of training
    adjusted_progress = (progress - 0.2) / 0.8
    lam = lambda_max * adjusted_progress

    reg = torch.tensor(0.0, device=outputs.device)
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size[0] >= block_size:
            w = m.weight  # [out_c, in_c, kH, kW]
            if w.size(-1) >= block_size and w.size(-2) >= block_size:
                # Mean squared magnitude within spatial blocks
                w_sq = w.pow(2).mean(dim=1, keepdim=True)  # [out_c, 1, kH, kW]
                pad = block_size // 2
                local = F.avg_pool2d(w_sq, block_size, stride=1, padding=pad)
                reg = reg + local.mean()
                count += 1

    if count > 0:
        reg = reg / count
    return lam * reg
```
