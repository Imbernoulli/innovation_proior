The starting pipeline already carries two strong regularizers — L2 weight decay through the optimizer (`weight_decay=5e-4`) and BatchNorm in all three architectures — yet the convolutional layers, where the spatial parameters live, are exactly where our best injected-noise tool has quietly stopped working. Dropout has all but disappeared from modern conv stacks because of spatial correlation: a convolutional feature map is one kernel slid over a smooth input, so neighboring output positions read heavily overlapping receptive fields and carry almost the same number. Zero one activation and the next layer's convolution, sliding over the hole, simply reads the same edge off the surviving neighbors — the information leaks around the hole and the "thinned sub-network" was never thinned. The fix in the activation-masking world is DropBlock, which removes information at the *scale of correlation* by deleting a contiguous $block\_size \times block\_size$ square so no correlated neighbor survives inside the hole. But that is a forward-graph layer — it samples a Bernoulli seed field, expands seeds with a stride-1 max-pool, multiplies the activations by the keep-mask, rescales by the realized survival fraction, and is a no-op at inference. The one region I am allowed to edit returns a single differentiable scalar *added to the loss*; it cannot insert a layer, mask an activation, or change inference behavior. So I must keep the DropBlock *insight* and re-express it as a loss-side penalty on the only thing the function can touch every step: the convolutional weights.

What I propose is a **DropBlock-inspired spatial co-activation penalty on conv weights** — a weight-space echo of "don't let any contiguous block of the feature map carry everything." The bridge from activations to weights is this: the output of a convolution at a position is the inner product of the kernel with the input patch under it, so a kernel whose energy is concentrated in a tight, contiguous cluster of taps is precisely a detector that fires maximally on a small contiguous input region — the very reliance DropBlock fights at the activation level. The weight-side analogue is therefore to penalize filters whose magnitude is spatially clustered within local blocks of the kernel itself, pushing them toward smoother, more distributed responses. Concretely, for each `Conv2d` weight $w$ of shape $[out_c, in_c, k_H, k_W]$ I collapse the input-channel axis — I care about spatial structure, not which input channel it came from — by taking the per-output-channel, per-tap mean squared magnitude $w_{sq} = \mathrm{mean}_{in}(w^2)$ of shape $[out_c, 1, k_H, k_W]$, a small spatial energy map for each filter. I then read off the energy *inside local contiguous blocks* of that map with a stride-1 average pool of kernel $block\_size$ and padding $\lfloor block\_size/2 \rfloor$ to preserve size, so each entry of $F.\mathrm{avg\_pool2d}(w_{sq},\,block\_size,\,1,\,pad)$ is the mean squared weight magnitude in the $block\_size \times block\_size$ neighborhood around a tap. The mean of that over all taps and filters is a scalar — large when filters pack energy into tight contiguous sub-blocks, small when energy is spread thin — and penalizing it nudges filters toward the latter. The whole thing is a square, a mean, an average pool, and a mean, fully differentiable with autograd supplying the gradient.

Two design parameters fall out, and the load-bearing care is in choosing them. The first is $block\_size$, the spatial scale of the contiguous region I am discouraging. The CIFAR architectures here use $3 \times 3$ convolutions almost everywhere, so I set $block\_size = 3$: the whole kernel, the smallest contiguous scale that still spans a genuine local neighborhood rather than a single tap. I apply the penalty only to convs whose kernel is at least $block\_size$ in both spatial dimensions, which correctly skips the $1 \times 1$ pointwise convolutions in VGG's classifier path and MobileNetV2's expansion/projection layers — they have no spatial structure to regularize. The second parameter, the strength, is the delicate one, and here I confront the same wall the activation-masking version hit. A strong penalty from step zero shapes filters that have not learned anything yet, fighting the network's search for *any* useful structure while the weights are still near Kaiming-random; worse, these are BatchNorm-heavy nets where BN couples the conv-weight scale to its running statistics, so a weight-shaping force applied before BN settles can destabilize the statistics the rest of training depends on. The cure DropBlock used at the activation level — ramp the drop probability linearly from zero — I transplant to the weights: keep the penalty *off entirely* for the first 20% of training, then ramp it linearly to its target. From the loop's `config` I form $progress = epoch / \max(total\_epochs - 1,\,1)$; for $progress < 0.2$ I return exactly zero so the network builds features and BN settles undisturbed; after that $adjusted = (progress - 0.2)/0.8$ runs 0 to 1 over the remaining 80% and $\lambda = \lambda_{max}\cdot adjusted$, reaching full strength only at the end when filters are mature and over-fitting is the live problem.

For the target strength I set $\lambda_{max} = 1e\text{-}4$. This penalty rides on top of L2 weight decay and BatchNorm, both already doing heavy lifting, and the quantity it penalizes — mean squared weight magnitude inside local blocks — is on the same scale as the squared weights L2 already shrinks. It is meant as a light, complementary nudge to the *spatial shape* of filters, not a second weight decay; pushed hard it would merely fight L2 for a marginal shape preference and cost accuracy. So $\lambda_{max} = 1e\text{-}4$, an order of magnitude below the L2 coefficient, perturbs filter shape gently without dominating the magnitude control L2 already provides. I also divide the accumulated penalty by the number of contributing layers so its scale does not grow with depth — a 56-layer ResNet and a 16-layer VGG see a comparably-scaled term, which matters because the one $\lambda_{max}$ has to travel across all three architectures. The penalty reads only `model` and the epoch fields of `config`, ignores `inputs`/`outputs`/`targets`, returns a scalar on `outputs.device`, and adds essentially nothing to step cost.

I am clear-eyed that this is a *weak* regularizer by construction, and I start the ladder here deliberately. It is two steps removed from the failure it imitates — DropBlock removes information from the map directly, while this only reshapes filter weights and never touches an activation; it lands on a pipeline that already shrinks those same weights with L2 and normalizes their activations with BN, leaving thin marginal room; and the delayed warm-up means it reaches full strength precisely when the cosine schedule has driven the learning rate toward zero and the weights are barely moving, so it has the least leverage exactly when it is finally strong. I expect it to sit near the floor of the field, plausibly the weakest, with the largest shortfall on the harder CIFAR-100 pairs — ResNet-56, whose residual paths let the network route around any single filter's shape preference, and VGG-16-BN, whose real capacity sink is a 512-wide dense head this conv-only penalty never touches — and a near-tie on the nearly-saturated MobileNetV2/FashionMNIST pair. The point of this rung is to establish that floor and to motivate moving the point of action away from filter shape and toward the output distribution and the weight spectrum, which the next rungs take up.

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
