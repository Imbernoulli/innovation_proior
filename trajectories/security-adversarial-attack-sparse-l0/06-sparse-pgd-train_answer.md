Sparse-RS delivered the step change I predicted: mean ASR $\approx 0.922$ — `0.853` on `Rebuffi-R18-L2`, `0.947` on `Augustin-L2`, `0.967` on `Engstrom-L2`. The random search built for the discrete $L0$ support went from SparseFool's $0.187$ to over $0.92$ and flattened the per-model spread, precisely because random search over feasible supports does not hinge on any one model's boundary linearity. This is the strongest baseline, and it is genuinely strong. But the very thing that makes it the floor for a finale is its one structural concession: it uses *no gradient at all*, even though the harness grants full white-box access including backprop. `Rebuffi-R18-L2` at $0.853$ means roughly $22$ of $150$ correctly-classified samples *survived* $10000$ queries of directed random search — the hard cases, where the right $24$-pixel support is rare enough that blind swapping, even with a provably efficient schedule, does not stumble onto it in budget. For exactly those cases the gradient is information Sparse-RS throws away. So the natural endpoint is to *use the gradient to choose the support* — the one lever every rung either lacked (Pixle, OnePixel, Sparse-RS are gradient-free) or used badly (JSMA greedy, SparseFool local-linear).

The obstacle is the one that sank the white-box rungs: gradient descent does not respect a *combinatorial* support. If I take a gradient step on a dense perturbation and then project onto the $L0$ ball by keeping the top-$k$ coordinates by magnitude, two things go wrong, the same two that made JSMA brittle. The support lurches discontinuously because the projection is a hard top-$k$, so the iterate never settles; and once the projection zeroes a coordinate, *no gradient flows back to it*, so the optimizer is blind to the possibility that a currently-unselected pixel would have been a better choice. That is why naive projected-gradient $L0$ attacks *over-report* robustness — they cannot find the support they should.

I propose **Sparse-PGD (sPGD)**. Stop treating the perturbation as one dense vector to project, and instead make the support its own differentiable variable. Decompose the perturbation as

$$ \delta = p \odot \mathrm{binarize}(m): $$

a dense **magnitude tensor** $p$ (the perturbation values, image-shaped) times the **binarized** version of a continuous **mask** $m$ with one scalar per spatial pixel. The binarization is a hard top-$k$ over the mask — keep the $k$ pixels with the largest mask values, zero the rest — so exactly $k$ pixels survive every forward pass and the $L0$ budget holds *by construction*, no projection of the perturbation required. Now the two sub-problems are explicit and each has its own gradient: the gradient on $p$ says how to push the values on the chosen pixels, the gradient on $m$ says *which* pixels to choose. The support is no longer an implicit byproduct of magnitude ranking; it is a thing the optimizer steers directly. Why a separate mask rather than just ranking $p$ itself by magnitude (which is what PGD0 does)? Because ranking $p$ ties the support choice to the *current* magnitudes — a pixel only enters the support if its perturbation is already large, but its perturbation only grows if it is already in the support, a chicken-and-egg the discontinuous projection resolves arbitrarily. A dedicated mask breaks the cycle: $m$ can grow for a pixel whose magnitude is still small, *pulling* it into the support on the strength of its gradient alone, and only then does $p$ start optimizing its value. The two variables let the attack decide *where* before it has decided *how much* — exactly the ordering a sparse attack wants and the one PGD0 cannot express.

But the hard top-$k$ is still non-differentiable — the same wall. The device that crosses it is **straight-through estimation**: apply the hard top-$k$ in the *forward* pass so the attack stays feasible, but in the *backward* pass let the gradient to $m$ flow as if through the *soft* mask $\sigma(m)$, not the binarized one. Then every pixel — including the ones currently outside the support — receives a gradient telling it whether raising its mask value (entering the support) would lower the loss. That is the exact cure for the dead-gradient failure: the support can move because the unselected coordinates are no longer gradient-zero. There is a second, subtler routing choice for the *perturbation's* gradient: send it through the soft mask (the "unprojected" variant, which updates $p$ as if all pixels were partially active, biasing toward exploration of new supports) or through the hard mask (the "projected" variant, which refines the current support faithfully). They fall into different local optima, so I alternate them across restarts and keep the best example either finds — the white-box analogue of the diversity Sparse-RS got from its population, and the direct answer to those $\sim22$ Rebuffi survivors one biased descent alone would miss.

Grounded in this task's edit surface: the harness hands me a differentiable deep copy, images in $[0,1]$, budget $\text{pixels} = 24$, and validates the $L0$ count channel-wise after the fact. I parameterize $p$ at full image resolution and keep $x + p$ valid *natively*, not by an end-clip that would collapse a sparse attack: after each magnitude step I clamp $p$ into $[-\text{eps}, \text{eps}]$ and then into $[-x,\,1-x]$, which guarantees $x + p \in [0,1]$ per pixel — with $\text{eps} = 1$ because the $L0$ model lets a chosen pixel take any value in the range. The mask $m$ is $(B,1,H,W)$, one scalar per spatial position (matching the harness's per-pixel, channel-collapsed $L0$ count); I sigmoid it before the top-$k$ so the ranking is bounded and the gradients well-scaled. The magnitude update is a PGD sign step $p \leftarrow p - \alpha\,\mathrm{sign}(\nabla_p)$ with $\alpha = 0.25\,\text{eps}$ (descent on the margin); the mask update is a *normalized* gradient step $m \leftarrow m - \beta\sqrt{HW}\,\nabla_m / \lVert\nabla_m\rVert$ with $\beta = 0.25$, because $m$ is a selection variable, not a bounded perturbation, and normalizing keeps the support moving at a consistent rate. The objective is the untargeted margin $f_y - \max_{r\neq y} f_r$, minimized — its sign is the misclassification certificate, and unlike cross-entropy it does not saturate as the attack nears success. I run $t = 300$ iterations across $n\_restarts = 2$ alternating routings and keep a running best across all iterations and both restarts, so the returned image is the strongest $24$-sparse example ever found. The straight-through top-$k$ lives in a small custom autograd function.

The bar is concrete. Sparse-RS already gets $0.922$ gradient-free, so a finale that merely matched it would not justify itself; the claim is that *using the gradient to choose the support* finds the hard examples random search misses in budget. The falsifiable expectation is that sPGD clears the $0.922$ mean and, most tellingly, lifts the *lowest* number — `Rebuffi-R18-L2` at $0.853$, where $\sim22$ samples survived — because that is exactly where a gradient-guided support search should pay off, converting survivors with a rare but *findable-by-gradient* $24$-pixel support into successes. If sPGD came back *below* Sparse-RS, the diagnosis would be that on these particular `L2`-robust surfaces the gradient is so flattened that even a straight-through support search cannot beat directed random search in budget, and the right move would be the ensemble: run sPGD *and* Sparse-RS and take the best of both. But the straight-through, mask-decomposed gradient is the one piece of information every rung on this ladder left on the table, and putting it to work is the natural endpoint of the climb.

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
