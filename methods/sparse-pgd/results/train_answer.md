We want a white-box attack that, given a classifier $f:[0,1]^{C\times H\times W}\to\mathbb{R}^K$, a correctly classified image $x$ with label $y$, and a fixed spatial budget $k$, produces an adversarial image $x'$ that changes at most $k$ spatial pixels (a pixel counts once no matter how many color channels move at that location), keeps every channel inside $[0,1]$, and flips the prediction so that $\arg\max_r f_r(x')\neq y$. This is the tool used to measure how much robust accuracy actually survives on adversarially trained models, so the attack has to do four things at once: return candidates that are exactly feasible under the spatial $L_0$ budget and the box, use the model gradient rather than spending a separate query on each coordinate, search the combinatorial support well enough that robust accuracy is not overestimated, and stay cheap enough to run for thousands of iterations or for a handful of iterations inside adversarial training.

The reason ordinary PGD does not transfer is geometric. Under $L_\infty$ or $L_2$ the feasible set is smooth and its projection is a gentle clip or rescale, so "step in the loss-increasing direction, project back" is a true projected descent. The spatial $L_0$ feasible set is instead a union of coordinate subspaces, one per support of size at most $k$, and the Euclidean projection onto it is a discrete top-$k$ operation: keep the largest-magnitude coordinates and zero the rest. A dense gradient step followed by that top-$k$ projection can abruptly swap the support, and the coordinates it drops are no longer treated as live candidates for the next iteration, so the iteration thrashes rather than converging. The prior options each leave a gap here. JSMA builds saliency maps and commits to high-saliency features greedily, with no joint iterative optimization of support and values. SparseFool repeatedly linearizes the decision boundary, but the local-linear model is brittle on robust models and is not an exact fixed-budget optimizer. PGD$_0$ does exactly "step, top-$k$ project, repeat," and the top-$k$ projection is precisely the unstable operation. Sparse-RS is a strong black-box random search but, by design, ignores the gradient even when it is available. SAIF shows that separating values from support-like variables is promising but leaves open whether a PGD-like white-box method can make the support update simple and direct. The genuinely hard part throughout is not choosing a selected pixel's value — once a pixel is selected, its channels may move anywhere in the box at no $L_0$ cost — it is learning the support with the gradient.

I propose Sparse-PGD (sPGD). The core idea is to stop treating the perturbation as one tensor projected onto $L_0$ and instead split it into the two decisions it actually contains: a dense magnitude tensor $p$ of the same shape as the image, answering "what value should a selected pixel take," and a binary spatial mask $m$ of shape $1\times H\times W$, broadcast over channels, answering "which pixels are selected." The perturbation is

$$\delta = p\odot m,$$

and the constraints now separate cleanly: $p$ must lie in $[-x,\,1-x]$ so that $x+p$ is a valid image, while $m$ must contain at most $k$ ones. To optimize the mask with gradients I put a continuous logit tensor $u$ behind it and form a soft mask $s=\sigma(u)$; the hard forward mask is the top-$k$ projection $m=\Pi_k(\sigma(u))$, which sets the $k$ largest spatial entries of $s$ to one and the rest to zero. The forward pass always uses the hard mask, because the attack must be feasible at every step. The backward pass cannot use the literal derivative of top-$k$ — it is zero almost everywhere and undefined at ties — so I discard it and route the gradient through the sigmoid in a straight-through fashion. That single choice is what lets the support move: even a pixel currently outside the top-$k$ still has a defined $\partial L/\partial\delta$ at its output coordinate, so its logit receives evidence about whether its score should rise.

The sign convention is fixed once and held everywhere. I want an untargeted attack, so I choose a loss whose larger value means a worse classification of the true label and I maximize it; with cross-entropy this pushes the correct class down, and both updates therefore step in the positive gradient direction (gradient ascent, plus signs), with the best candidate being the one of largest loss. The reference loss is cross-entropy for confident samples with a denominator-free margin fallback $\max_{r\neq y}f_r - f_y$ for low-confidence samples — both larger when the image is more adversarial, so the convention is consistent. Mixing in the opposite margin $f_y-\max_{r\neq y}f_r$ would require flipping every update sign and reversing the best-candidate comparison, which is the easiest way to silently produce a wrong attack.

The magnitude update is a box-constrained sign step,

$$p \leftarrow \Pi_{S_p}\!\big(p + \alpha\,\mathrm{sign}(g_p)\big),$$

where the projection $\Pi_{S_p}$ first clamps to $[-\varepsilon_\infty,\varepsilon_\infty]$ and then to $[-x,\,1-x]$, so $0\le x+p\le 1$ before the mask is ever applied; multiplying by the mask cannot violate the box, since it either keeps a valid entry or zeroes it. There are two deliberate routes for $g_p$. The projected gradient $g_p=(\partial L/\partial\delta)\odot m$ updates only the currently selected pixels — faithful to the forward support, exploitative. The unprojected gradient $g_p=(\partial L/\partial\delta)\odot\sigma(u)$ updates $p$ densely, weighted by the soft mask, so values outside the current top-$k$ are not frozen — less faithful but exploratory. I keep both as siblings (MaskingB and MaskingA respectively) rather than choosing one, because they answer different needs.

The mask-logit gradient is the support signal. Since the spatial mask is shared across channels, the gradient to the soft mask at a pixel is the channel sum $\sum_c (\partial L/\partial\delta_c)\,p_c$, and passing through $\sigma$ multiplies by $\sigma(u)(1-\sigma(u))$, giving the straight-through approximation

$$g_u \approx \Big(\textstyle\sum_c (\partial L/\partial\delta_c)\,p_c\Big)\,\sigma(u)\,(1-\sigma(u)).$$

The magnitude and mask steps must not be the same kind of step. The value tensor $p$ is box-constrained like an $L_\infty$ perturbation, so a sign step is natural. The mask logits are unconstrained and exist only to rank pixels; a per-coordinate sign step would make the ranking thrash and the effective step size scale badly with dimension. So I normalize the mask gradient and scale by the square root of the number of mask entries,

$$u \leftarrow u + \beta\,\sqrt{H W}\,\frac{g_u}{\lVert g_u\rVert_2},\qquad m\leftarrow\Pi_k(\sigma(u)),$$

with a small-gradient guard: divide by $\lVert g_u\rVert_2 + 10^{-10}$ and zero the step when the norm falls below a tiny threshold so saturated logits are not pushed farther for no reason. The constants follow from the budgets: the selected-pixel magnitude range is the full image range, so $\varepsilon_\infty=1$ ($255/255$); the magnitude step is $\alpha=0.25\,\varepsilon_\infty$; the mask coefficient is $\beta=0.25$, applied as $0.25\sqrt{HW}$.

One more failure mode remains: because the hard mask only changes when the ordering of the soft scores changes, the logits can drift for several steps while the top-$k$ support stays identical and the sample remains correctly classified — the support stalls. So if a still-correct sample's projected mask is unchanged for $\texttt{patience}$ consecutive iterations (default $3$ for the unstructured attack), I reinitialize that sample's mask logits randomly and clear its counter. There is no need to perturb $p$: its values stay valid and the fresh support can reuse or revise them. The same logic lifts to a structured extension where, instead of pixels, one selects groups — rows, columns, patches, or a custom binary pattern — by projecting a group mask to $k$ groups and mapping it to a spatial mask with a transposed convolution against the pattern kernel (clipped to one), routing the backward gradient back to the group grid by convolution and again ignoring the nondifferentiable clip; with a $1\times1$ kernel this reduces exactly to the unstructured pixel case. The full evaluator, Sparse-AutoAttack, cascades $\text{sPGD}_{\text{unproj}}$, then $\text{sPGD}_{\text{proj}}$, then Sparse-RS, skipping later attacks for any sample already broken — using the gradient that Sparse-RS ignores while still respecting the fixed sparse budget at every forward pass. What makes the whole method work is the division of labor: forward feasibility from hard top-$k$ on $\sigma(u)$ plus the two clamps on $p$, and backward support learning from discarding the top-$k$ derivative so every pixel's logit still sees $(\partial L/\partial\delta)\,p\,\sigma'(u)$.

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
