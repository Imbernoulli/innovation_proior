One shared backbone feeds a fine 100-way head and a coarse 20-way head, and the single knob I am allowed to turn is how their two cross-entropy losses fold into the one scalar SGD descends. The scaffold default just adds them, $L = L_\text{fine} + L_\text{coarse}$, so the optimizer steps along $g = g_\text{fine} + g_\text{coarse}$ on the shared trunk. The failure I most distrust in this hard-parameter-sharing setup — the one Caruana already flagged in 1997 — is that joint training can land *below* what each task reaches alone: the shared network underperforms exactly the efficiency it was supposed to buy. The optimizer never sees the two gradients separately; it sees their sum and steps along it. If anything is broken it must live in the geometry of that sum, and a fix that only rescales the *losses* can never reach a problem that lives in *directions*.

Stare at the geometry. Two gradient vectors $g_\text{fine}, g_\text{coarse}$ in the same shared-trunk parameter space; the step goes along $g_\text{fine} + g_\text{coarse}$. They fight when they point partly against each other — negative inner product, $g_\text{fine}\cdot g_\text{coarse} < 0$, equivalently negative cosine $\cos\varphi < 0$. In that case
$$\|g_\text{fine} + g_\text{coarse}\|^2 = \|g_\text{fine}\|^2 + \|g_\text{coarse}\|^2 + 2\,g_\text{fine}\cdot g_\text{coarse}$$
is strictly *less* than $\|g_\text{fine}\|^2 + \|g_\text{coarse}\|^2$: the overlap cancels. Conflict alone is not yet fatal — two equal opposing gradients still sum to a direction that decreases $L_\text{fine} + L_\text{coarse}$. What makes it hurt is conflict co-occurring with two other things: *magnitude dominance*, where one loss (the easier 20-way coarse, say) is so much smaller that $g \approx g_\text{fine}$ and the step actively moves against the coarse task; and *high curvature*, where the local linear model a gradient step trusts overestimates the gain on the dominating task and underestimates the harm to the dominated one. The disease is the triad — conflicting, dominating, high curvature — and when all three hit, the summed gradient walks the optimizer into a bad trade it cannot see, and joint training stalls below single-task. Crucially, every magnitude-only fix — learning a per-task variance, balancing loss magnitudes, any $\sum_i w_i L_i$ with nonnegative $w_i$ — is a *scaling*: if the gradients conflict, $w_\text{fine} g_\text{fine} + w_\text{coarse} g_\text{coarse}$ is still two vectors pointing partly against each other, still cancelling in the overlap. Reweighting changes how much each fighter weighs; it does not stop the fight.

The fix therefore has to touch the geometry. I propose **PCGrad — projecting conflicting gradients — in its two-task form**: a *gradient-surgery* rule that removes the part of one gradient that directly undoes the other, and nothing else. The conflict between $g_\text{fine}$ and $g_\text{coarse}$ lives entirely in the component of $g_\text{fine}$ that lies *along* $g_\text{coarse}$, because the inner product only sees that component. Decompose $g_\text{fine}$ into its part parallel to $g_\text{coarse}$ and its part orthogonal. The parallel part is the standard projection $\big((g_\text{fine}\cdot g_\text{coarse})/\|g_\text{coarse}\|^2\big)\,g_\text{coarse}$; under conflict $g_\text{fine}\cdot g_\text{coarse} < 0$, so it points *opposite* to $g_\text{coarse}$ — it is exactly the piece of $g_\text{fine}$ undoing the coarse task's progress. Strip it out,
$$g_\text{fine} \leftarrow g_\text{fine} - \frac{g_\text{fine}\cdot g_\text{coarse}}{\|g_\text{coarse}\|^2}\,g_\text{coarse},$$
and what remains is $g_\text{fine}$ projected onto the *normal plane* of $g_\text{coarse}$: it has zero inner product with $g_\text{coarse}$, no longer conflicts, and was changed by the minimum amount needed. No quadratic program, no line search — one inner product and one axpy, a closed form. This is done *only* when they conflict: if $\cos\varphi \ge 0$ the parallel component is *helping*, pulling fine and coarse in compatible directions, and ripping it out would destroy the positive transfer that is the entire reason for sharing a trunk. The rule is the multi-task, *symmetric* version of the conflict-detection-by-inner-product idea from continual learning — I project $g_\text{coarse}$ onto the normal plane of $g_\text{fine}$ by the same formula. With only two tasks there is no loop and no task-ordering shuffle to keep it unbiased: it is a single symmetric projection.

The combined step is $\Delta\theta = g_\text{fine}^\text{PC} + g_\text{coarse}^\text{PC}$, and it is a legitimate descent direction, not a hack. Writing $g_0 = g_\text{fine}, g_1 = g_\text{coarse}$ and $R = \|g_0\|/\|g_1\|$, the de-conflicted sum rewrites as $g^\text{PC} = (1 - \cos\varphi/R)\,g_0 + (1 - \cos\varphi\cdot R)\,g_1$; under conflict $\cos\varphi < 0$ makes both coefficients exceed 1, so PCGrad does not merely cancel the conflict — it *amplifies* each task's own direction, which is what lets the step escape the cancellation that flattens the plain sum. With a Lipschitz gradient and step $t \le 1/L$, the quadratic upper bound collapses to $L(\theta^+) \le L(\theta) - \tfrac12 t\,(1 - \cos^2\varphi)\,\|g\|^2$: as long as $\cos\varphi > -1$ the factor $(1-\cos^2\varphi)$ is strictly positive, so the objective strictly decreases unless $g=0$ or the gradients are exactly anti-parallel — which two stochastic minibatch gradients essentially never are. The single-step win over the plain sum is precisely the triad regime: PCGrad provably reaches a lower loss when the conflict is sharp enough ($\cos\varphi \le -\Phi$ with $\Phi = 2\|g_0\|\|g_1\|/(\|g_0\|^2+\|g_1\|^2)$ fusing "they conflict" with "their magnitudes differ"), the curvature is high enough, and the step large enough that the curvature mis-estimate bites.

Landing this in the task's edit surface forces a shape that differs from a generic PCGrad wrapper. The wrapper assumes it owns the optimizer — it sits between `backward` and `step`, reads each task's `.grad`, projects, writes back, and lets the base optimizer step. The scaffold gives none of that: the only thing I fill is `MultiTaskLoss.forward(fine_loss, coarse_loss, epoch, total_epochs)`, which receives the two *already-reduced scalar* losses and must *return a scalar* on which the loop calls `.backward()`. So all of PCGrad runs *inside* `forward`, before the loop's single backward. First, I recover the shared parameters myself, because the interface never hands them over: both losses are tensors with a `grad_fn`, so I walk the autograd graph backward from `fine_loss.grad_fn` following `next_functions`, collecting every leaf node whose `.variable` requires grad — those leaves are exactly the trunk-plus-head parameters that fed the loss — and cache the list so I am not re-walking each step. Second, I compute the two per-task gradients *explicitly* with `torch.autograd.grad(fine_loss, params, retain_graph=True)` (and the same for coarse), not by calling `.backward()`, because the loop owns the one backward and I must not consume the graph or populate `.grad` early; `allow_unused=True` with a zero fill handles the head a given task does not touch. I flatten both into vectors $g_0, g_1$, run the single symmetric projection when $\text{dot} = g_0\cdot g_1 < 0$ using the *originals* of both for the two projections, and form $g_\text{pcgrad} = g_0 + g_1$, with small $+10^{-12}$ floors on the squared norms keeping the divisions safe. The final device is what lets a projected gradient survive the "return a scalar" contract: I build a *surrogate loss* whose gradient with respect to each parameter $p$ is the matching chunk of $g_\text{pcgrad}$. Take that chunk, **detach** it (now a constant target), and add $(\text{chunk}\cdot p).\text{sum}()$ to the surrogate; since $\partial(\text{chunk}\cdot p)/\partial p = \text{chunk}$, the loop's backward on the returned surrogate deposits $g_\text{pcgrad}$ into every `p.grad`, and SGD steps along the de-conflicted direction — exactly as if a wrapper had written the corrected gradient back.

Two honest worries remain, and they set up the next rung. Coarse here is a deterministic *coarsening* of fine, so the two gradients are likely *correlated* — $\cos\varphi \ge 0$ — on most steps, in which case PCGrad does nothing but pay two extra backward passes. And even when it fires, it fixes only the *direction* of the conflict; it never asks *how much* each task should count, so the fine/coarse magnitude imbalance between a 100-way and a 20-way cross-entropy is left entirely on the table. My falsifiable expectation is that this rung lands respectably but not at the top, and that on a capacity-scarce backbone the lever that actually matters is the relative weight PCGrad never touches — in which case the next rung stops projecting directions and starts learning the weights.

```python
# EDITABLE region of pytorch-vision/custom_mtl.py (lines 195-216) — step 1: PCGrad (2-task, in-forward)
class MultiTaskLoss(nn.Module):
    """PCGrad: Gradient Surgery for Multi-Task Learning (Yu et al., 2020).

    Projects conflicting task gradients onto the normal plane of the
    other when their cosine similarity is negative, reducing gradient
    interference between tasks.
    """

    def __init__(self, num_tasks=2):
        super().__init__()
        self._shared_params = None

    def _get_shared_params(self, loss):
        """Extract shared model parameters from the computation graph."""
        if self._shared_params is not None:
            return self._shared_params
        # Walk the computation graph to find leaf parameters
        params = []
        seen = set()
        def _walk(grad_fn):
            if grad_fn is None:
                return
            for child, _ in grad_fn.next_functions:
                if child is None:
                    continue
                cid = id(child)
                if cid in seen:
                    continue
                seen.add(cid)
                if hasattr(child, 'variable'):
                    p = child.variable
                    if p.requires_grad:
                        params.append(p)
                _walk(child)
        _walk(loss.grad_fn)
        self._shared_params = params
        return params

    def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
        params = self._get_shared_params(fine_loss)
        if len(params) == 0:
            return fine_loss + coarse_loss

        # Compute per-task gradients
        grads_fine = torch.autograd.grad(
            fine_loss, params, retain_graph=True, allow_unused=True,
        )
        grads_coarse = torch.autograd.grad(
            coarse_loss, params, retain_graph=True, allow_unused=True,
        )

        # Flatten gradients into vectors
        g0 = torch.cat([
            g.flatten() if g is not None else torch.zeros_like(p).flatten()
            for g, p in zip(grads_fine, params)
        ])
        g1 = torch.cat([
            g.flatten() if g is not None else torch.zeros_like(p).flatten()
            for g, p in zip(grads_coarse, params)
        ])

        # PCGrad: project conflicting gradients when cosine similarity < 0
        dot = torch.dot(g0, g1)
        if dot < 0:
            # Project each gradient onto the normal plane of the other.
            # Use originals for both projections (symmetric dot product).
            g0_norm_sq = torch.dot(g0, g0) + 1e-12
            g1_norm_sq = torch.dot(g1, g1) + 1e-12
            g0_proj = g0 - (dot / g1_norm_sq) * g1
            g1_proj = g1 - (dot / g0_norm_sq) * g0
            g0 = g0_proj
            g1 = g1_proj

        # Combined projected gradient
        g_pcgrad = g0 + g1

        # Construct a surrogate loss whose gradient equals g_pcgrad.
        # loss = sum_i (g_pcgrad_i * param_i), so grad w.r.t. param_i = g_pcgrad_i
        offset = 0
        surrogate = torch.tensor(0.0, device=fine_loss.device)
        for p in params:
            numel = p.numel()
            chunk = g_pcgrad[offset:offset + numel].reshape(p.shape).detach()
            surrogate = surrogate + (chunk * p).sum()
            offset += numel
        return surrogate
```
