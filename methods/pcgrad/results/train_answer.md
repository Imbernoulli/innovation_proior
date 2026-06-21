We want one network with shared parameters $\theta$ to learn several tasks at once, on the premise that structure shared across related tasks should make joint learning more efficient than training each task from scratch. The standard recipe is to sum the per-task losses into one objective and descend it, $L(\theta) = \sum_i L_i(\theta)$, stepping along the combined gradient $g = \sum_i g_i$ with $g_i = \nabla L_i(\theta)$. In practice this free lunch repeatedly fails to materialize: joint training reaches *worse* final accuracy and data efficiency than separate training, so badly that some pipelines deliberately train tasks independently first and only then distill them into one network — paying away exactly the efficiency the shared model was supposed to buy. The usual explanations — tasks learn at different speeds, the landscape has plateaus, it is an architecture question of how much to share and where to branch — describe the symptom but not the mechanism. And the mechanism has to live in the one operation those accounts skip: the *combination* of the gradients. The optimizer never sees the individual $g_i$; it sees only their sum and steps along it. So whatever goes wrong must be visible in the geometry of $g = \sum_i g_i$.

Staring at that geometry, two gradients conflict when they point partly against each other, $g_i \cdot g_j < 0$, equivalently $\cos\phi_{ij} = (g_i \cdot g_j)/(\|g_i\|\,\|g_j\|) < 0$, and then $\|g_i + g_j\|^2 = \|g_i\|^2 + \|g_j\|^2 + 2\,g_i\cdot g_j$ is shorter than $\|g_i\|^2 + \|g_j\|^2$ — the overlap cancels. But conflict alone is not the disease: two equal-and-opposite gradients still sum to a vector that descends $L = L_1 + L_2$. Harm needs more. If $g_1$ dominates $g_2$ in magnitude — captured by the gradient magnitude similarity $\Phi(g_i,g_j) = 2\|g_i\|\,\|g_j\| / (\|g_i\|^2 + \|g_j\|^2)$, which is $1$ when the norms match and slides toward $0$ as one dwarfs the other — then $g \approx g_1$, the small task barely registers, and because they conflict the step actively *damages* it. And the trade only becomes a genuine mistake under high curvature: a gradient step trusts a local linear model, but in a region of high positive curvature that model is asymmetrically wrong, overestimating the realized gain on the dominating task and underestimating the harm to the task it moves against. Measuring curvature in the direction the optimizer actually moves gives the path-averaged $H(L;\theta,\theta') = \int_0^1 \nabla L(\theta)^T \nabla^2 L(\theta + a(\theta'-\theta))\,\nabla L(\theta)\,da$, and since deep-net loss valleys are narrow and steeply curved, high $H$ is the common case. So the disease is the *co-occurrence* — conflicting gradients, dominating magnitudes (small $\Phi$), and high curvature — a tragic triad that walks the optimizer into a trade it cannot see and stalls it, visible even in a 2D two-valley example where Adam slides into one valley and jams where the gradients conflict, differ sharply in magnitude, and sit in high curvature. Crucially, none of the existing fixes reach this. Uncertainty weighting and GradNorm reweight the *losses*: $w_1 g_1 + w_2 g_2$ still conflicts for any nonnegative weights, because scaling is a magnitude operation and the cancellation is directional. MGDA does touch geometry — it solves $\min_\alpha \|\sum_t \alpha_t g_t\|^2$ over the simplex — but it is still a convex combination that down-weights conflicting tasks rather than excising the conflicting component, it pays a per-step quadratic program, and near a Pareto-stationary point its min-norm direction collapses toward zero. The continual-learning projections GEM and A-GEM have the right flavor — detect harm by the *sign* of the inner product and project it away — but they protect a privileged past task asymmetrically and (for GEM) still solve a QP, which does not fit learning all tasks simultaneously where no task is privileged. And cosine regularization drives all pairwise cosines to zero *unconditionally*, orthogonalizing even cooperating gradients and throwing away the positive transfer that motivated sharing in the first place.

I propose PCGrad — Projecting Conflicting Gradients — a closed-form gradient-surgery step that removes only the offending component and keeps the rest. The conflict between $g_i$ and $g_j$ lives entirely in the component of $g_i$ along $g_j$, since the inner product $g_i\cdot g_j$ sees only that component. Decompose $g_i$ into its part parallel to $g_j$ and its part orthogonal to $g_j$. The parallel part is the standard projection $((g_i\cdot g_j)/\|g_j\|^2)\,g_j$; when the tasks conflict, $g_i\cdot g_j < 0$, so this part points *opposite* to $g_j$ — it is precisely the piece of $g_i$ that is undoing $g_j$'s progress. Strip it out:
$$g_i \;\leftarrow\; g_i - \frac{g_i\cdot g_j}{\|g_j\|^2}\,g_j .$$
What remains is $g_i$ projected onto the normal plane of $g_j$; it now has zero inner product with $g_j$, so it no longer conflicts, and I changed $g_i$ by the minimum amount needed — I removed exactly the offending parallel component and nothing else. No QP, just one inner product and one axpy. The rule must be *conditional*: I apply it only when $g_i\cdot g_j < 0$. If $\cos\phi_{ij}\ge 0$ the gradients already cooperate, the parallel component is helping, and ripping it out would destroy positive transfer — the very mistake cosine regularization makes. The operation is *symmetric* — I do the same to $g_j$, projecting it onto the normal plane of $g_i$ — because both tasks are first-class. With more than two tasks, each $g_i$ must be de-conflicted against every other $g_j$ in a loop, and one subtlety has to be right: after I project $g_i$ against $g_j$ its direction has changed, so the test against the *next* task $k$ must read the *running* de-conflicted vector $g_i^{PC}$, not the original, or I would re-introduce a conflict I just removed or miss a new one. Because the loop is sequential it is order-dependent — projecting against $j$ then $k$ differs from $k$ then $j$, since the intermediate $g_i^{PC}$ differs — and a fixed order would bias the result toward whichever task comes last and gets the final say on $g_i$'s direction. So I sample the other tasks in *random order* each step, which makes PCGrad symmetric in task order in expectation, mattering most exactly when there are many tasks and conflict is frequent. The final update applies the sum of all de-conflicted gradients, $\Delta\theta = \sum_i g_i^{PC}$. For two conflicting tasks this collapses to the closed form $g^{PC} = g - \frac{g_1\cdot g_2}{\|g_1\|^2}g_1 - \frac{g_1\cdot g_2}{\|g_2\|^2}g_2$, and writing $R = \|g_1\|/\|g_2\|$ gives the revealing rewrite
$$g^{PC} = \Big(1 - \frac{\cos\phi_{12}}{R}\Big)g_1 + \big(1 - \cos\phi_{12}\,R\big)g_2 ,$$
where under conflict ($\cos\phi_{12} < 0$) both coefficients exceed $1$: PCGrad does not merely remove conflict, it *amplifies* each task's own direction, which is what lets the step escape the cancellation that stalls the plain sum. The whole thing assumes nothing about the model — it is an operation on the gradient vectors of the shared parameters, sitting between "compute the per-task gradients" and "hand the optimizer a gradient," so it drops in front of SGD-with-momentum or Adam unchanged and composes with any architecture or loss-weighting scheme.

That PCGrad does not break anything I can verify directly. With $L_1, L_2$ convex and differentiable, $\nabla L$ Lipschitz with constant $L$, and step $t \le 1/L$, the conflicting case $\cos\phi_{12} < 0$ runs the quadratic upper bound $L(\theta^+) \le L(\theta) + \nabla L(\theta)^T(\theta^+ - \theta) + \tfrac{1}{2}L\|\theta^+ - \theta\|^2$ through the algebra and collapses, after substituting $\cos\phi_{12} = (g_1\cdot g_2)/(\|g_1\|\,\|g_2\|)$ and using $t\le 1/L$, to the clean
$$L(\theta^+) \le L(\theta) - \tfrac{1}{2}\,t\,(1 - \cos^2\phi_{12})\,\|g\|^2 .$$
As long as $\cos\phi_{12} > -1$ the factor is strictly positive, so the loss strictly decreases until either $g = 0$ (the optimum) or $\cos\phi_{12} = -1$ (exactly anti-parallel gradients, where the factor and $g^{PC}$ both vanish) — and exact anti-alignment is what stochastic minibatch gradients avoid. Dropping convexity but keeping the Lipschitz gradient, the same inequality telescopes to $\min_k \|g_k\|^2 \le 2(L(\theta_0) - L^*)/(K(1-\alpha^2)t)$ whenever $\cos\phi_{12,k} \ge \alpha > -1$, so even non-convex PCGrad drives the gradient norm to zero, with a rate that degrades as $\alpha \to -1$ — the more severe the typical conflict, the slower, which is the right dependence. Beyond not breaking, PCGrad should *win* exactly where the plain sum is most misleading. With $\nabla L$ Lipschitz and a curvature lower bound $H(L;\theta,\theta^{MT}) \ge \ell\|g\|^2$, Taylor with the integral remainder gives a lower bound on the plain step's $L(\theta^{MT})$, the bound (with the same algebra as above) gives an upper bound on $L(\theta^{PCGrad})$, and their difference factors so that $L(\theta^{PCGrad}) \le L(\theta^{MT})$ under the sufficient conditions (a) $\cos\phi_{12} \le -\Phi(g_1,g_2)$ — the gradients must conflict *enough*, a condition that fuses conflict and magnitude dominance through $\Phi$, since unequal magnitudes make $\Phi$ small so even a mildly negative cosine clears the bar; (b) $\ell \ge \xi(g_1,g_2)\,L$ with the curvature bounding measure $\xi(g_1,g_2) = (1 - \cos^2\phi_{12})\,\|g_1 - g_2\|^2/\|g_1 + g_2\|^2$ — the curvature must be high enough; and (c) $t \ge 2/(\ell - \xi L)$, with a positive curvature margin, so the step is large enough for the curvature mis-estimation to actually bite. These are precisely the tragic-triad conditions recovered as a guarantee, and at $t = 2/(\ell - \xi L)$ the improvement lower bound is the clean $t\,(1 - \cos^2\phi_{12})(\|g_1\|^2 + \|g_2\|^2) \ge 0$. Finally, since I will actually run momentum rather than vanilla SGD, the rewrite $g^{PC} = (1 - \cos\phi_{12}/R)g_1 + (1 - \cos\phi_{12}\,R)g_2$ together with the mean-value form $g^{PC} = H_k(\theta_k - \theta^*)$ shows that for $L_i$-smooth, $\mu_i$-strongly-convex tasks the eigenvalues of $H_k$ lie in $[\mu_k, L_k]$, and the standard heavy-ball tuning $\alpha_k = 4/(\sqrt{L_k} + \sqrt{\mu_k})$, $\beta_k = \max\{|1 - \sqrt{\alpha_k\mu_k}|, |1 - \sqrt{\alpha_k L_k}|\}^2$ contracts the companion matrix's spectral norm to $(\sqrt{\kappa_k} - 1)/(\sqrt{\kappa_k} + 1) < 1$ — so PCGrad with momentum converges linearly, inheriting the same single failure mode at $\cos\phi_{12} = -1$ and otherwise speeding up.

```python
import copy
import numpy as np
import random
import torch


class PCGrad:
    """Project Conflicting Gradients. Wraps a base optimizer; de-conflicts per-task
    gradients of the shared parameters, then steps the base optimizer as usual."""

    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        return self._optim.step()

    def pc_backward(self, objectives):
        """objectives: list of per-task scalar losses [L_1, ..., L_n]."""
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)

    def _project_conflicting(self, grads, has_grads):
        shared = torch.stack(has_grads).prod(0).bool()    # params touched by every task
        pc_grad = copy.deepcopy(grads)
        for g_i in pc_grad:
            random.shuffle(grads)                         # random task order
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)             # conflict test: sign of inner product
                if g_i_g_j < 0:                           # cos(g_i, g_j) < 0
                    g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)   # project onto g_j's normal plane
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction == 'mean':
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
        else:
            exit('invalid reduction method')
        merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _pack_grad(self, objectives):
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)               # one backward per task
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _retrieve_grad(self):
        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None:                        # task may not touch every param (multi-head)
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

    def _flatten_grad(self, grads, shapes):
        return torch.cat([g.flatten() for g in grads])

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _set_grad(self, grads):
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1
```

The two-task specialization, for example a fine head and a coarse head sharing a backbone, where the loop reduces to a single symmetric projection:

```python
import torch


def pcgrad_two_task(g0, g1):
    """g0, g1: flattened gradients of the shared parameters for the two tasks.
    Returns the combined update Delta theta = g0^PC + g1^PC."""
    dot = torch.dot(g0, g1)
    if dot < 0:                                           # conflicting
        g0_proj = g0 - dot / (g1.norm() ** 2) * g1        # remove g0's component along g1
        g1_proj = g1 - dot / (g0.norm() ** 2) * g0        # remove g1's component along g0 (original g0)
        g0, g1 = g0_proj, g1_proj
    return g0 + g1
```
