The training objective for a large neural network has stopped being informative on its own. I can drive the empirical loss $L_S(w)$ close to zero, but in an overparameterized model that scalar only certifies that this particular point fits the sample; it says nothing about whether I sit in a stable basin or balanced on a narrow spike. Two solutions can share the same training loss and differ sharply in population loss, which means the information that distinguishes them is not the value at the point but the behavior of the loss in the neighborhood around it. The practical problem, then, is not just to find a low-loss point — it is to select among the many low-loss points using a property that tracks generalization and can be folded into ordinary gradient-based training.

Flatness is the natural candidate signal. The old flat-minimum intuition prefers a solution whose loss stays low even when the weights are specified imprecisely: a flat solution tolerates a coarse, low-bit description, while a sharp one needs many bits because the loss moves quickly under tiny weight changes. Large-batch training makes this concrete — large and small batches reach similar training losses, but large-batch optimization tends to settle in sharper, higher-curvature regions that generalize worse. The trouble is that raw flatness is not a theorem by itself. Deep networks carry scale symmetries and reparameterizations that can make the same function look flatter or sharper, so a usable principle cannot lean on a bare curvature term; it has to pair the local-geometry term with some control on parameter scale or description cost. The existing routes toward wide basins each leave the optimizer slot unfilled: explicit flat-minimum search needs box-volume and second-order machinery, local-entropy methods smooth the landscape but estimate their objective with an inner stochastic dynamics loop, and weight averaging can drift into wider regions but never asks, per step, which nearby direction is actually dangerous.

I propose Sharpness-Aware Minimization. The pairing I need comes from PAC-Bayes: instead of certifying a single deterministic weight vector, it bounds the population loss of a stochastic predictor drawn from a posterior centered at $w$ by the empirical loss averaged under that posterior plus a KL cost to a prior. With a Gaussian posterior around $w$, the empirical term becomes an average of $L_S(w + \epsilon)$ — already a neighborhood quantity that tells me to fit not at $w$ but under perturbations of $w$. Optimizing that average directly, though, is too passive: random directions in a high-dimensional space almost all miss the one steep wall near me, and averaging over them dilutes exactly the vulnerability that sharpness is about. Since most of a Gaussian's mass lies inside a ball of appropriate radius, the maximum loss inside that ball upper-bounds the averaged perturbed loss on the high-probability event, so I replace the soft average by the hard local worst case and minimize

$$\min_w\; L_S^{\mathrm{SAM}}(w) + \lambda \lVert w \rVert_2^2, \qquad L_S^{\mathrm{SAM}}(w) = \max_{\lVert \epsilon \rVert_p \le \rho} L_S(w + \epsilon).$$

This objective has the right shape because it decomposes as $L_S^{\mathrm{SAM}}(w) = L_S(w) + \big[\max_{\lVert\epsilon\rVert\le\rho} L_S(w+\epsilon) - L_S(w)\big]$: a fit term plus a sharpness term, the rise in loss under the worst nearby move. The bound's remaining piece grows like $\lVert w\rVert^2/\rho^2$, which is why a norm penalty belongs here too. The weight decay is not the main idea — it is merely the simple surrogate for the complexity term that scale control demands; the distinctive pressure is to train the whole neighborhood, not the isolated point.

The obvious wall is that this objective hides an inner maximization over a nonconvex loss, and solving it exactly every step would kill the method. So I take the cheapest useful adversary. Because $\rho$ is small, I linearize: $L_S(w+\epsilon) \approx L_S(w) + \epsilon^\top g$ with $g = \nabla_w L_S(w)$. The constant drops out of the inner problem and I am left maximizing a linear function over a norm ball, $\arg\max_{\lVert\epsilon\rVert_p\le\rho} \epsilon^\top g$, which is exactly where dual norms do the work. Hölder's inequality gives $\epsilon^\top g \le \lVert\epsilon\rVert_p \lVert g\rVert_q \le \rho \lVert g\rVert_q$ with $1/p + 1/q = 1$, and the equality condition hands me the adversarial perturbation in closed form,

$$\hat\epsilon = \rho \cdot \mathrm{sign}(g)\, \frac{\lvert g\rvert^{q-1}}{\big(\lVert g\rVert_q^q\big)^{1/p}},$$

which for the default $p = 2$ collapses to the clean

$$\hat\epsilon = \rho \cdot \frac{g}{\lVert g\rVert_2 + \epsilon}.$$

The inner maximization is now trainable: one gradient, normalized to radius $\rho$, stepping uphill in weight space — the same first-order adversarial trick that works for input perturbations, but with the parameters as the target. The sign is load-bearing. The inner problem is a maximum, so $\hat\epsilon$ points uphill and I add it to $w$; the outer problem is a minimum, so I descend using the gradient measured at that uphill point. Subtracting the perturbation first would sample an easier nearby point and hide the steep direction — the update must climb first, then descend. I keep $p = 2$ because that is the ball the bound naturally produces and because it preserves the gradient's magnitude structure; the $p=\infty$ sign step throws that structure away and an $L_1$ perturbation collapses onto a single coordinate.

There is one more simplification that makes the method scalable. Differentiating $L_S^{\mathrm{SAM}}(w) \approx L_S(w + \hat\epsilon(w))$ exactly yields not only the gradient at the perturbed point but also a term from $\hat\epsilon(w)$'s dependence on $w$, and since $\hat\epsilon$ is built from $\nabla L_S(w)$ that term carries the Hessian — realizable as Hessian-vector products, but enough to turn a light two-pass optimizer into a heavier second-order procedure. The dominant signal I want is the gradient at the locally bad point, which already says how to lower the worst nearby loss; the second-order term only refines how the adversary itself shifts as $w$ moves. So I drop it and use

$$\nabla_w L_S^{\mathrm{SAM}}(w) \approx \nabla_w L_S(w)\big|_{w + \hat\epsilon(w)}.$$

This is the decisive move: the method becomes a thin wrapper around any base optimizer. On a minibatch I compute the ordinary gradient $g = \nabla L_B(w)$, form $\hat\epsilon = \rho\, g/(\lVert g\rVert_2 + \epsilon)$, ascend to $w_{\mathrm{adv}} = w + \hat\epsilon$, compute the ordinary gradient $g_{\mathrm{sam}} = \nabla L_B(w_{\mathrm{adv}})$ there, restore $w$, and let the base optimizer apply $g_{\mathrm{sam}}$ — about two backpropagations per step, no Hessian and no inner chain. This is why the method is not noise injection: noise would pick $\epsilon$ without consulting the loss slope and merely test whether the basin is usually fine, whereas this perturbation is constructed to raise the loss as much as the first-order model allows and so tests whether the basin has an exposed wall. It is also not weight decay, which pulls down the norm at the current point and never evaluates $L_S(w + \epsilon)$ at all. With minibatches the meaning shifts slightly — each batch (or each data-parallel shard, if shards build their own perturbations before the final gradients are averaged) measures a smaller-subset sharpness rather than one global sharpness, which is not a defect and can even correlate better with generalization. The method asks for a single new hyperparameter, $\rho$, and reuses the existing loss, model, and base optimizer.

```python
class SAM:
    def __init__(self, params, base_optimizer, rho=0.05):
        self.params = list(params)
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.state = {}

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for p in self.params:
            if p.grad is None:
                continue
            e_w = self.rho * p.grad / (grad_norm + 1e-12)
            self.state[p] = e_w
            p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for p in self.params:
            if p.grad is None:
                continue
            p.sub_(self.state[p])
        self.base_optimizer.step()
        self.state.clear()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        return torch.norm(
            torch.stack([p.grad.norm(p=2) for p in self.params if p.grad is not None])
        )

    def zero_grad(self):
        self.base_optimizer.zero_grad()
```
