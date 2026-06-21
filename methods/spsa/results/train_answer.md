We want to certify how robust an image classifier really is inside the $L_\infty$ ball of radius $\epsilon$ around a clean image: does there exist a perturbation $\delta$ with $\|\delta\|_\infty \le \epsilon$ that flips the prediction? The only honest way to answer is to attack, and the attack everyone reaches for is projected gradient descent — step on the margin loss along $\nabla_x J$, project back into the ball, repeat. It works beautifully when the gradient is real, but I keep hitting models where it is not. Some put a non-differentiable transform in front of the network — JPEG compression, quantization, a discrete purification loop — and backprop through the whole pipeline simply does not exist, so PGD has nothing to step on. Others are differentiable but the gradient is a lie: tiny, noisy, or pointed the wrong way, so PGD wanders, finds nothing, and I dutifully report a high robust accuracy. The thing that should scare me is that from the outside I cannot tell these apart. "PGD failed" might mean the model is robust, or it might mean the model hid its gradient from me — the documented failure called gradient masking. As long as the attacker depends on the gradient, a defense can defeat the attacker without defeating the attack surface, and I would never know. Transfer attacks from a differentiable surrogate dodge the gradient problem but only succeed insofar as the surrogate resembles the target, so a passing transfer attack is weak evidence too. Coordinatewise finite differencing in the Kiefer–Wolfowitz style is genuinely gradient-free — it queries only the function value — but assembling one full gradient costs on the order of $2D$ forward queries, and the optimization variable here is the entire image, $D = C\cdot H\cdot W$, which is over a hundred thousand on ImageNet; that economics is hopeless under any realistic budget.

So I impose a brutal constraint on myself: the attack may only feed inputs forward and read back logits — no gradients, no weights. If a model survives that, "it survived" can no longer be blamed on missing or masked gradients. I keep the same objective PGD uses, the correct-class advantage $J(x) = m(x)_{y_0} - \max_{j\ne y_0} m(x)_j$ over $x$ in the $\epsilon$-ball, where $m(x)_j$ is the logit of class $j$ and $J(x)<0$ exactly when $x$ is misclassified. The margin form is deliberate: cross-entropy saturates once the model is confident — the softmax pins at $0$ or $1$ and the loss goes flat, so its value barely moves as I perturb $x$ — whereas the logits stay roughly linear out there, so the margin keeps a usable slope long after cross-entropy has died. That slope matters, because differences in the objective are all I will have.

The method I propose is SPSA — simultaneous-perturbation stochastic approximation — turned into a forward-only $L_\infty$ attack. The starting point is that descending an objective I can only measure, never differentiate, is the classical stochastic-approximation setting: if I had a noisy gradient I would run Robbins–Monro, $x_{k+1} = x_k - a_k\,\hat g_k$, with the step decaying neither too fast nor too slow. The textbook way to manufacture $\hat g$ from values alone is finite differences, but probing each coordinate in isolation along $e_i$ is exactly what costs $D$ separate probes. The insight is to stop isolating. Perturb all coordinates at once with a single random vector $v\in\mathbb{R}^D$ and take the two-sided difference along it,
$$df = J(x + c\,v) - J(x - c\,v) = 2c\,(v\cdot\nabla J) + O(c^3),$$
where the symmetric form makes the quadratic Taylor terms cancel, so the leading bias is $O(c^2)$ rather than $O(c)$ and the noise stays centered. That is one scalar — the directional derivative along $v$ — mixing all $D$ partials together. To pull the $i$-th partial back out, I divide by $v_i$:
$$\hat g_i = \frac{df}{2c\,v_i} = \partial_i J + \sum_{j\ne i}\frac{v_j}{v_i}\,\partial_j J + O(c^2).$$
The $j=i$ term gives exactly $\partial_i J$; every other partial appears too, but carried by a random coefficient $v_j/v_i$. This is only useful if that cross-talk vanishes in expectation. With $v$ having independent, mean-zero components, $E[v_j/v_i] = E[v_j]\,E[1/v_i] = 0$ for $j\ne i$, provided $E[1/v_i]$ is finite — so $E[\hat g_i] = \partial_i J + O(c^2)$. The estimator is almost unbiased, and the entire gradient vector — all $D$ components — comes from the same two scalars $J(x\pm c\,v)$, regardless of $D$. That is the economics I needed: I stopped demanding each measurement be clean and instead let each measurement be wrong per-sample but right on average, buying back accuracy by averaging rather than by isolation.

The finite-inverse-moment caveat is not a technicality; it decides everything. The required regularity needs finite inverse moments such as $E[|1/v_i|]$ and $E[1/v_i^2]$. The instinctive choice $v\sim N(0,I)$ fails: the Gaussian density is bounded away from zero near the origin while $1/|t|$ and $1/t^2$ blow up there, so those inverse moments diverge and the cross-talk terms lose a finite mean; in practice this surfaces as occasional gigantic estimates whenever some $v_i$ lands near zero and I divide by it. Uniform-around-zero fails for the same reason. So I need a mean-zero distribution whose components stay away from zero, and the cleanest is the symmetric Bernoulli (Rademacher), $v_i\in\{+1,-1\}$ each with probability $1/2$: mean zero, and since $v_i=\pm 1$ its inverse moments of every order are bounded. The choice is forced by the condition that makes the trick work, not an aesthetic preference, and it carries a bonus — because $1/v_i = v_i$ for $\pm 1$, dividing by the perturbation is the same as multiplying by it, so $\hat g = (df/2c)\cdot v$ needs no per-coordinate division at all.

A single two-query estimate is correct on average but jittery — its direction points partly sideways from the cross-talk, and the defended model may itself be stochastic (random resizing, dropout, sampling purifiers), adding a noise floor to each query. The cure is the same averaging the asymptotics rely on, applied within a step: draw $n$ independent Rademacher vectors and average,
$$\bar g = \frac{1}{n}\sum_{i=1}^{n}\frac{J(x + c\,v^{(i)}) - J(x - c\,v^{(i)})}{2c}\,v^{(i)},$$
so the variance falls like $1/n$ and the surviving cross-talk is knocked toward its zero mean. This costs $2n$ queries per step, which is exactly a batch a GPU evaluates at once, and $n$ is the dial between cheap-but-weak and expensive-but-strong. The asymptotic theory is reassuring: with decaying $c_k$ and $a_k$, the SPSA iterate converges almost surely to a stationary point under the same smoothness conditions as the $2D$-query finite-difference scheme, and reaches the same per-iteration mean-squared error while using a factor of $D$ fewer evaluations — the per-step misdirections average out across iterations, so the path tracks the true descent path even though no single step follows it.

With $\bar g$ in hand I am back in Robbins–Monro, but the plain update $x' = x - a\,\bar g$ uses one global step for an estimate whose coordinates are unevenly scaled and noisy. The right tool is Adam: keep per-coordinate exponential moving averages of the estimate and its square and step by their ratio $\hat m/(\sqrt{\hat v}+\epsilon)$. The first moment smooths noise across steps — a second layer of averaging on top of the within-step averaging — and the $1/\sqrt{\hat v}$ rescaling gives every pixel its own effective step so a few large-derivative pixels do not dominate. Feeding $\bar g$ to Adam as if it were a true gradient works because the estimator is unbiased enough that Adam cannot tell the difference, and Adam's robustness to noisy, unevenly-scaled gradients is exactly what this directional estimate needs; it converges faster here than a raw step. I keep the descent variable as the perturbation $dx$ rather than the image, because the constraint lives on $dx$: after each Adam step I clamp $dx$ componentwise to $[-\epsilon,\epsilon]$ (the Euclidean projection onto the $L_\infty$ box is per-coordinate clipping, since the ball is a box) and clamp $x_0+dx$ into $[0,1]$, folding the feasible result back into $dx$. The probe radius $c$ trades Taylor bias against the noise floor: smaller $c$ shrinks the $O(c^2)$ bias, but the signal in $df$ scales with $c$ while the model's own noise is a fixed floor, so the per-probe signal-to-noise degrades like $c$ as $c\to 0$. A value around $0.01$ on $[0,1]$-scaled pixels sits in the sweet spot — well inside the $\epsilon$-ball, large enough to produce a measurable $df$ — and a modest Adam step of about $0.01$ keeps the noisy direction from yanking the iterate out of the productive region. In code the margin computed is $M = \max_{\text{wrong}} - \text{true}$, which is positive once the attack succeeds; for an untargeted attack Adam minimizes its negative, which is precisely $J$. The result is a gradient-free $L_\infty$ attack whose cost is independent of the input dimension — the trustworthy adversary needed to tell real robustness from hidden vulnerability.

```python
import torch
from torch.nn.modules.loss import _Loss


class MarginalLoss(_Loss):
    def forward(self, logits, targets):
        top_logits, top_classes = torch.topk(logits, 2, dim=-1)
        target_logits = logits[torch.arange(logits.shape[0]), targets]
        max_nontarget_logits = torch.where(
            top_classes[..., 0] == targets, top_logits[..., 1], top_logits[..., 0],
        )
        loss = max_nontarget_logits - target_logits
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        raise ValueError("unknown reduction: '%s'" % (self.reduction,))


class SPSA:
    def __init__(self, model, eps=0.3, delta=0.01, lr=0.01,
                 nb_iter=1, nb_sample=128, max_batch_size=64):
        self.model = model
        self.eps, self.delta, self.lr = eps, delta, lr
        self.nb_iter, self.nb_sample = nb_iter, nb_sample
        self.max_batch_size = max_batch_size
        self.loss_fn = MarginalLoss(reduction="none")
        self.targeted = False

    def loss(self, logits, y):
        m = self.loss_fn(logits, y)
        return m if self.targeted else -m

    def linf_clamp_(self, dx, x, eps):
        dx_clamped = torch.clamp(dx, min=-eps, max=eps)
        x_adv = torch.clamp(x + dx_clamped, min=0.0, max=1.0)
        dx += x_adv - x - dx
        return dx

    def _batch_sizes(self, n, mbs):
        b = [mbs] * (n // mbs)
        if n % mbs:
            b.append(n % mbs)
        return b

    @torch.no_grad()
    def spsa_grad(self, images, labels, delta, nb_sample, max_batch_size):
        grad = torch.zeros_like(images)
        images = images.unsqueeze(0)
        labels = labels.unsqueeze(0)

        def f(xv, yv):
            return self.loss(self.model(xv), yv)

        images = images.expand(max_batch_size, *images.shape[1:]).contiguous()
        labels = labels.expand(max_batch_size, *labels.shape[1:]).contiguous()
        v = torch.empty_like(images[:, :1, ...])               # one v per pixel
        for bs in self._batch_sizes(nb_sample, max_batch_size):
            x_, y_ = images[:bs], labels[:bs]
            vb = v[:bs].bernoulli_().mul_(2.0).sub_(1.0)        # Rademacher +-1
            v_ = vb.expand_as(x_).contiguous()                 # shared across channels
            x_shape = x_.shape
            x_ = x_.reshape(-1, *images.shape[2:])
            y_ = y_.reshape(-1, *labels.shape[2:])
            v_ = v_.reshape(-1, *v.shape[2:])
            df = f(x_ + delta * v_, y_) - f(x_ - delta * v_, y_)
            df = df.view(-1, *([1] * (v_.dim() - 1)))
            grad_ = (df / (2.0 * delta * v_)).view(x_shape).sum(dim=0)
            grad += grad_
        grad /= nb_sample
        return grad

    def perturb(self, x, y):
        dx = torch.zeros_like(x)
        dx.grad = torch.zeros_like(dx)
        optimizer = torch.optim.Adam([dx], lr=self.lr)
        for _ in range(self.nb_iter):
            optimizer.zero_grad()
            dx.grad = self.spsa_grad(x + dx, y, self.delta,
                                     self.nb_sample, self.max_batch_size)
            optimizer.step()
            dx = self.linf_clamp_(dx, x, self.eps)
        return x + dx
```
