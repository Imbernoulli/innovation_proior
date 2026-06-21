Iterating paid off where the diagnosis said it would. MI-FGSM lifted the easy scenarios to the ceiling — 0.981 / 0.984 on ResNet20-C10/C100, 0.999 on MobileNetV2-C10, 0.967 on MobileNetV2-C100 — confirming that the samples FGSM left there were curvature-underfit, not safe. On VGG it moved the needle, $0.617 \to 0.714$ and $0.621 \to 0.720$, but the numbers are still in the low 0.7s and the gain is smaller than the easy-scenario gains were dramatic. So either VGG's residual ~28% is architectural robustness no first-order attack can touch, or the *particular* iterative scheme is leaving white-box loss on the table — and there is a specific reason to suspect the latter. The momentum I added was justified by *transfer*: averaging the direction across steps keeps the component shared with a held-out model and discards one model's private quirks. But here attack and grader are the *same* architecture; there is no held-out boundary to align with. What I actually want is the strongest possible solution to the inner maximization $\max_{\lVert\delta\rVert_\infty \le \varepsilon} J(\theta, x+\delta, y)$ on this one model, and for that objective an accumulated velocity is not obviously helpful and can hurt: it is a running sum of *past* normalized gradients from points the iterate has already left, so near the high-loss boundary, where the surface bends, a stale momentum term can point the step partly along a direction that was good several iterations ago but is now wrong — overshooting past the local maximum or oscillating along the boundary instead of settling on it. Momentum trades faithful local ascent for direction stability; that is a good trade when stability buys transfer and a possibly-bad one when the only goal is to maximize *this* loss.

I propose **PGD**, projected gradient descent (here, ascent), derived from the inner objective rather than from a momentum patch. I want $\delta$ solving $\max_\delta L(\theta, x+\delta, y)$ subject to $\lVert\delta\rVert_\infty \le \varepsilon$ and $x+\delta \in [0,1]^d$ — a constrained maximization of the loss over a little box around $x$. There is no closed form ($L$ is a deep network, wildly non-concave in $x$), so I climb it as well as a gradient method can. The reason to climb it *well* is exactly the VGG number: a sloppy maximizer reports the model as more robust than it is, because the samples it fails to flip are not safe, they are out of its reach. The cheapest move, linearizing to $L(x+\delta) \approx L(x) + g^\top\delta$ with $g = \nabla_x L$, gives the corner $\delta = \varepsilon\,\mathrm{sign}(g)$ — that is FGSM, and it underfits. The honest correction is to take a *small* sign step, re-read the gradient where I land, and repeat:
$$x_{t+1} = x_t + \alpha\,\mathrm{sign}\big(\nabla_x L(\theta, x_t, y)\big),$$
following the curved surface instead of betting on one tangent plane. But the instant I iterate, the accumulated move can wander outside the $\varepsilon$-box and the iterate can leave $[0,1]$, so I must *project* back into the feasible set after every step — and I want to know the projection is exact, not a clamp-and-hope.

What makes PGD principled is that the projection is exact. The feasible set is the intersection of $\lVert x' - x\rVert_\infty \le \varepsilon$ and $x' \in [0,1]^d$. For the $L_\infty$ ball the Euclidean projection $\Pi(z) = \arg\min_{\lVert x'-x\rVert_\infty \le \varepsilon} \lVert x' - z\rVert_2$ decouples coordinate by coordinate, because both the squared distance and the constraint are per-coordinate: minimizing $(x'_i - z_i)^2$ subject to $x_i - \varepsilon \le x'_i \le x_i + \varepsilon$ returns $z_i$ if it is already inside the interval, else the nearer endpoint — which is exactly clamping $x'_i$ to $[x_i - \varepsilon,\, x_i + \varepsilon]$. So element-wise clipping of the perturbation to $[-\varepsilon, \varepsilon]$ *is* the Euclidean projection onto the $L_\infty$ box, not a heuristic. The pixel-box $[0,1]$ is the same kind of object, so its projection is clamping to $[0,1]$, and their intersection is the coordinate interval $[\max(x_i - \varepsilon, 0),\, \min(x_i + \varepsilon, 1)]$ — reached by clipping the perturbation to $[-\varepsilon, \varepsilon]$, adding it to $x$, then clamping the image to $[0,1]$. That is the same dual-clamp every rung has used, now justified as the exact projection rather than carried along by analogy.

Two design levers remain, and dropping the velocity is what lets both work cleanly. The step size and count: this task runs $\mathrm{steps} = 40$ with $\alpha = \varepsilon/4$ and no momentum. The requirement is that from any start inside the $\varepsilon$-ball the iterate can reach the opposite side and still have steps left to move *along* the boundary once pinned; the box diameter in any coordinate is $2\varepsilon$, so $\mathrm{steps}\cdot\alpha$ must comfortably exceed $2\varepsilon$. With $\alpha = \varepsilon/4$ and 40 steps the total reach is $10\varepsilon$, five times the diameter, so the iterate easily crosses the box from any interior start and spends most of its steps refining on the high-loss boundary. The larger per-step $\varepsilon/4$ (versus MI-FGSM's $\varepsilon/10$) is affordable *precisely because* there is no momentum to overshoot with: each step is a faithful local sign-ascent of the current loss, re-corrected the next step, so a bigger step explores the box faster without a stale velocity carrying it past the maximum. The start: I keep the random launch, and the reasoning is sharper than "dodge the corrupted gradient." The loss surface right next to a data point has sharp curvature artifacts localized at $x$, so the steepest direction *at $x$* is a poor guide to where the loss climbs further out; and always launching from $x$ commits the whole trajectory to that one initial direction and climbs into whatever single maximum sits in the basin containing $x$ — and since the inner problem is non-concave, there are many maxima scattered across the box I would never see. Sampling each coordinate uniform in $[-\varepsilon, \varepsilon]$, setting $x_0 = x + u$, and clipping into $[0,1]$ jumps off the non-smooth point before the first gradient read, so the first trusted gradient is evaluated somewhere better-behaved and the trajectory can land in a different, possibly higher, basin — at no extra gradient cost.

That non-concavity is the one threat to calling this the strongest first-order attack, and the honest answer is empirical: run the projected iteration from many random starts and watch the final-iterate loss. The useful pattern is that the loss climbs consistently from each start and plateaus quickly, and across restarts the plateau values *concentrate* — many genuinely distinct maxima reaching almost the same loss. If that is the landscape, formal non-concavity is not the practical blocker it first appears, and a first-order adversary already using input-gradients from several starts has little evidence of a significantly better point to chase. The loss itself is a pluggable choice — cross-entropy is the simplest default and the one this fill uses, with a logit-margin loss the drop-in when I want to push on the decision boundary directly — but the iteration is the same. And Danskin's theorem closes the loop: the negative loss-gradient at the inner maximizer is, in the tie-free exact-max case, a descent direction for the saddle-point objective a robust-training defense would solve, so the same maximizer I am building is exactly what such a defense would call in its inner loop — which is why "solve the inner max well" is the principled target, not a heuristic. So I drop the momentum, keep the random start, run 40 plain sign-ascent steps at $\alpha = \varepsilon/4$, and project each step. I expect a dead heat with MI-FGSM on the saturated scenarios and, if the stale velocity really was blunting the maximizer, VGG to tick modestly *above* 0.714 and 0.720 into the low 0.72s — which would separate "weak maximizer" from "robust architecture," the question the VGG number has been asking since FGSM.

```python
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    import torch.nn.functional as F

    _ = (device, n_classes)
    model.eval()
    steps = 40
    alpha = eps / 4.0

    x = images.detach()
    x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + delta, 0.0, 1.0)

        x_adv = x_adv.detach()

    return x_adv
```
