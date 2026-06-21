I have a fixed classifier $f_\theta$ with full white-box access — I can backpropagate and read off the input-gradient $\nabla_x L(\theta, x, y)$ of any differentiable loss — and I am handed a single image $x \in [0,1]^d$ that the model already classifies correctly, together with a per-pixel budget $\varepsilon$. I want to move $x$ so the model gets it wrong while changing each pixel by at most $\varepsilon$ and keeping the result a valid image. Written cleanly, I want $\delta$ solving

$$\max_{\delta}\; L(\theta, x + \delta, y)\quad\text{subject to}\quad \|\delta\|_\infty \le \varepsilon,\; x + \delta \in [0,1]^d.$$

This is a constrained maximization of the loss over a small box around $x$, and there is no closed form: $L$ is a deep network, wildly non-concave in $x$. So I will not solve it exactly; I will climb it as well as a gradient method can. The reason I care to climb it *well* rather than merely "get one misclassification" is that the strength of this maximizer is the thing being measured. At a tight budget a sloppy attack reports the model as more robust than it is, because the examples it fails to flip are not actually safe — they are just out of its reach. The existing options each fall short of this. Box-constrained L-BFGS finds small perturbations reliably but runs a full quasi-Newton inner loop per example, far too expensive to call at scale or inside a defense's training step. FGSM linearizes once around $x$ and takes a single full-budget sign step; it is cheap, but the linearization is only exact infinitesimally and a full $\varepsilon$ jump lands at a box corner that need not be near the true maximizer, leaving loss on the table. The basic iterative method (BIM) fixes part of this by taking many small sign steps, but launches deterministically from $x$ along a single trajectory, so if the launch gradient is distorted it inherits the mistake. R+FGSM prepends a random kick to escape that distorted point but is still one linearized step afterward. What I want is a single fixed-budget $L_\infty$ procedure that follows the curved surface, stays feasible, and does not bet everything on the gradient read at the suspicious clean point.

I propose PGD — Projected Gradient Descent — projected gradient *ascent* on the loss, equivalently projected gradient descent on the negative loss, which is where the name comes from. The first load-bearing step is to solve the one subproblem that *does* have a closed form: maximizing the linearized loss $g^\top \delta$ with $g = \nabla_x L$ over the box $\|\delta\|_\infty \le \varepsilon$. The constraint $|\delta_i| \le \varepsilon$ decouples across coordinates, so I maximize each $g_i \delta_i$ independently by pushing $\delta_i$ to the boundary in the sign of $g_i$, giving $\delta = \varepsilon\,\mathrm{sign}(g)$ with gain $\varepsilon \sum_i |g_i| = \varepsilon \|g\|_1$. This is why the step is the *sign* of the gradient and not the raw gradient: it is steepest ascent with respect to the $L_\infty$ geometry. Under an $L_2$ budget the same maximization would instead give $\delta = \varepsilon\,g/\|g\|_2$ — the normalized raw gradient — so the norm in the constraint dictates the shape of the step. That single full-budget step is exactly FGSM, and its defect is plain: I maximized the *linearization*, not the loss, then took $\delta$ as large as the budget allows, where the tangent plane is essentially fiction. The gradient I trusted was the gradient at $x$; the moment I move it goes stale.

So I take a small step of size $\alpha$ in the sign direction, re-evaluate the gradient where I land, and repeat, replacing one big leap by many little steepest-ascent steps that each re-read the local slope. But the instant I iterate, the accumulated move can wander outside the $\varepsilon$-box and leave $[0,1]$. I need to put the iterate back into the feasible set after every step, which is what projection is for. The feasible set is the $L_\infty$ ball $\|x' - x\|_\infty \le \varepsilon$ intersected with $[0,1]^d$, and I want to know the projection is *exactly right*, not a convenient hack. The Euclidean projection $\Pi(z) = \arg\min_{\|x'-x\|_\infty \le \varepsilon} \|x' - z\|_2$ decouples across coordinates because both the squared distance and the constraint do; each 1-D problem is the squared distance from $z_i$ to the interval $[x_i - \varepsilon, x_i + \varepsilon]$, minimized at $z_i$ itself if inside and at the nearest endpoint otherwise. That is precisely coordinate-wise clipping — projection onto the $L_\infty$ box *is* clipping, not an approximation of it. The pixel box $[0,1]$ is the same kind of object, so its projection is clamping to $[0,1]$, and the intersection projects coordinate $i$ onto $[\max(x_i - \varepsilon, 0),\, \min(x_i + \varepsilon, 1)]$. Since these are same-axis boxes, I implement it by clipping the perturbation $\delta = x' - x$ to $[-\varepsilon, \varepsilon]$ and then clipping $x + \delta$ to $[0,1]$; the two clips collapse to that single interval clamp. The iteration is therefore

$$x_{t+1} = \Pi_C\!\left( x_t + \alpha\,\mathrm{sign}\big(\nabla_x L(\theta, x_t, y)\big) \right),\qquad C = \{z : \|z - x\|_\infty \le \varepsilon,\; z \in [0,1]^d\},$$

the multi-step generalization of the sign attack with every iterate kept feasible.

The step size $\alpha$ and step count are attack-budget knobs, not part of the definition, and the choice has a clear rationale at both extremes. If $\alpha$ is huge I overshoot the box on every step and the projection just slams me onto the boundary — back to a one-step corner attack with extra cost. If $\alpha$ is tiny each step is faithful but I need many of them to traverse the ball. The useful reach calculation is that for a random start to be able to cross the box in a coordinate, the signed path length $\text{steps} \cdot \alpha$ should exceed the coordinate diameter $2\varepsilon$ with some slack; one thorough setting is $\alpha = 2.5\,\varepsilon/\text{steps}$ over 100 steps. Dataset scale sets the concrete numbers: 40 steps of size $0.01$ for MNIST at $\varepsilon = 0.3$, or steps of size $2$ with $\varepsilon = 8$ in 0–255 pixel units for CIFAR; on $[0,1]$ inputs a CIFAR-style default of $\varepsilon = 8/255$ with $\alpha = \varepsilon/4 = 2/255$ matches that scale.

There remains the question of *where I start*. Starting deterministically at $x_0 = x$ means the very first gradient I trust is $\nabla_x L$ at the clean point, and that gradient can be brittle: the loss surface right next to a data point can carry sharp local curvature artifacts that mask the true ascent direction, so the steepest direction at $x$ is a poor guide to where the loss climbs a little further out. Always launching from $x$ commits the whole trajectory to that one possibly distorted direction and to whatever local maximum sits in the basin containing $x$ — and since the inner problem is non-concave there can be many basins. The fix is to not start at $x$: sample each coordinate of $u$ uniformly in $[-\varepsilon, \varepsilon]$, set $x_0 = \mathrm{clip}(x + u, 0, 1)$, and run the projected iteration from there. The random kick jumps off the non-smooth point before the first gradient read, so that gradient is evaluated somewhere better-behaved, at no extra gradient cost; if I can afford several launches I keep the highest-final-loss candidate and sample more than one basin. Probing the landscape this way shows a specific and reassuring pattern: the loss rises consistently from each random start and plateaus quickly, the plateau values concentrate tightly across many restarts with no extreme outliers, and yet the maxima themselves are distinct, nearly orthogonal points in the box — many local maxima of nearly equal loss. This does not prove no isolated higher maximum exists, but it is exactly the evidence that a first-order adversary has little better to chase, which is what makes random-start projected iteration the strongest practical first-order target.

This construction strictly contains the prior art as degenerate cases: FGSM is one step of size $\varepsilon$ from $x_0 = x$; BIM is the deterministic-start, single-trajectory case with the same sign step and coordinate clips; R+FGSM is a single step after one random pre-step, whereas PGD folds the random start into the full iteration. The loss $L$ is itself a pluggable lever. Cross-entropy is the default — the simplest scalar to backpropagate — but a classifier's actual failure condition is a logit comparison, so the iteration also accepts a Carlini-Wagner-style logit hinge $-\mathrm{relu}(z_y - \max_{j \ne y} z_j + \kappa)$, which grows the wrong-vs-true margin until a confidence threshold $\kappa$ is reached; only the differentiated scalar changes, the iteration is unchanged. Finally, this is the principled inner-max target because of Danskin's theorem, which closes the loop with robust training $\min_\theta \mathbb{E}[\max_{\delta \in S} L(\theta, x+\delta, y)]$. With $\phi(\theta) = \max_{\delta \in S} g(\theta, \delta)$ and $S$ compact, if each $g(\cdot, \delta)$ is differentiable with continuous $\nabla_\theta g$, then $\phi$ is locally Lipschitz and directionally differentiable with $\phi'(\theta; d) = \sup_{\delta \in S^*(\theta)} d^\top \nabla_\theta g(\theta, \delta)$, and at a unique maximizer $\nabla\phi(\theta) = \nabla_\theta g(\theta, \delta^*)$. Getting the sign right matters: with $h = \nabla_\theta g(\theta, \delta^*)$, the direction $+h$ gives $\phi'(\theta; h) = \|h\|_2^2 > 0$ (ascent), so the descent direction the defender steps along is $-h$, with $\phi'(\theta; -h) = -\|h\|_2^2 < 0$. When several maximizers tie, the safe choice is the nonzero minimum-norm point $p$ of the convex hull of active gradients, where projection geometry gives $p^\top a \ge \|p\|_2^2$ and hence $\phi'(\theta; -p) \le -\|p\|_2^2 < 0$; the tie-free case collapses back to the single-gradient rule. So finding the worst-case $\delta$ with PGD and backpropagating the loss at $x + \delta$ yields exactly the parameter-gradient robust training steps against — attack and defense are two halves of one min-max problem. The whole attack is a few tens of gradient-sign-and-clip steps from a random start:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def run_attack(
    model: nn.Module,
    images: torch.Tensor,   # (N, C, H, W) in [0, 1]
    labels: torch.Tensor,   # (N,)
    eps: float,             # L_inf budget
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    _ = n_classes
    model.eval()
    steps = 40
    alpha = eps / 4.0                       # CIFAR-style default: eps=8/255 -> step size 2/255

    x = images.clone().detach().to(device)       # ball is centered at the clean image
    labels = labels.clone().detach().to(device)

    # random-start perturbation, projected into [0,1]
    x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), labels)        # ascend this loss (pluggable)
        grad = torch.autograd.grad(loss, x_adv)[0]          # white-box input-gradient

        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()             # L_inf steepest-ascent step
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)   # project onto L_inf ball
            x_adv = torch.clamp(x + delta, 0.0, 1.0)            # project onto the box intersection
        x_adv = x_adv.detach()

    return x_adv
```
