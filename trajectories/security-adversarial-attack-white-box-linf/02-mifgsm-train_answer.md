The single sign step landed where the linearization argument said it would, and the *shape* of the result is the whole story. FGSM scored 0.825 / 0.839 on ResNet20-C10/C100 and 0.878 / 0.825 on MobileNetV2-C10/C100 — a large majority flipped on the bottlenecked and depthwise architectures — but only 0.617 and 0.621 on VGG11BN-C10/C100. That split is diagnostic, not noise. One signed jump of size $\varepsilon$ in every coordinate maximizes the loss's *tangent plane* at the clean point, and over a full-budget jump that plane stops describing the true loss surface. Wherever the loss bends fast away from its linearization, the corner FGSM lands on is not where the loss is actually high — and VGG11-BN, whose wider, shallower low-resolution feature maps absorb small per-pixel perturbations more robustly, is exactly where that bending costs the most. So roughly 38% of VGG samples survived not because they are safe but because one linearized step could not reach them. The fix is forced by the diagnosis: stop trusting one linearization across the whole $\varepsilon$-box, and re-read the slope as I move.

I propose **MI-FGSM**, momentum iterative FGSM: take *small* sign steps, re-evaluate the gradient where I land, and step again, but accumulate the steps into a momentum-smoothed direction rather than chasing the raw local slope greedily. The iterative skeleton is the obvious correction — with step size $\alpha$, $x_{t+1} = x_t + \alpha\,\mathrm{sign}(\nabla_x J(\theta, x_t, y))$, projecting back into the budget each step — and each step is still the $L_\infty$-steepest move, the sign of the gradient, for exactly the Hölder reason FGSM used, only now read from the *current* point so I follow the curved surface. But before running plain iterative FGSM I want to pre-empt a known pathology of greedy gradient ascent. The loss surface around a point is bumpy, the raw input-gradient jitters from step to step, and a greedy sign-ascent that chases whatever the slope says *right now* zig-zags — consecutive increments do not pull consistently one way — which dives into the nearest sharp local maximum and stalls there, short of the high-loss region a smoother trajectory would reach. In the classic transfer setting those sharp maxima are one model's private quirks; here I attack and am graded by the same architecture so transfer is not the headline, but the *optimization* pathology is identical and still wastes steps oscillating.

The cure, once the attack is named correctly as plain sign-gradient ascent on $J$, is the oldest remedy in optimization: momentum. I do not step on the raw gradient; I step on an accumulated velocity that averages the recent gradients, which cancels the components that flip sign step to step (the zig-zag) and reinforces the components that persist (the consistent ascent direction), with the built-up velocity carrying the iterate over small humps instead of letting each one trap it. Two details decide the method, and both are derived rather than guessed. The step is forced: the budget is $L_\infty$, so the optimal move for a given direction is the sign, $x_{t+1} = x_t + \alpha\,\mathrm{sign}(g_{t+1})$, with each coordinate moving at most $\alpha$. The accumulation is where the naive $g_{t+1} = \mu\, g_t + \nabla_x J(x_t, y)$ would quietly wreck the momentum: the input-gradient's *magnitude* is not stable across iterations — large far from a boundary, small near one, sometimes spiking — so dumping the raw gradient into $g$ lets a single big-magnitude step dominate the running sum and the "average" becomes whichever step happened to have the largest gradient, the very magnitude-noise I am trying to smooth. Momentum must be an average of *directions*, every step getting a fair vote, so I normalize the current gradient by its total absolute mass before accumulating:
$$g_{t+1} = \mu\, g_t + \frac{\nabla_x J(\theta, x_t, y)}{\lVert \nabla_x J(\theta, x_t, y)\rVert_1}.$$
Each iteration's contribution becomes a unit-mass vector, so $\mu$ is a clean trade-off between accumulated history and the present direction rather than between whatever raw magnitudes the surface handed me. In code this is dividing by the mean absolute value over the pixel dimensions, proportional to the per-sample $L_1$ norm; since I take $\mathrm{sign}(g_{t+1})$ downstream, only the *pattern of relative magnitudes across coordinates* survives, and the per-iteration normalization is what keeps a stale big-gradient step from unilaterally flipping the accumulated direction.

The decay is $\mu = 1$: the recursion $g_{t+1} = g_t + \nabla_x J / \lVert\nabla_x J\rVert_1$ simply *sums* all the normalized gradients seen so far with equal weight — undiscounted accumulation of the consensus direction, with no decay throwing away history and no blow-up of any single step since each addend is unit-mass. There is a clean consistency check that this is the right generalization rather than a third unrelated trick: setting $\mu = 0$ gives $g_{t+1} = \nabla_x J / \lVert\nabla_x J\rVert_1$, whose sign is just $\mathrm{sign}(\nabla_x J)$, so I recover iterative sign-FGSM exactly (the normalization is annihilated by the sign), and a single step from the clean point recovers FGSM. The family contains both ancestors and interpolates between them.

The step count, step size and start are where this fill diverges from the bare momentum recipe, and the divergence pushes the method into the strong-single-model-white-box regime rather than the transfer regime. The natural momentum-attack default ties $\alpha = \varepsilon/T$ so that $T$ aligned sign steps exactly fill the budget; I instead run $T = 40$ steps at $\alpha = \varepsilon/10$ and start from a *random* point in the $\varepsilon$-box rather than the clean image. With $\alpha = \varepsilon/10$ and 40 steps the total reach is $40\cdot\varepsilon/10 = 4\varepsilon$, well past the $2\varepsilon$ box diameter, so after the iterate is pinned to the high-loss boundary it spends its remaining steps *refining* its position there instead of merely crawling toward it — more gradient updates correcting stale-gradient error rather than trusting the first plane. The random start matters because the gradient evaluated exactly at the clean point is corrupted by sharp curvature artifacts localized at $x$; launching from a uniform perturbation in the $\varepsilon$-box, clipped to $[0,1]$, jumps off that brittle point before the first gradient read so the momentum accumulates from a better-behaved direction. So I should not import the transfer story as the *reason* this works: on a single-model white-box budget the operative reasons are that iterating recovers FGSM's underfit and that momentum plus a random start gives a steadier, less-trapped ascent than greedy iterative FGSM.

The diagnosis tells me to keep the expectation asymmetric. The ~82–88% FGSM already flipped on ResNet and MobileNetV2 are the easy samples where even the tangent plane reached the boundary; iterating cannot do worse, it only pins them harder, so I expect MI-FGSM to roughly match FGSM there (high-0.96–0.99 once iterated). The marginal value of momentum is concentrated on the survivors, the curved-surface samples, which are concentrated on VGG — so the real gain should be on the two VGG scenarios, lifting them substantially above 0.617 and 0.621 into the low-0.7s, recovering most but not all of the underfit (VGG's residual robustness to small $L_\infty$ is partly architectural, which no first-order iteration can touch).

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
    alpha = eps / 10.0
    decay = 1.0

    x = images.detach()
    x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    momentum = torch.zeros_like(x)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
        momentum = decay * momentum + grad

        with torch.no_grad():
            x_adv = x_adv + alpha * momentum.sign()
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + delta, 0.0, 1.0)

        x_adv = x_adv.detach()

    return x_adv
```
