We want to craft an adversarial example $x^*$ that lies close to a legitimate input $x$ — within an $L_\infty$ budget $\|x^* - x\|_\infty \le \varepsilon$ so no pixel moves perceptibly — yet is misclassified by a trained network $f$. The catch is that we care about two figures of merit at once: the white-box success rate, where the example is crafted on and evaluated against the same model, and the transfer success rate, where the example is crafted on a surrogate and must still fool a held-out black-box target. The annoyance that starts everything is that the two attacks I have each handle exactly the half of the job the other botches. FGSM linearizes the loss around $x$, $J(x+\eta,y)\approx J(x,y)+\eta\cdot\nabla_x J$, and over the box $\|\eta\|_\infty\le\varepsilon$ that linear functional is maximized coordinate-by-coordinate at $\eta=\varepsilon\,\mathrm{sign}(\nabla_x J)$ — each pixel goes to its extreme $\pm\varepsilon$ in whichever direction raises the loss. One backprop, done, and it transfers decently because it commits a single large move along the coarse direction every coordinate's sign agrees on, a direction different models tend to share. But it is a weak white-box adversary: the tangent-plane approximation is only good for a tiny step, so at the $\varepsilon$ we actually use, $x+\varepsilon\,\mathrm{sign}(\nabla_x J)$ lands where the true loss has long since stopped resembling its linearization, and the budget is under-used. FGSM underfits the model. The standard fix, I-FGSM, takes many small clipped steps recomputing the gradient each time, $x^*_{t+1}=\mathrm{Clip}_{x,\varepsilon}\{x^*_t+\alpha\,\mathrm{sign}(\nabla_x J(x^*_t,y))\}$, so it climbs the real curved surface and becomes a strong white-box adversary — but its examples transfer poorly, and, tellingly, worse the more iterations it runs. More optimization, less generalization to other models: that is the signature of overfitting.

So the trade-off is one phenomenon, not two unrelated weaknesses, and seeing why tells me how to beat both ends at once. Different models trained on the same task learn decision boundaries that are aligned around a data point — close enough that crossing one model's boundary often crosses another's, which is the whole basis of transfer — but aligned is not identical. These networks are wildly non-linear, so around any point each model has its own idiosyncratic regions, little pockets ("holes," sharp local maxima of the loss) where it misbehaves and the others do not. I-FGSM greedily chases the sign of this surrogate's gradient at every step; on a curved, bumpy surface that per-step direction jitters, the consecutive perturbation increments zig-zag rather than pull consistently one way, and a greedy zig-zagging ascent dives straight into the nearest sharp local maximum — which, being sharp and private, is one of the surrogate's holes rather than a feature of the shared geometry. The iterative attack overfits by construction: it optimizes the surrogate's loss so aggressively that it exploits exactly the parts that do not generalize. FGSM escapes that trap only by being too crude to fall in. What I want, then, is to keep the iterative scheme (following the real surface is what gives white-box strength) but stop it from diving into the surrogate-specific holes — settle along the direction that stays consistent across iterations, which is plausibly the part shared with other models, and coast through the little sharp maxima instead of getting pinned.

The key recognition is that I have been mislabeling this as "perturbation engineering" when it is plainly an optimization: I-FGSM is sign-gradient ascent on $J$, and "greedy ascent zig-zagging into poor local optima on a noisy surface" is the oldest pathology in optimization with an equally old remedy — momentum. So I propose MI-FGSM, the Momentum Iterative Fast Gradient Sign Method: maintain a velocity $g$, accumulate the gradient into it each step, and take the pixel step along $g$ rather than along the bare current gradient. Two details have to be derived, not guessed: what goes into the accumulation, and how $g$ becomes a step. The step is forced. The budget is $L_\infty$ and the $L_\infty$-optimal move for a given direction is its sign — each nonzero coordinate goes $\pm\alpha$, none moves more than $\alpha$ — so the update is $x^*_{t+1}=x^*_t+\alpha\,\mathrm{sign}(g_{t+1})$. This is the same reason FGSM and I-FGSM use the sign, and it buys explicit per-step budget control: with $\alpha=\varepsilon/T$, $T$ agreeing sign-steps exactly fill a coordinate's budget, and I clip into the $\varepsilon$-ball anyway to absorb the steps that reverse. Running a generic optimizer like Adam on the objective instead would give no clean handle on the $L_\infty$ distance — I would be perpetually projecting — so the sign step is what preserves explicit budget control while I add momentum.

The accumulation needs more care, because the naive $g_{t+1}=\mu\,g_t+\nabla_x J(x^*_t,y)$ quietly wrecks the momentum. The magnitude of the input gradient is not stable across iterations — large far from a boundary, shrinking near one, occasionally spiking — so if I dump the raw gradient into $g$, the iteration with the biggest-magnitude gradient dominates the running sum and a small one contributes nearly nothing. That is not an average of directions; it is the very magnitude-noise I am trying to smooth, sneaking back in. Momentum should be an average over time with each step getting a fair vote, so I normalize the current gradient before accumulating:
$$g_{t+1} = \mu\cdot g_t + \frac{\nabla_x J(x^*_t,y)}{\|\nabla_x J(x^*_t,y)\|_1}.$$
The $L_1$ norm is the natural scale — the total absolute mass over all pixels — and dividing by it makes each iteration a unit-mass vector, so $\mu$ becomes a clean weight between accumulated history and present direction rather than between whatever raw magnitudes the surface handed me. The particular norm is not sacred; what is load-bearing is normalizing per iteration so the running average is over directions and every step votes equally. It also composes correctly with the sign step: since I take $\mathrm{sign}(g_{t+1})$, only the pattern of relative magnitudes across coordinates within $g_{t+1}$ matters, and the per-iteration normalization keeps old and new gradients comparable so neither a stale big-gradient step nor a fresh one can unilaterally flip the accumulated direction.

The loop is then $\alpha=\varepsilon/T$, $g_0=0$, $x^*_0=x$, and for each $t$: compute $\mathrm{grad}=\nabla_x J(x^*_t,y)$, set $g_{t+1}=\mu\,g_t+\mathrm{grad}/\|\mathrm{grad}\|_1$, step $x^*_{t+1}=x^*_t+\alpha\,\mathrm{sign}(g_{t+1})$, then clip into the $\varepsilon$-ball and the valid pixel range. The test that this is the right generalization rather than a third unrelated trick is that the known methods fall out by turning the knob. Set $\mu=0$: $g_{t+1}=\mathrm{grad}/\|\mathrm{grad}\|_1$ and its sign is just $\mathrm{sign}(\mathrm{grad})$, recovering I-FGSM exactly because the normalization is annihilated by the sign — so I have added a knob on top of the strong white-box case without breaking it. Take $T=1$ with $g_0=0$: $x^*_1=x+\varepsilon\,\mathrm{sign}(\mathrm{grad}/\|\mathrm{grad}\|_1)=x+\varepsilon\,\mathrm{sign}(\nabla_x J)$, which is FGSM. The family contains both ancestors and interpolates between them. For the decay I choose $\mu=1$ as the default, because then $g_{t+1}=g_t+\mathrm{grad}/\|\mathrm{grad}\|_1$ simply sums all the normalized gradients seen so far with equal weight — maximal undiscounted accumulation of the consensus direction, no decay throwing away history and no blow-up of any single step since each addend is unit-mass; too small a $\mu$ slides back toward the greedy zig-zag, too large over-weights stale gradients from points I have left, and $\mu=1$ gives $g_t$ the clean meaning of the running sum of all past unit-normalized directions.

The mechanism end to end is the reason to trust it beyond "momentum is generically good." The accumulated $g$ is an average of directions across many points on the trajectory; the components arising from a sharp private local maximum flip and cancel in that average, while the component pointing toward the shared boundary survives and accumulates, so the $\mathrm{sign}(g)$ step points predominantly along the direction the models agree on and the built-up velocity carries the iterate through the sharp maxima instead of being pinned. The result is a perturbation aligned with the shared boundary — exactly the kind that transfers — without giving up white-box strength, since it is still iterative sign-gradient ascent on the real surface. The trade-off is alleviated, not merely slid along. The same recipe generalizes once the accumulator is separated from the budget geometry: the only thing tying me to $L_\infty$ was the sign, and for an $L_2$ budget Cauchy-Schwarz makes the best fixed-length step $g_{t+1}/\|g_{t+1}\|_2$, giving $x^*_{t+1}=x^*_t+\alpha\,g_{t+1}/\|g_{t+1}\|_2$; targeted attacks flip the objective, minimizing $J(x^*,y^*)$ and stepping in the negative budget-optimal direction. And the holes argument practically demands attacking an ensemble: if one perturbation must fool several models at once it cannot afford any single model's private hole and is forced onto the shared, transferable direction, so I fuse the models in logits, $l(x)=\sum_k w_k\,l_k(x)$, and accumulate the gradient of cross-entropy on the fused logits — fusing pre-softmax keeps each model's fine-grained disagreements visible rather than washing them out. Crafting an adversarial example is, by this analogy, like training a model where transferability is generalization; momentum (a better optimizer) and the ensemble (more "training data") are exactly the two levers one reaches for to generalize better, and here they make the example transfer better. Committing the non-targeted $L_\infty$ method to the one empty slot in the harness:

```python
import torch
import torch.nn as nn


def run_attack(
    model: nn.Module,
    images: torch.Tensor,    # (N, C, H, W), values in [0, 1]
    labels: torch.Tensor,    # (N,)
    eps: float,              # L_inf budget
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    model.eval()
    steps = 10
    alpha = eps / steps      # alpha = eps / T
    decay = 1.0              # mu

    x = images.detach().to(device)
    labels = labels.detach().to(device)
    adv = x.clone().detach()
    momentum = torch.zeros_like(x)                   # g_0 = 0
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(steps):
        adv.requires_grad = True
        outputs = model(adv)
        cost = loss_fn(outputs, labels)              # maximize J(x*, y)
        grad = torch.autograd.grad(
            cost, adv, retain_graph=False, create_graph=False
        )[0]

        # mean-abs is proportional to the per-sample L1 norm in the formula; the fixed
        # factor preserves the same accumulated sign direction for fixed-size images.
        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        grad = grad + momentum * decay               # g_{t+1} = normalized grad + mu * g_t
        momentum = grad

        adv = adv.detach() + alpha * grad.sign()              # L_inf-optimal step
        delta = torch.clamp(adv - x, min=-eps, max=eps)        # project into eps-ball
        adv = torch.clamp(x + delta, min=0, max=1).detach()    # keep in [0, 1]

    return adv
```
