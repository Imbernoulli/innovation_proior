Let me start from the thing that actually annoys me, which is that I have two attacks and each is good at exactly the half of the job the other is bad at. I can compute the gradient of the loss with respect to the input — one backprop, the same machinery I use to train, just pointed at the pixels — so crafting an adversarial example is, in principle, just maximizing J(x*, y) over the little box ||x* - x||_inf <= eps. The cleanest thing to do with that gradient is what FGSM does: linearize, J(x + η, y) ≈ J(x, y) + η·∇_x J, and over the box ||η||_inf <= eps the linear functional η·∇_x J is maximized coordinate by coordinate at η = eps·sign(∇_x J), because each pixel independently wants to go to its extreme ±eps in whatever direction raises the loss. One shot, done. And it transfers decently to other models — I think because it commits one big move along the direction every coordinate's sign agrees on, a coarse direction that different models tend to share. But it's a weak white-box adversary, and I know why: the linear approximation is only any good for a tiny step. At the eps I actually care about, x + eps·sign(g) lands somewhere the true loss has long since stopped looking like its tangent plane, so I've under-used my budget. FGSM underfits the model.

The fix everyone reaches for is to stop trusting the linearization over the whole box and instead take many small steps, recomputing the gradient each time so I'm always following the *local* slope. That's I-FGSM: x*_0 = x, and x*_{t+1} = Clip_{x,eps}{ x*_t + α·sign(∇_x J(x*_t, y)) }, with α small, say ε/T, and a clip back into the eps-ball after every step so I never leave the budget. This is a genuinely strong white-box adversary because it climbs the real curved loss surface instead of trusting one tangent plane. But now the *other* half of the job falls apart: the examples it produces don't transfer. And the more iterations I run, the worse the transfer gets, which is a strange and telling signal — more optimization, less generalization to other models. That's the exact shape of overfitting.

So I have a trade-off staring at me: one-step transfers but underfits; iterative fits but overfits and won't transfer. Before I try to beat it I want to understand *why* it's a trade-off and not just two unrelated weaknesses, because if it's really one phenomenon I might be able to fix both ends at once.

Different models trained on the same task learn decision boundaries that are *aligned* around a data point — close enough that a perturbation crossing one model's boundary often crosses another's. That's the whole basis of black-box transfer. But "aligned" is not "identical." These networks are wildly non-linear, so around any given point each model has its own idiosyncratic regions — call them holes — little pockets where this particular model misbehaves and the others don't. Now reread I-FGSM in that light. It greedily chases the sign of *this* surrogate's gradient at every step. The per-step direction is whatever the local slope says right now, and on a curved, bumpy surface that direction jitters from step to step. The cosine between consecutive perturbation increments is the diagnostic I should care about: an unstable iterative attack will have updates that are not pulling consistently one way, but zig-zagging. And a greedy zig-zagging ascent is exactly the thing that finds and dives into the nearest sharp local maximum of the loss — which, because it's sharp and local, is very likely one of the surrogate's private holes rather than a feature of the shared boundary geometry. So the iterative attack overfits *by construction*: it optimizes the surrogate's loss so aggressively that it ends up exploiting precisely the parts of the surrogate that don't generalize. One-step FGSM avoids that trap only by being too crude to fall into it. The trade-off is one phenomenon — fitting the surrogate's idiosyncrasy — seen from two sides.

That reframing tells me what I want: keep the iterative scheme, because following the real curved surface is what makes it a strong white-box adversary, but stop it from diving into the surrogate-specific holes. I want it to settle along the direction that's *consistent* across iterations — the part of the gradient that keeps pointing the same way, which is plausibly the part shared with other models — and to coast through the little sharp maxima instead of getting pinned in them.

Now I notice I've been describing the attack the wrong way in my head, as "perturbation engineering," when it is literally an optimization: I-FGSM is sign-gradient ascent on J. And the pathology I just named — greedy gradient ascent on a noisy, bumpy surface zig-zagging into poor local optima — is the oldest pathology in optimization, with an equally old remedy. Momentum. Polyak's trick: don't step on the raw gradient, step on an accumulated velocity g_{t+1} = μ·g_t + (current gradient), which averages the recent gradients. The averaging cancels the components that flip sign step to step (the zig-zag) and reinforces the components that persist (the consistent direction), and the built-up velocity carries you over small humps and out of shallow local optima instead of letting each one trap you. Sutskever and colleagues leaned on exactly this to stabilize SGD. Everything I just said I wanted — stabilize the direction, keep the consistent part, escape the sharp holes — is what momentum *does*. So let me put a velocity into the iterative attack.

The skeleton, then: maintain g, accumulate the gradient into it each step, and take the actual pixel step along g instead of along the bare current gradient. g_0 = 0, x*_0 = x, and at each step compute the input gradient at the current point, fold it into g with decay μ, then move. The question is the two details — what exactly goes into the accumulation, and how exactly I turn g into a step — and I want to derive both, not guess them.

Take the step first, because it's forced. My budget is L_inf and I already know the L_inf-optimal move for a given direction is the sign: each nonzero coordinate goes ±α and no coordinate moves by more than α. So the update is x*_{t+1} = x*_t + α·sign(g_{t+1}). This is the same reason FGSM and I-FGSM use sign — it's the max-norm-optimal direction — and it buys me something I really want operationally: with the sign, every step has explicit L_inf size control. Set α = ε/T and T agreeing nonzero sign steps fill the budget in a coordinate; I clip into the eps-ball anyway to be safe against the steps that disagree and reverse. Contrast this with just running a generic optimizer like Adam on the attack objective — it would give me no clean handle on the L_inf distance at all; I'd be perpetually projecting. The sign step is what keeps the explicit budget control while I add the momentum.

Now the accumulation — and here's where I have to be careful, because the naive g_{t+1} = μ·g_t + ∇_x J(x*_t, y) has a problem that would quietly wreck the momentum. The magnitude of the input gradient is not stable across iterations. Early on, far from any boundary, the gradient can be large; as I approach a boundary or a flat region it can shrink, or a single iteration can hit a spot where it spikes. If I dump the raw gradient into g, then an iteration with a big-magnitude gradient dominates the running sum and an iteration with a small one contributes almost nothing — the "average" is really just whichever step happened to have the largest gradient. That's not momentum; that's the same magnitude-noise I was trying to smooth out, sneaking back in through the back door. Momentum is supposed to be an average of *directions* over time, with each step getting a fair vote. So before I accumulate, I should normalize the current gradient to put every iteration on equal footing in magnitude:

g_{t+1} = μ·g_t + ∇_x J(x*_t, y) / ||∇_x J(x*_t, y)||_1.

The L1 norm is the natural scale here — it's the total absolute mass of the gradient over all pixels — and dividing by it makes each iteration's contribution a unit-mass vector, so μ is now a clean trade-off weight between accumulated history and the present direction rather than a weight between whatever raw magnitudes the surface happened to hand me. The particular norm isn't sacred — any scale measure would do the same job — what's load-bearing is that I normalize *per iteration* so the running average is over directions and every step votes equally. And it composes correctly with the sign step downstream: since I take sign(g_{t+1}) at the end, what matters is the *pattern of relative magnitudes across coordinates within* g_{t+1}, which is precisely what decides each coordinate's sign; the per-iteration L1 normalization keeps old and new gradients comparable so neither a stale big-gradient step nor a fresh one can unilaterally flip the accumulated direction.

Let me write the loop out and sanity-check it against the two degenerate cases, because a good generalization should contain the things it generalizes as special cases:

  α = ε/T;  g_0 = 0;  x*_0 = x
  for t = 0 … T-1:
    grad = ∇_x J(x*_t, y)
    g_{t+1} = μ·g_t + grad / ||grad||_1
    x*_{t+1} = x*_t + α·sign(g_{t+1})        # then clip into the eps-ball and valid range

Set μ = 0: g_{t+1} = grad/||grad||_1, and sign of that is just sign(grad) — so I recover I-FGSM exactly (the normalization is annihilated by the sign). Good, μ = 0 is vanilla iterative, so I haven't broken the strong-white-box case; I've added a knob on top of it. And one step, T = 1, with g_0 = 0 gives x*_1 = x + ε·sign(grad/||grad||_1) = x + ε·sign(∇_x J), which is FGSM. So the family I've written down contains both ancestors and interpolates between them with μ. That's the test of having found the right generalization rather than a third unrelated trick: turn the knob and the known methods fall out.

Now what does μ buy me between those extremes, and is there a right value? μ controls how much of the accumulated past direction I keep. Too small and I'm back to the greedy zig-zag (μ = 0 *is* the zig-zag). Too large and I over-weight stale gradients from points I've long since moved away from, and the direction stops being responsive to where I actually am now. The clean default is μ = 1, because the recursion g_{t+1} = g_t + grad/||grad||_1 just *sums* all the normalized gradients seen so far with equal weight — maximal, undiscounted accumulation of the consistent direction, with no decay throwing away history and no blow-up of any single step because each addend is unit-mass. So μ = 1 gives g_t a simple meaning: the running sum of all past unit-normalized gradient directions, the accumulated consensus direction.

Let me make sure I believe the mechanism end to end, because I want this to be more than "momentum is generically good." The iterative attack fails to transfer because it dives into the surrogate's private holes — sharp local maxima that are artifacts of one model's non-linearity, not of the shared boundary. Momentum's accumulated g is an average of directions across many points along the trajectory; the components of the gradient that come from a sharp local feature flip and cancel in that average, while the component that keeps pointing toward the shared boundary survives and accumulates. So the sign(g) step points predominantly along the *consistent* direction — the part of the geometry the models agree on — and the velocity carries the iterate through the sharp maxima instead of letting them pin it. The result is a perturbation aligned with the shared boundary, which is exactly the kind that transfers. And I haven't given up white-box strength: it's still iterative sign-gradient ascent climbing the real surface. That's the whole point — the same momentum that restores transfer keeps the white-box attack strong, so the trade-off I started with is *alleviated*, not just slid along.

The same structure carries over once I separate the accumulator from the geometry of the budget. The only thing tying me to L_inf was the sign() step. If the budget is L2, Cauchy-Schwarz says the fixed-length step with the largest linearized gain in direction g is g/||g||_2, so the same accumulated g drives x*_{t+1} = x*_t + α·g_{t+1}/||g_{t+1}||_2. Each such step has L2 length α, and with α = ε/T the whole path has length at most ε before any valid-range clipping. And targeted attacks just flip the objective: to force a chosen label y* I *minimize* J(x*, y*), so I accumulate the gradient of J(x*, y*) and step in the negative budget-optimal direction, x*_{t+1} = x*_t − α·sign(g_{t+1}) for L_inf, or x*_{t+1} = x*_t − α·g_{t+1}/||g_{t+1}||_2 for L2. So "accumulate a normalized-gradient velocity, then take the budget-optimal step along it" is a recipe, not a single attack — any iterative gradient attack becomes its momentum variant by swapping the current gradient for the accumulated one.

There's one more lever the transferability picture hands me, and it's worth pulling because the holes argument practically demands it. If a single perturbation has to be adversarial for *several* models at once, it can't afford to exploit any one model's private hole — it's forced onto the direction the models share, which is the very thing that transfers to a held-out model. So attack an ensemble. The cleanest way to combine K models is to fuse their *logits*, l(x) = Σ_k w_k·l_k(x), and take the cross-entropy on the fused logits as the loss whose gradient I accumulate. Fusing in logits rather than in probabilities or in the per-model losses keeps the fine-grained pre-softmax disagreements visible — the logits carry the un-normalized log-relationships, so a direction that fools the *aggregate* must contend with every model's detailed response, not a washed-out average of confidences. That's the variant I'd expect to transfer best, and it slots straight into the same loop: only the loss whose gradient I take changes. There's even a tidy way to think about all of this at once — crafting an adversarial example is like *training a model*, where transferability plays the role of generalization. Then momentum (a better optimizer) and attacking an ensemble (more "training data," i.e. more models) are exactly the two things you'd reach for to make a trained model generalize better, and here they make the adversarial example generalize — transfer — better. The analogy isn't a proof, but it's why I trust that these two additions are pulling in the same direction.

So let me commit the method to the code I'd actually run, filling the one empty slot in the harness — the crafting procedure. I keep the iterative sign-ascent skeleton, add the accumulated normalized-gradient velocity, take the L_inf-optimal sign step, and project back into the eps-ball and the valid pixel range each iteration:

```python
import torch
import torch.nn as nn


def run_attack(
    model: nn.Module,
    images: torch.Tensor,    # (N, C, H, W), values in [0, 1]
    labels: torch.Tensor,    # (N,)
    eps: float,              # L_inf budget: ||x_adv - x||_inf <= eps
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    model.eval()

    steps = 10               # T iterations of iterative sign-ascent
    alpha = eps / steps      # alpha = eps / T: aligned sign-steps can fill the budget
    decay = 1.0              # mu = 1.0: undiscounted accumulation of the consensus direction

    x = images.detach().to(device)
    labels = labels.detach().to(device)
    adv = x.clone().detach()
    momentum = torch.zeros_like(x)          # g_0 = 0
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(steps):
        adv.requires_grad = True
        outputs = model(adv)
        cost = loss_fn(outputs, labels)                    # J(x*_t, y); maximize it
        grad = torch.autograd.grad(
            cost, adv, retain_graph=False, create_graph=False
        )[0]                                               # current input gradient

        # normalize this iteration's gradient so every step votes equally in the average.
        # mean-abs is proportional to the per-sample L1 norm used in the formula; the
        # fixed factor cancels from the accumulated direction for fixed-size images.
        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        grad = grad + momentum * decay                    # g_{t+1} = normalized grad + mu*g_t
        momentum = grad

        # L_inf-optimal step along the accumulated direction, then dual clamping.
        adv = adv.detach() + alpha * grad.sign()
        delta = torch.clamp(adv - x, min=-eps, max=eps)
        adv = torch.clamp(x + delta, min=0, max=1).detach()

    return adv
```
