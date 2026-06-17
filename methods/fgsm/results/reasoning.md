Let me start from what actually hurts. I have a trained classifier, and I know — this is established, not something I'm about to discover — that there exist inputs a hair away from correctly classified images that the model gets confidently wrong. Szegedy and his collaborators showed it: take a correctly classified `x`, and there's an `r`, small enough to be invisible, with `x + r` misclassified. The unsettling part isn't that one model has blind spots; it's that the *same* perturbed image fools a different architecture trained on a different slice of the data, and the different models tend to agree on the wrong label. And it isn't fragile noise — random distortion of the same per-pixel size mostly leaves the predictions intact, while these crafted ones essentially never survive. So whatever is going on is structural and shared across models, not memorized junk.

How do I make one of these today? The only recipe I have is Szegedy's: to drive `x` to a chosen label `l`, minimize `||r||_2` subject to `f(x + r) = l` and `x + r ∈ [0,1]^m`, relaxed into `minimize c·||r||_2 + loss_f(x + r, l)` over the box, solved with box-constrained L-BFGS and a line search over the penalty weight `c` until the label actually flips. It works, it finds tiny transferable perturbations — it's the existence proof for the whole phenomenon. But look at the cost: an iterative constrained optimizer, *with a line search over `c`*, run per single image. If I want to study this across many models, or — the thing I really want — fold adversarial examples into the training loop so the model learns to resist them, I'd have to solve one of these optimizations for every example in every minibatch. Dead on arrival. I need something I can compute in about the cost of one backward pass. Wall, immediately: the generation method is the bottleneck.

So let me back up and ask the prior question, because I suspect the explanation and the cheap method are the same insight. *Why* do these examples exist, and why do they transfer? The going story is that deep nets are wildly nonlinear, over-parameterized, under-regularized — that they carve the input space into bizarre discontinuous pockets, and adversarial examples are needles hidden in those pockets. But that story actively fights the evidence. If each model's blind spots were idiosyncratic high-curvature pockets, why would a needle found for one model land in a needle for a *different* model with different depth and weights, trained on disjoint data, and why would they agree on the wrong class? Independent quirks don't line up like that. And the clincher: a plain linear softmax classifier on raw pixels has the *same* vulnerability. There's no extreme nonlinearity there to blame. The nonlinearity story is not just unproven, it's pointed the wrong way.

Let me take the opposite hypothesis seriously and just work it out for the simplest possible model — a linear one — and see if adversarial examples fall out of *linearity* rather than nonlinearity. A linear unit computes `w^T x`. Feed it `x̃ = x + η`:

  `w^T x̃ = w^T x + w^T η`.

The perturbation moves the activation by exactly `w^T η`. Now, what's the most damage I can do to that activation under the constraint that no single pixel moves by more than the precision I care about — `||η||_∞ ≤ ε`? I want to maximize `w^T η = Σ_i w_i η_i` with each `|η_i| ≤ ε`. The terms decouple: each `w_i η_i` is maximized independently by pushing `η_i` to the corner of its allowed interval *in the direction of `w_i`* — set `η_i = ε` if `w_i > 0`, `η_i = −ε` if `w_i < 0`. That's `η = ε·sign(w)`, and it gives

  `w^T η = ε Σ_i |w_i| = ε ||w||_1`.

Stare at the scaling. Say the weight vector has `n` components with average magnitude `m`. Then `w^T η = ε·m·n`. The size of my perturbation, `||η||_∞ = ε`, *does not grow with `n`*. But the change I produce at the output grows *linearly* in `n`. So in high dimension I can make a huge number of changes, each individually below the precision threshold — each one invisible, each one "shouldn't matter" — and they add up coherently into one large swing at the output, because I aligned every single one with its weight. It's a kind of accidental steganography: a linear readout is forced to attend to the one direction that aligns with its weights, and I've hidden a large signal in that direction while every pixel looks untouched. That's it. Adversarial examples don't need curvature; they need *dimension* and *linearity*. A simple linear model with high-dimensional input has them necessarily. This already explains why shallow softmax is vulnerable, with no appeal to anything deep.

Now the bridge to real networks. They're not linear — but the modern ones are *built to be nearly linear* because that's what makes them trainable. Rectified linear units are piecewise linear. Maxout units are piecewise linear. LSTMs gate things to keep gradients flowing through near-linear paths. Sigmoid networks are deliberately kept in the non-saturating, locally-linear regime so they optimize. So my working hypothesis is: these models are *too linear to resist a linear perturbation*. If that's right, then the same closed-form perturbation that's optimal for a linear model should also damage the nonlinear ones, because near the input they behave linearly.

So I want to do for the *cost* `J(θ, x, y)` what I just did for a single linear activation. The thing I'm attacking isn't one unit's pre-activation; it's the scalar training loss, and I want to *raise* it (raise the loss = push toward misclassification). I don't have a global linear form for `J`, but I have the next best thing: its first-order behaviour around the current input. Taylor-expand the loss in the input:

  `J(θ, x + η, y) ≈ J(θ, x, y) + η^T ∇_x J(θ, x, y)`.

To make this concrete I need the increase, the part I can control, which is `η^T ∇_x J(θ, x, y)`. Write `g = ∇_x J(θ, x, y)` — one gradient of the loss with respect to the *input*, which backprop hands me in a single backward pass, the same machinery I already use to get gradients with respect to weights, just stopped one step earlier at the input. Now I maximize the linearized increase under the same per-pixel budget:

  `max_{||η||_∞ ≤ ε} η^T g`.

This is exactly the linear-model problem again, with `g` playing the role `w` played. I can solve it the same way, coordinate by coordinate, but let me also see it as the clean general statement, because it tells me the choice of norm is doing real work. By Hölder's inequality with the conjugate exponents `∞` and `1`,

  `η^T g ≤ ||η||_∞ ||g||_1 ≤ ε ||g||_1`,

and equality in Hölder for this pair holds when every nonzero-gradient coordinate has the same sign as `g_i` and is pushed to the boundary `|η_i| = ε`; zero-gradient coordinates do not matter, so I choose `0` there. So a canonical maximizer is forced:

  `η = ε·sign(∇_x J(θ, x, y))`,   achieving   `η^T g = ε ||g||_1`.

There it is, in closed form, no inner loop. The geometry is worth saying out loud so I don't reach for the wrong thing: the feasible set `{η : ||η||_∞ ≤ ε}` is an axis-aligned cube, the objective is *linear*, so the maximum sits at a *corner* of the cube — and a corner is exactly a vector all of whose coordinates are `±ε`, i.e. `ε·sign(g)`. This is why it's the *sign* and not the gradient itself. If my budget had been `ℓ_2` instead of `ℓ_∞`, the feasible set would be a ball, the maximizer would be `ε·g/||g||_2`, and large-magnitude coordinates would soak up most of the budget. But under `ℓ_∞` every coordinate has its own independent allowance `ε`, and the right move is to spend each one fully in its gradient's direction — magnitude is irrelevant, only the *sign* of each partial derivative matters. The fast gradient sign method: one forward pass, one backward pass, an elementwise sign, scale by `ε`, done.

And why is `ℓ_∞` the right budget in the first place, not just the convenient one? Because of feature precision. An 8-bit image throws away everything below `1/255` of its dynamic range; a change to one pixel smaller than the sensor can even represent is, by construction, semantically meaningless. The honest notion of "imperceptible" is therefore *per-coordinate* — no single feature moves by more than the precision floor — which is a max-norm constraint, not an aggregate `ℓ_2` one. The norm isn't a mathematical convenience I picked to get a clean answer; it's the formalization of "below the precision of the features," and it just happens to also give the cleanest answer.

One worry I should check before trusting this on a real network: it's only a *first-order* method. I linearized `J`, and `J` is not linear in `x` for a deep net, so `η = ε·sign(g)` maximizes the *approximation*, not the true loss inside the box. Could it just fail? Two things reassure me. First, in the one case where the model genuinely *is* linear in `x` — logistic regression — there's no approximation at all and this is the exact worst case in the box, so I can dissect it cleanly (I'll do that next). Second, my whole hypothesis was that the nonlinear models are *too linear to resist*, so the first-order step *should* transfer; if it reliably fools them in practice, that success is itself evidence for the linearity explanation. The method and the hypothesis test each other. So I expect it to work precisely *because* it's a linear attack, and I'd want to confirm that across maxout/conv/softmax models — but the logic is self-consistent: if linearity is the cause, a linear attack is the cure-finder.

Let me actually do the logistic-regression case, because it's where the sign step is exact and it connects the attack to something I already know: weight decay. Train a single unit to predict `y ∈ {−1, 1}` with `P(y=1) = σ(w^T x + b)`. The per-example loss in the usual `±1` convention is the softplus `ζ(−y(w^T x + b))` with `ζ(z) = log(1 + e^z)`, so training descends `E_{x,y}[ζ(−y(w^T x + b))]`. Now I want the worst-case adversarial perturbation of `x` and what training on it looks like. The cost as a function of `x` is `ζ(−y(w^T x + b))`; its gradient in `x` is `ζ'(·)·(−y)·w`, and since `ζ' = σ > 0`, the sign of the input-gradient is

  `sign(∇_x J) = sign(−y w) = −y·sign(w)`.

The fast sign perturbation is therefore `η = ε·sign(∇_x J) = −y·ε·sign(w)`. Substitute `x̃ = x + η` into the margin:

  `y(w^T x̃ + b) = y(w^T x + b) + y·w^T η = y(w^T x + b) − ε·w^T sign(w)`.

Use `w^T sign(w) = Σ_i w_i sign(w_i) = Σ_i |w_i| = ||w||_1`:

  `y(w^T x̃ + b) = y(w^T x + b) − ε ||w||_1`.

The adversary subtracts `ε ||w||_1` from the signed margin before the softplus sees it, so the literal `±1` worst-case loss is `E_{x,y}[ζ(ε ||w||_1 − y(w^T x + b))]`. In the compact binary-margin notation used for the analytic comparison to weight decay, I write the adversarial logistic objective as

  `E_{x,y}[ ζ( y(ε ||w||_1 − w^T x − b) ) ]`.

The `ε ||w||_1` term looks like `ℓ_1` regularization — and now I can see the precise difference, which I'd never have seen by just writing down "add `λ||w||_1`." Here the `||w||_1` penalty is *subtracted off the activation, inside `ζ`*, not added on top of the loss. That changes its behaviour: once the model is confident enough that `ζ` saturates (the example is well inside the right side with margin), `ζ'` collapses and the penalty effectively *deactivates* — a comfortably-classified example with good margin gets no extra pressure, because no per-input perturbation of size `ε` could actually flip it. Ordinary `ℓ_1` weight decay can't do that; it's added to the cost and keeps pulling regardless of margin, so it's strictly more pessimistic — it assumes the adversary can always do the worst-case damage even when the margin makes that impossible. Concretely that means weight decay over-estimates the achievable damage and has to use a coefficient *smaller* than the precision `ε` to avoid wrecking training, and the gap only widens for multiclass softmax (no single `η` can align with every class's weight vector at once) and for deep nets. So adversarial training with the sign method is a *less* pessimistic, margin-aware relative of `ℓ_1` decay — and it's cheap, which is the whole point.

That cheapness is what unlocks the second half. Szegedy showed mixing adversarial examples into training regularizes, but with L-BFGS in the inner loop it couldn't be run at scale. With the sign method I can regenerate fresh adversarial examples *every minibatch, against the current weights*, for the price of one extra backward pass. So train on a blend of clean and adversarial loss:

  `J̃(θ, x, y) = α J(θ, x, y) + (1 − α) J(θ, x + ε·sign(∇_x J(θ, x, y)), y)`,

with `α = 0.5` as the obvious symmetric starting point. Because the perturbation is recomputed from the current `θ` each step, the supply of adversarial examples chases the model as it moves — it's adversarial example generation and training fused into one loop. The reading of this is clean: I'm minimizing the worst-case loss over the `ε` max-norm box around each input, i.e. an upper bound on the expected loss under any input noise bounded by `ε`. And it's why *random* noise training is so much weaker — a zero-mean perturbation has expected inner product zero with the gradient direction, so on average it doesn't raise the loss at all; the sign step deliberately picks the one corner of the box that does maximal first-order damage. Adversarial training is hard-example mining among the noisy inputs, keeping only the ones that actually resist.

One subtlety in that objective I should flag rather than gloss. The derivative of `sign(·)` is zero almost everywhere and undefined at zero, so when I backprop the outer training gradient through `J̃`, I can't differentiate *through* the perturbation `ε·sign(∇_x J)` — its dependence on `θ` carries no usable gradient. In practice the perturbation is treated as a constant when taking the outer step: compute `η` from the current params, then take a gradient step on `J(θ, x + η, y)` holding `η` fixed. The model can't anticipate, through the sign, how the attacker will react to a weight change — but it still gets pushed to be correct on the current worst-case point each step, which is enough to regularize.

Let me also pin down *why these examples generalize*, because the linear view explains it and it sharpens what the attack is doing. Under linearity, an adversarial example is not a needle in a fine pocket — it's any `η` whose inner product with `g` is positive and whose `ε` is large enough. That's a broad half-space-like region, not a precise location. So perturbing along `sign(g)` lands in a *contiguous* region of misclassification: sweep `ε` and the logits move piecewise-linearly, the wrong prediction stays stable across a wide band of `ε`. Because it's a fat subspace and not a pocket, an example built for one model has a high prior probability of also lying in the misclassification region of another model — and since different models trained on the same task learn approximately the same linear reference weights (that's just ML generalizing), their gradient directions align, so the examples transfer *and* the models agree on the wrong class. The cheap method isn't just convenient; its very form — move along the sign of the gradient — is what makes the examples broad and transferable.

So let me write the attack as I'd actually ship it, filling the one empty slot in the white-box harness: take the clean batch, get one input-gradient of the cross-entropy loss, step every pixel by `ε` in the sign of its gradient, and clamp the candidate image to the valid pixel range. The cross-entropy is the standard classification loss `J`; raising it is pushing the logits away from the true label.

```python
import torch
import torch.nn as nn


def run_attack(
    model: nn.Module,
    images: torch.Tensor,   # (N, C, H, W), values in [0, 1]
    labels: torch.Tensor,   # (N,)
    eps: float,             # per-pixel L-inf budget
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    _ = n_classes
    model.eval()
    x = images.detach().clone().to(device).requires_grad_(True)
    y = labels.detach().clone().to(device)
    logits = model(x)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, y)                    # J(theta, x, y); we want to RAISE it
    grad = torch.autograd.grad(
        loss, x, retain_graph=False, create_graph=False
    )[0]                                         # g = d J / d x, one backward pass

    with torch.no_grad():
        # eta = eps * sign(g): the L-inf-ball maximizer of the linearized loss increase
        x_adv = x + eps * grad.sign()
        x_adv = torch.clamp(x_adv, min=0, max=1)

    return x_adv.detach()
```

The causal chain, start to finish. I was stuck because the only known way to make an adversarial example ran an iterative constrained optimizer per image — far too slow to study at scale or to train against. Rather than treat the phenomenon as exotic nonlinearity, I tested the opposite hypothesis on the simplest model: a linear unit's activation moves by `w^T η`, and under a per-pixel budget `||η||_∞ ≤ ε` the damage is maximized at `η = ε·sign(w)`, growing like `ε·m·n` — linearly in dimension while the perturbation stays invisibly small. So linearity plus high dimension is sufficient to create adversarial examples, which already explains why even shallow linear classifiers are vulnerable. Modern nets are deliberately near-linear to be trainable, so I lifted the same move to the loss: linearize `J` around `x`, maximize the first-order increase `η^T g` over the `ℓ_∞` ball, and Hölder pins the maximizer to a cube corner, `η = ε·sign(∇_x J)` — the fast gradient sign method, one backward pass. The `ℓ_∞` norm is the right one because feature precision is per-coordinate. In the logistic-regression case the step is exact and reveals adversarial training as a margin-aware, less-pessimistic cousin of `ℓ_1` weight decay. The method being cheap is what makes adversarial training practical — regenerate the worst-case point each minibatch and minimize the loss there, treating the non-differentiable sign perturbation as fixed during the outer step. And because the attack moves along the gradient sign, the misclassification region is a broad subspace, so the examples transfer across models that learned the same reference weights, which is exactly the cross-model, cross-dataset generalization I started out unable to explain.
