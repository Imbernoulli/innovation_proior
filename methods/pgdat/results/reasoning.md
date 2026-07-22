Let me start from what actually hurts. I have a convolutional net that hits ninety-nine-plus percent on MNIST, and yet if I nudge every pixel by an amount smaller than the sensor would ever record, it confidently calls a seven a three. This isn't a fluke input I had to hunt for — there's one of these next to essentially every image, and the same nudged image fools other architectures too. So the model got the right answers on the test set without learning anything I'd call the concept. If I'm going to put this thing near a steering wheel or a face lock, "high test accuracy" is just not the property I need.

What property do I need? I keep reading defenses — distill the network, squeeze the input bit-depth, bolt a detector on the front — and every one of them has the same shape of flaw: it's tuned against one attack, and I have no way to say anything about attacks I haven't tried. I can't certify "nothing within this allowed perturbation set can flip the label." That's the actual gap. Not "build a defense" — defenses are everywhere — but "state a goal precise enough that hitting it *means* something." So before I touch an algorithm I want to write down the thing I wish were true.

Standard training minimizes the expected loss on clean data, `min_θ E_{(x,y)}[L(θ,x,y)]`. The trouble is that the expectation is over clean `x`, and the adversary gets to move `x` afterward. So let me bake the adversary into the objective: for each example, before I pay the loss, let an adversary pick the *worst* perturbation `δ` inside the allowed set `S`. Write `S` as the `ℓ_∞` ball of radius `ε` — that's the threat model everyone agrees on, "no pixel moves by more than `ε`," and it matches the linear story that a max-norm-bounded shift is what fools these high-dimensional linear-ish models. Then the quantity I actually care about is

    ρ(θ) = E_{(x,y)~D}[ max_{δ∈S} L(θ, x+δ, y) ],     S = { δ : ||δ||_∞ ≤ ε },

and I want `min_θ ρ(θ)`. Stare at this for a second, because it's doing more than it looks. If I ever drive `ρ(θ)` small, that means the loss is small for *every* `δ` in `S`, for the typical example — there is no admissible perturbation left that produces high loss, so by construction there's no adversarial example within the budget. That's the guarantee I couldn't get from the bolt-on defenses. It isn't "robust to FGSM"; it's "robust to the whole class `S`," and the value of `ρ` is itself a number that measures how robust.

And the same expression unifies the two camps that had been talking past each other. The inner `max_{δ∈S} L(θ,x+δ,y)` — that *is* the attacker's job, find the perturbation that maximizes loss. The outer `min_θ` — that *is* the defender's job. Attack and defense aren't separate research programs; they're the inner and outer halves of one saddle-point problem. FGSM, the one-step sign attack, is just one cheap stab at the inner max. FGSM adversarial training, where you mix in `J(θ, x + ε·sign(∇_x J))`, is just an outer step taken on top of that cheap inner stab. Everything people had been doing piecemeal is a special case of solving this one min-max. That's the reframing; now I have to actually solve it.

The reason nobody had committed to this is that it looks hopeless on two counts. The outer problem is the usual non-convex deep-net optimization — fine, SGD handles that empirically. But the inner problem is a *maximization* of a wildly non-concave function over the ball, and the prior min-max papers (Huang, Shaham) looked at exactly this and concluded the inner max is too hard to solve, so they fell back to one-step linearization and only ever evaluated against FGSM. So my first real question isn't "what's the algorithm," it's "is the inner maximization actually intractable, or did people just assume it was?"

Let me think about how I'd even attack the inner max. I have an `ℓ_∞`-constrained maximization of `L` over `δ`. The natural tool for large-scale constrained optimization is projected gradient ascent: take a gradient step on `δ`, then project back into the constraint set. Step direction: I want to increase `L(θ, x+δ, y)` as fast as possible for a bounded step. If I bound the step in `ℓ_∞`, the steepest-ascent direction is the solution of `max_{||v||_∞ ≤ α} (∇_δ L)ᵀ v`. That's a linear objective over a box, so the maximizer pins each coordinate to its corner: `v = α·sign(∇_δ L)`. So the step is `δ ← δ + α·sign(∇_δ L)`, which is exactly FGSM's move but taken *repeatedly* and with a small step `α` instead of one jump of size `ε`. That's the whole point — FGSM is the one-step `α=ε` version; nothing forces me to take a single step. The sign also has the nice side effect that the step size `α` is in pixel units regardless of the gradient's magnitude, so I don't have to retune `α` as the gradient scale drifts; that's why I take the sign rather than the raw gradient here.

Now the projection. After a step, `δ` might leave the `ℓ_∞` ball, and `x+δ` might leave the valid pixel range `[0,1]`. Project back: clip `δ` coordinatewise to `[-ε, ε]` (that's the Euclidean projection onto an `ℓ_∞` ball — each coordinate independently), then clip `x+δ` into `[0,1]`. Concretely, in terms of the perturbed image `x_t = x + δ_t`:

    x_{t+1} = clip_{[0,1]}( clip_{[x-ε, x+ε]}( x_t + α·sign(∇_x L(θ, x_t, y)) ) ).

That's projected gradient descent on the negative loss — PGD as the inner adversary.

Where do I start `δ`? FGSM starts at `δ=0`, right at the clean point. But right at `x` the loss is close to linear in the ball — that's Goodfellow's own explanation for why one step works at all — so starting there and stepping biases me toward the linear regime, the very regime that one-step methods already exploit and that doesn't represent the whole ball. Let me instead start from a *random* point inside the ball, `δ_0 ~ Uniform(-ε, ε)` per coordinate (then clip into `[0,1]`). Random start does two things: it breaks me out of the linear neighborhood of `x`, and across restarts it samples different basins of the inner landscape.

Which raises the worry that killed this for everyone else: PGD is climbing a non-concave surface, so it only finds *local* maxima, and if those local maxima are all over the place — some easy hills, some enormous spikes I'll never land on — then training against "whatever PGD happened to find" is training against a moving, unrepresentative target. I can't just assert the inner problem is fine. Let me actually probe the landscape. Pick many random starting points in `x+S`, run PGD from each, and watch.

From a random start, the adversarial loss climbs in a fairly consistent way and then *plateaus* quickly. It doesn't wander; a handful of steps and it's essentially done. If I run a huge number of random restarts, say on the order of `10^5`, and collect the *final* loss value from each, those values form a tight, well-concentrated distribution. No fat tail of monstrous outliers. The local maxima PGD reaches are *different points* — when I measure pairwise `ℓ_2` distances between the maximizers they're as far apart as random points in the ball, and the angles between them sit near ninety degrees, so these really are distinct basins, not the same hilltop re-found — and yet they all reach *about the same loss*. (Even along the straight segment between two of these maxima the loss stays high, dipping only by a constant factor in the middle; it's not that there's one true peak and everything else is a foothill.) That's the folklore picture from training itself — many distinct local minima of comparable value — showing up here in the *input* landscape of the inner max.

So the thing I assumed was a wall — "PGD only finds local maxima" — turns out not to bite, because the local maxima are interchangeable in the only way that matters: their loss. If every first-order trajectory ends at roughly the same height, then "what PGD finds" is a stable, representative estimate of the inner max, not a lottery. And I notice something the linear/subspace view of these perturbations would not have predicted: some of the perturbations PGD lands on have *negative* inner product with the gradient at `x`, and the correlation with the clean gradient direction decays as the perturbation grows. So the loss surface in the ball is genuinely not the simple linear ramp the one-step picture assumes — which is exactly why one step underperforms, and exactly why running PGD to its plateau matters.

If the loss values found by *first-order* trajectories concentrate like this, then any adversary that only uses gradients of the loss with respect to the input is going to be pulled into that same concentrated band. PGD, run from random starts to its plateau, is then not just *an* attack; it's a stand-in for the strongest thing a first-order adversary can do. I'll call it the universal first-order adversary, in the sense that beating PGD should mean beating the whole class of gradient-based attacks. I can't rule out some isolated maximum with much higher loss — the surface is non-concave — but `10^5` restarts didn't find one, which says such a point, if it exists, is hard to *find* with first-order methods. And the only attacks that scale on deep nets *are* first-order; second-order is out for the same memory reasons that make us train with SGD in the first place. So "robust to first-order attacks" is, for current practice, the tractable target. That's the conjecture I'm willing to build the defense on: train so the inner PGD can't find a high-loss point, and robustness to the broad class of gradient-based attacks comes along.

So the inner problem is solved well enough by random-start PGD. Back to the outer problem: I have `ρ(θ) = E[ φ(θ) ]` where `φ(θ) = max_{δ∈S} L(θ, x+δ, y)`, and I want to descend it with SGD. To take an SGD step I need `∇_θ φ(θ)`. But `φ` is a *max* over `δ` — the maximizer `δ*` itself depends on `θ`. Naively I'd have to differentiate through the entire inner optimization: how `δ*` moves as `θ` moves. That sounds like it needs the implicit function theorem and second derivatives, exactly the kind of thing I want to avoid.

Let me push on that and see if it really does. Suppose I just *ignore* the dependence of `δ*` on `θ` — freeze the perturbation at the value PGD found, build the adversarial image `x + δ*`, and backprop the ordinary loss `L(θ, x+δ*, y)` through `θ` as if `x+δ*` were a fixed input. Is that gradient legitimate, or am I fooling myself by dropping the `dδ*/dθ` term?

This is exactly Danskin's theorem, and it's worth doing carefully because the whole method's validity rests on it. Take `g(θ, δ) = L(θ, x+δ, y)` and `φ(θ) = max_{δ∈S} g(θ, δ)`, with `S` compact and `g(·,δ)` differentiable in `θ` with `∇_θ g` continuous. Let `δ*(θ)` be the set of inner maximizers. Danskin says `φ` is locally Lipschitz and directionally differentiable, and its directional derivative in a direction `h` is

    φ'(θ, h) = sup_{δ ∈ δ*(θ)} hᵀ ∇_θ g(θ, δ).

The intuition is that gradients are local objects: near `θ`, the function `φ` agrees with `g(·, δ*)` for the maximizing `δ*`, so to first order their slopes coincide — the way `δ*` shifts contributes nothing to the *value* of the max at the optimum, by the envelope argument, because at the maximum the inner gradient with respect to `δ` is zero (or pinned by the constraint), so moving `δ*` is a zeroth-order effect on `φ`. When the maximizer is unique, `δ*(θ) = {δ*}`, the sup collapses and `φ` is plainly differentiable with `∇φ(θ) = ∇_θ g(θ, δ*)`. So the `dδ*/dθ` term I was afraid of genuinely drops out. I do *not* need to differentiate through the inner optimization.

Let me confirm the sign, because this is exactly where it is easy to fool myself. Take `δ̄` a maximizer of the inner problem and write `h = ∇_θ L(θ, x+δ̄, y)`. Plug the *positive* direction `h` into Danskin:

    φ'(θ, h) = sup_{δ ∈ δ*(θ)} hᵀ ∇_θ L(θ, x+δ, y) ≥ hᵀ ∇_θ L(θ, x+δ̄, y) = hᵀh = ||∇_θ L(θ, x+δ̄, y)||² ≥ 0,

where the inequality is just that the sup over the maximizer set is at least the value at the particular maximizer `δ̄` (which lies in that set). If `h` is nonzero, this says the adversarial loss increases along `+h`; it is an ascent-direction check, not the descent step yet. The descent statement is clean in the generic case I actually use for backprop: when the active maximizer is unique, Danskin collapses to `∇φ(θ) = h`, so

    φ'(θ, -h) = -||h||² < 0.

That is the sign I need. Freezing the adversarial point and backpropagating the ordinary loss gives the outer gradient, and the optimizer's update in the `-h` direction descends the saddle-point objective. Replacing each clean input by its adversarial version and training the network the ordinary way *is* SGD on the adversarial objective whenever the found adversarial point is the active maximizer.

Two honest gaps before I trust this on a real net. First, my loss isn't continuously differentiable — ReLU and max-pooling have kinks, so the smoothness hypothesis of Danskin fails on a set of points. But that set has measure zero; in practice I never sit exactly on a kink, so I'll treat the conclusion as holding almost everywhere. Second — the real one — PGD does *not* return a certified global inner maximizer; it returns an approximate, local one. Danskin as stated wants the true max. The repair is local: restrict attention to a neighborhood `S' ⊆ S` in which the local maximum PGD found is the active/global maximum, apply Danskin on `S'`, and conclude that the gradient at that point is a descent direction for the adversarial loss restricted to `S'`. In words: if PGD found a genuinely high-loss adversarial example, then stepping `θ` to reduce the loss there makes progress against that local adversary. It isn't a certificate for the exact global object, but the landscape study tells me PGD's point is representative of what first-order attacks can actually find.

Now I have a complete algorithm, and I want to nail down the few remaining knobs by reasoning about what each is for, not by guessing.

The inner step size `α` versus the budget `ε`. FGSM jumps once by `ε`. I'm taking `k` steps; if each step were `ε` I'd leave the ball immediately and just bounce around its surface. I want several smaller steps that *traverse* the ball and let the projection keep me legal, so `α < ε` with enough steps `k` that `k·α` comfortably exceeds `ε` — that way PGD can reach the boundary from any interior start and still have room to move along it. On MNIST that's `ε = 0.3`, `α = 0.01`, `k = 40`; on CIFAR-10, in `[0,255]` pixel units, that's `ε = 8`, `α = 2`, `k = 7`, equivalently `ε = 8/255` and `α = 2/255` after scaling pixels to `[0,1]`. The common rule is the important part: small sign steps, projection after each step, and total possible travel beyond the radius so the boundary is reachable from any random start.

Why bother with the random start instead of the simpler `δ=0`? Beyond escaping the linear neighborhood of `x`, there's a failure mode I want to avoid: if I always start at the same deterministic point, the adversary produces a narrow, predictable family of perturbed images, and the network can overfit to exactly those — the label-leaking pathology seen with one-step training, where the model scores beautifully on its own adversary's examples and learns nothing transferable. Random starts make the inner adversary sample the ball, so the outer training never gets a fixed target to memorize. And there's a convenient consequence: because I train for many epochs and draw a *fresh* random start every time I revisit an example, I'm effectively doing random restarts across epochs already — so I don't need to pay for multiple PGD restarts per batch during training; one start per example per epoch is enough.

Why multi-step at all, when FGSM training is so much cheaper? Because the inner max is the part the prior min-max work got wrong. FGSM linearizes the loss in the ball; the landscape study showed the loss is *not* linear out to the `ε` of interest — the perturbations PGD finds even point partly against the clean gradient. Train against the linearized adversary and a non-linear adversary walks right past your defense; that's precisely why FGSM-trained models fall to iterative attacks. The saddle-point objective tells me to train against the *strongest* feasible inner adversary, and PGD-to-plateau is the strongest first-order one I have. Spending the inner steps is the price of a defense that generalizes across attacks instead of one that overfits to its own weak adversary.

One more thing the formulation forces me to take seriously: capacity. The objective separates a clean decision boundary from a robust one, and a robust boundary must separate not points but the `ℓ_∞` balls *around* the points — a strictly harder, more contorted boundary. So I shouldn't expect a network sized for clean accuracy to be able to represent the robust classifier at all. I'd expect that if I take a too-small net and train it against PGD, it can't fit the adversarial examples and collapses to something trivial like a constant prediction, and that as I grow capacity the achievable value of the saddle point drops. So the method is really "strongest adversary *and* enough capacity"; either alone isn't enough.

Let me write the per-step procedure exactly as it'll run. Put the model in eval mode while I craft the attack so that the batch-norm statistics and any dropout don't shift under me as I perturb the input — I want the gradient of the deployed function, not of a randomized one. Initialize the adversarial batch at a uniform-random point in the `ℓ_∞` ball, clipped to valid pixels. Then `k` times: turn on input gradients, forward through the model, compute cross-entropy against the *true* labels (maximizing the true-label loss is what drives a misclassification — the untargeted attack), take the gradient with respect to the input, step along its sign by `α`, project the perturbation back to `[-ε, ε]`, and clamp the image to `[0,1]`. Detach between steps so the graph doesn't accumulate across the `k` iterations — each step only needs the gradient at the current point. After the inner loop, switch to train mode, forward the adversarial batch once more, compute cross-entropy, and take a single SGD step. In the active-maximizer regime from the Danskin argument, that step descends the local adversarial objective.

```python
import torch
import torch.nn.functional as F


class AdversarialTrainer:
    """Solve  min_theta  E[ max_{||delta||_inf <= eps} L(theta, x+delta, y) ].
    Inner max: random-start L_inf PGD (the strongest first-order adversary).
    Outer min: one ordinary SGD step on the loss at the adversarial point
    (valid by Danskin when the found point is the active inner maximizer)."""

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps                    # L_inf budget defining the ball S
        self.alpha = alpha                # inner step size, alpha < eps
        self.attack_steps = attack_steps  # k PGD steps; k*alpha > eps to reach the boundary
        self.num_classes = num_classes

    def train_step(self, images, labels, optimizer):
        # Eval mode while crafting the attack, then ordinary training on x_adv.
        self.model.eval()
        adv_images = images.clone().detach()

        # Same random-start shape as torchattacks.PGD.forward.
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(
            -self.eps, self.eps
        )
        adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()

        for _ in range(self.attack_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(adv_images)
            # Untargeted attack: maximize true-label cross-entropy.
            cost = F.cross_entropy(outputs, labels)
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()

        self.model.train()
        outputs = self.model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {'loss': loss.item()}
```
