# PGD Adversarial Training

## Problem

A classifier trained by empirical risk minimization can be near-perfect on clean inputs yet
confidently misclassify inputs perturbed by an imperceptible amount (`||δ||_∞ ≤ ε`). Bolt-on
defenses are each tuned to one attack and certify nothing about a *class* of perturbations. We want
a single objective whose smallness guarantees robustness to all admissible perturbations, plus a
training procedure that drives it small on deep networks.

## Key idea

Cast adversarial robustness as a **robust-optimization saddle point**. For an `ℓ_∞` threat model
`S = {δ : ||δ||_∞ ≤ ε}`:

    min_θ  ρ(θ),     ρ(θ) = E_{(x,y)~D}[ max_{δ∈S} L(θ, x+δ, y) ].

The **inner maximization** is the attack (find the worst perturbation); the **outer minimization**
is the defense. A small `ρ(θ)` means the loss is small for *every* `δ ∈ S`, so no adversarial
example exists within the budget. Two facts make it solvable in practice:

1. **The inner max is tractable with first-order methods.** Random-start projected gradient descent
   (PGD) on the input, run to its plateau, lands at well-concentrated loss values across many random
   restarts and across distinct basins. PGD therefore serves as a practical "universal first-order
   adversary": robustness to PGD is the yardstick for robustness to gradient-based attacks broadly.
2. **The outer gradient is the gradient at the active inner maximizer (Danskin).** No differentiation
   through the inner optimization is needed in the active-maximizer case — freeze the PGD-found `δ*`
   and take an ordinary SGD step on `L(θ, x+δ*, y)`.

Robust classification also requires **larger model capacity** than clean classification, because the
robust decision boundary (separating `ℓ_∞` balls, not points) is more contorted.

## Inner adversary: `ℓ_∞` PGD

Start at a random point in the ball, then iterate `k` projected sign-gradient ascent steps of size
`α`:

    x_0   = clip_{[0,1]}( x + Uniform(-ε, ε) )
    x_{t+1} = clip_{[0,1]}( clip_{[x-ε, x+ε]}( x_t + α·sign(∇_x L(θ, x_t, y)) ) ).

The step direction `sign(∇_x L)` is the `ℓ_∞` steepest-ascent direction (`argmax_{||v||_∞≤α} (∇L)ᵀv`)
and keeps the step in pixel units; the two clips are the projection onto the `ℓ_∞` ball and onto the
valid pixel range. FGSM is the one-step special case (`k=1`, `α=ε`, no random start). Use `α < ε`
with `k·α > ε` so PGD can reach and move along the boundary from any start.

## Why the outer step is valid (Danskin's theorem)

Let `g(θ,δ) = L(θ, x+δ, y)` and `φ(θ) = max_{δ∈S} g(θ,δ)` with `S` compact, `g(·,δ)` differentiable
and `∇_θ g` continuous. With maximizer set `δ*(θ)`,

    φ'(θ, h) = sup_{δ∈δ*(θ)} hᵀ ∇_θ g(θ, δ);   if δ*(θ)={δ*}:  ∇φ(θ) = ∇_θ g(θ, δ*).

Choosing `h = ∇_θ L(θ, x+δ̄, y)` at a maximizer `δ̄` gives
`φ'(θ,h) ≥ hᵀh = ||∇_θ L(θ, x+δ̄, y)||² ≥ 0`, which identifies `+h` as an ascent direction. In the
unique/active-maximizer case, Danskin collapses to `∇φ(θ)=h`, so
`φ'(θ,-h) = -||h||² < 0`; this is the descent-direction inequality used by the outer SGD step.
In the unique/active-maximizer case, training the network normally on adversarial images is the SGD
step for the corresponding adversarial objective. ReLU/max-pool kinks are treated as measure-zero,
and PGD's local maximizer is handled by applying the theorem on a subregion `S'` where that point is
the active maximum.

## Algorithm (per training step)

1. `model.eval()`; init `x_adv` at a uniform-random point in the `ℓ_∞` ball, clip to `[0,1]`.
2. Repeat `k` times: forward, compute CE against true labels, take input-gradient, step
   `x_adv += α·sign(grad)`, project `δ` to `[-ε,ε]`, clamp `x_adv` to `[0,1]`.
3. `model.train()`; forward `x_adv`, compute CE, take one SGD step.

Concrete settings: MNIST `ε=0.3, α=0.01, k=40`; CIFAR-10 `ε=8/255, α=2/255, k=7`, with a wide
residual network. Because training runs many epochs with a fresh random start each visit, no
per-batch restarts are needed.

## Code

```python
import torch
import torch.nn.functional as F


class AdversarialTrainer:
    """PGD adversarial training: solve min_theta E[ max_{||delta||_inf <= eps} L ]."""

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps                    # L_inf perturbation budget
        self.alpha = alpha                # PGD inner step size (alpha < eps)
        self.attack_steps = attack_steps  # number of PGD steps k
        self.num_classes = num_classes

    def train_step(self, images, labels, optimizer):
        # Inner max: same random-start L_inf PGD structure as torchattacks.PGD.forward.
        self.model.eval()
        adv_images = images.clone().detach()

        adv_images = adv_images + torch.empty_like(adv_images).uniform_(
            -self.eps, self.eps
        )
        adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()

        for _ in range(self.attack_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(adv_images)
            cost = F.cross_entropy(outputs, labels)  # maximize true-label CE
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()

        # Outer min: freeze the adversarial batch and take one ordinary SGD step.
        self.model.train()
        outputs = self.model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {'loss': loss.item()}
```
