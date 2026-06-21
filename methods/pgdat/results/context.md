# Context: training image classifiers that resist adversarial perturbations (circa 2016-2017)

## Research question

A modern image classifier trained by empirical risk minimization reaches near-human accuracy on
natural inputs, yet there exist inputs `x_adv` that are visually indistinguishable from a correctly
classified `x` — differing in each pixel by an amount far below the sensor's precision — that the
network confidently misclassifies. These adversarial examples are not rare corner cases: efficient
algorithms find one for almost any input, and a single perturbed image often fools many different
architectures at once. With classifiers moving into security-critical systems (autonomous driving,
face recognition, malware detection), this is a real vulnerability, and it also signals that the
models are not learning the underlying concept in a robust way.

The question is how to train a classifier that is reliably accurate under adversarial perturbations
within a fixed budget, together with a way to measure and compare robustness across methods.

## Background

The phenomenon was discovered by Szegedy et al. (2014, *Intriguing properties of neural networks*):
using box-constrained L-BFGS one can find, for almost any correctly classified image, a tiny additive
perturbation that flips the label, and the same perturbed image transfers across models trained on
different data. They already observed that *training on a mixture of adversarial and clean examples
regularizes the model* — generating those examples with L-BFGS was too expensive to use as an
inner loop, so adversarial training was not practical at the time.

Goodfellow et al. (2015) reframed the cause. Their **linear explanation**: for a weight vector `w`
and perturbation `η`, the activation shifts by `wᵀη`; under the constraint `||η||_∞ ≤ ε` the worst
case is `η = ε·sign(w)`, which grows the activation by `ε·m·n` for `n` input dimensions of average
weight magnitude `m`. The perturbation's size does not grow with `n`, but its effect on the output
does — so in high dimensions many imperceptible per-pixel changes add up to a large change in the
logits. Adversarial examples are then not a mysterious nonlinearity artifact but a consequence of
models being *too linear*. This view singles out the `ℓ_∞` ball as the natural attack set and points
at the sign of the input gradient as the worst-case direction.

This established the **`ℓ_∞` threat model**: the adversary may add any `δ` with `||δ||_∞ ≤ ε` (with
pixels then clipped to the valid `[0,1]` range), where `ε` is small enough to be perceptually
negligible — `ε = 0.3` is standard on MNIST, `ε = 8/255` on CIFAR-10. There is also a longer
lineage outside deep learning: robust optimization, where one optimizes against the worst case in an
uncertainty set, goes back to Wald (1945); and adversarial machine learning predates deep nets
(Dalvi et al. 2004; Globerson & Roweis 2006).

Several **motivating empirical facts about existing systems** were on the table and are load-bearing
for any new method:

- A normally trained convolutional MNIST classifier reaches ~99% clean accuracy but its accuracy
  under a one-step `ℓ_∞` attack collapses to single digits. The vulnerability is the rule, not a
  tail event.
- Models adversarially trained against a *one-step* attack become robust to that attack but remain
  vulnerable to iterative attacks (Tramèr et al. 2017): one-step robustness does not imply
  robustness to stronger adversaries.
- One-step adversarial training can induce **label leaking** (Kurakin et al. 2017): the one-step
  adversary produces a restricted, predictable set of perturbed inputs that the network overfits to.
- On ImageNet, larger-capacity networks were observed to tolerate adversarial training better
  (Kurakin et al. 2017), hinting that model capacity matters for robustness.

## Baselines

**Standard ERM training.** Minimize `E_{(x,y)}[L(θ,x,y)]` with SGD. State of the art on clean
accuracy.

**Fast Gradient Sign Method and FGSM adversarial training (Goodfellow et al. 2015).** Generate an
adversarial example in one step by moving every pixel along the sign of the input gradient,
`x_adv = x + ε·sign(∇_x L(θ,x,y))`, and train on an objective that mixes clean and adversarial loss,
`J̃(θ,x,y) = α·J(θ,x,y) + (1-α)·J(θ, x + ε·sign(∇_x J)),` with `α = 0.5`. Cheap enough to run inside
the training loop, and it improves robustness to one-step attacks.

**Defensive distillation, feature squeezing, and adversarial-example detectors (Papernot et al.
2016; Xu et al. 2018; Carlini & Wagner 2017).** A family of defenses that either smooth the model's
output surface, reduce input precision, or add a detector that flags suspicious inputs. Each is built
and evaluated against specific attacks.

**Min-max formulations with a weak inner solver (Huang et al. 2015; Shaham et al. 2018).** These
formulated the defense as a worst-case (min-max) objective and used one-step adversaries to
approximate the inner worst-case search, evaluating robustness against FGSM.

## Evaluation settings

The natural yardsticks already in use:

- **MNIST** (LeCun et al. 1998): 28×28 grayscale digits, 10 classes, pixels in `[0,1]`, with a small
  convolutional network (e.g. two conv layers of 32 and 64 filters, each with 2×2 max-pooling, then
  a 1024-unit fully connected layer). Standard `ℓ_∞` budget `ε = 0.3`.
- **CIFAR-10** (Krizhevsky 2009): 32×32 color images, 10 classes, with a residual network (He et al.
  2016) and standard augmentation (random crops, horizontal flips, per-image standardization).
  `ℓ_∞` budget `ε = 8/255`. CIFAR-100 (100 classes) is the same setting at higher class count.
- **Optimizer / schedule**: SGD with momentum and weight decay; learning rate annealed over training
  (e.g. cosine). Minibatch training, fixed initialization across compared methods.
- **Threat models for evaluation**: white-box attacks where the adversary has the gradients (one-step
  FGSM, and multi-step iterative attacks with varying numbers of steps and random restarts; attacks
  using the Carlini–Wagner logit-margin loss); and black-box / transfer attacks from an independent
  copy, from a naturally trained copy, and from a different architecture.
- **Metrics**: clean accuracy on unperturbed test images; accuracy under a one-step attack; and
  accuracy under a strong multi-step iterative `ℓ_∞` attack with many steps.

## Code framework

The new procedure plugs into the same minibatch training harness already used for ERM. Everything in
the harness already exists — a differentiable classifier `model`, a cross-entropy loss, an SGD
optimizer with its learning-rate schedule, and the data loader feeding clean images in `[0,1]` with
their labels. The training loop and schedule are fixed and external. The editable surface is one
class with a fixed constructor and one training-step method. The empty slot is the body of that
step: given a clean batch and an optimizer, choose the batch and loss used for the parameter update.

```python
import torch
import torch.nn.functional as F


class AdversarialTrainer:
    """Owns the editable single-batch training rule."""

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.attack_steps = attack_steps
        self.num_classes = num_classes

    def train_step(self, images, labels, optimizer):
        # images: (N, C, H, W) in [0, 1];  labels: (N,)
        # optimizer: SGD, already configured with lr / momentum / weight_decay
        # TODO: choose the batch and loss used for this parameter update.
        pass


# existing minibatch training loop the trainer plugs into (fixed, external)
def train(model, trainer, data_loader, optimizer, scheduler):
    model.train()
    for images, labels in data_loader:       # clean batch, pixels in [0, 1]
        stats = trainer.train_step(images, labels, optimizer)
        scheduler.step()                     # cosine LR annealing, external
```

The loop hands one clean batch per step to `train_step`; that method is where the procedure lives.
