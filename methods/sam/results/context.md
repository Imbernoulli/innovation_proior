# Context

## Research question

Modern neural networks are heavily overparameterized: they have enough capacity to fit, and even memorize, the entire training set, driving the training loss to essentially zero. In this regime the value of the training loss says almost nothing about how the model will behave on unseen data. The training loss landscape is non-convex with a multiplicity of local and global minima, and different minima that achieve the *same* low training loss can generalize very differently. The standard recipe — pick an optimizer (SGD, Adam, RMSProp) and minimize the empirical loss `L_S(w)` (perhaps with weight decay) — is therefore geometry-blind: it drives `w` into *some* basin of low training loss, but it exercises no control over *which* basin, and so no direct control over test-time performance.

The question is whether the training procedure itself can be made to seek out parameter regions that are likely to generalize, scalably and as a drop-in replacement for the usual optimizer. A satisfactory answer would (a) start from a principled, quantifiable connection between the geometry of the loss surface around `w` and the generalization gap; (b) turn that connection into an objective that can be minimized by ordinary gradient descent; (c) cost only a small constant factor more than standard training (no materializing or inverting a Hessian, which is intractable for models with millions to billions of parameters); and (d) require essentially no architecture changes, so it can be applied on top of existing, already-regularized models.

## Background

**The flat-minima hypothesis.** A line of work argues that the *width* of the basin a solution sits in is tied to how well it generalizes. Hochreiter & Schmidhuber (1995, 1997) made this precise through a minimum-description-length argument: a minimum is "flat" if the loss stays low over a large, low-precision box of weights around it; such a minimum needs fewer bits to specify, and by an MDL/Occam argument should generalize better than a sharp minimum where the loss spikes up under tiny weight changes. They proposed an explicit "flat minimum search" that drives optimization toward such regions, but it requires quantities tied to the second derivatives of the loss and is awkward to compute and differentiate through at the scale of a deep network.

**Large-batch training and sharpness.** Keskar et al. (2016) studied why large-batch SGD generalizes worse than small-batch SGD despite reaching the same training loss. Their diagnostic finding: large-batch training converges to *sharp* minima — points where the loss has large positive curvature (large Hessian eigenvalues) and rises steeply in a small neighborhood — whereas small-batch noise pushes the iterate into *flat* minima. They quantified this with a sharpness metric defined as the maximum increase of the loss inside a small box around `w`, i.e. `max_{‖ε‖≤ρ} L(w+ε) − L(w)`. This is a *measurement* of an existing phenomenon, not a training procedure, and the metric (a constrained inner maximization, or a top Hessian eigenvalue) is itself expensive.

**Empirical correlation of sharpness with the generalization gap.** Jiang et al. (2019), in a large-scale study of 40 candidate complexity measures across thousands of trained models, found that sharpness-based measures correlate most strongly with the actual generalization gap — more strongly than norm-based or margin-based measures. This is the strongest available empirical evidence that *if* one could push training toward lower sharpness, one would expect better generalization; it is a fact about already-trained models, established before any new procedure is built.

**PAC-Bayes generalization bounds.** PAC-Bayes (McAllester, 1999) bounds the expected population loss of a *stochastic* predictor — one whose weights are drawn from a posterior distribution `Q` — in terms of its expected training loss plus a complexity term `sqrt((KL(Q‖P) + log(n/δ)) / (2(n−1)))`, where `P` is a data-independent prior. Dziugaite & Roy (2017) and Neyshabur et al. (2017) instantiated this for neural networks by taking `P` and `Q` to be Gaussians centered at the origin and at the trained weights `w`. The key consequence: the bound's training term becomes the loss *averaged over Gaussian perturbations of the weights*, `E_{ε∼N(0,σ²I)}[L_S(w+ε)]`, which is large exactly when the loss surface around `w` is sharp. The bound thus formally ties a perturbed-loss quantity to the population loss, but it is a bound, not an algorithm, and the perturbation is averaged (random), not worst-case.

**Adversarial input perturbations.** A separate body of work studies robustness to adversarial *input* perturbations (Goodfellow et al., 2015, the fast gradient sign method): given a fixed network, one searches for a small bounded perturbation `δ` of the *input* `x` that maximizes the loss, `argmax_{‖δ‖≤ρ} L(x+δ)`, to expose or harden against worst-case inputs. This is a line of work about input-space robustness, established independently of any training-geometry objective.

**Stochastic gradient descent.** The base procedure throughout is minibatch SGD (Robbins & Monro, 1951): sample a batch, compute the gradient of the batch loss by backpropagation, step `w ← w − η g`. Momentum, weight decay, and learning-rate schedules (e.g. cosine decay) are standard add-ons. Whatever new objective is proposed must ultimately be reducible to "compute a gradient, hand it to SGD/Adam."

## Baselines

- **SGD / Adam on the empirical loss.** The default: `min_w L_S(w)` (+ optional `λ‖w‖²` weight decay), minimized by SGD with momentum or Adam. Core algorithm: one gradient evaluation per step, `w ← w − η ∇L_S(w)`. Gap: minimizes only the loss *value* at `w`; it is indifferent to the curvature around `w`, so among many minima with equal training loss it provides no pressure toward the flatter, better-generalizing ones.

- **Flat-minimum search (Hochreiter & Schmidhuber).** Explicitly augments the loss with a term penalizing sharpness, derived from an MDL objective involving the loss's second derivatives. Gap: the sharpness penalty involves Hessian-related quantities that are expensive to evaluate and to differentiate through; it has not been shown to scale to or improve generalization on large modern models.

- **Entropy-SGD / local-entropy methods (Chaudhari et al., 2016).** Replace the loss by a smoothed "local entropy" objective that favors wide valleys, optimized with an inner Langevin/SGLD loop. Gap: the inner loop adds substantial cost per outer step, and the smoothing is over a randomized neighborhood rather than a directly optimized worst case.

- **Weight averaging (SWA, Izmailov et al., 2018).** Rather than penalize sharpness, average the weights visited along an SGD trajectory; the average tends to sit in a flatter region and generalizes better. Gap: it does not *optimize* a sharpness objective — flatness is an incidental consequence of averaging — and it offers no per-step signal pushing the iterate toward flat regions.

- **Random/Gaussian weight perturbation.** Inject Gaussian noise into the weights during training (related to the PAC-Bayes posterior). Gap: the perturbations are sampled at random and their effect is averaged, so the procedure controls an average over a randomized neighborhood rather than acting on any directed feature of the local loss surface.

## Evaluation settings

The natural yardsticks are standard supervised image-classification benchmarks and their established models and protocols (these datasets, architectures, and metrics all predate any new procedure):

- **Datasets:** CIFAR-10, CIFAR-100, SVHN, Fashion-MNIST for from-scratch training; ImageNet for large-scale training; and a suite of finetuning targets (Flowers, Oxford-IIIT Pets, Stanford Cars, FGVC-Aircraft, Birdsnap, Food-101). A noisy-label variant of CIFAR-10 (a fraction of training labels randomly flipped, clean test set) tests robustness to label noise.
- **Models:** WideResNet (with Shake-Shake), PyramidNet (with ShakeDrop), ResNet-{50,101,152}, EfficientNet-{b7,L2}; smaller ResNets for diagnostics.
- **Augmentation regimes:** basic (flip / pad / crop), Cutout, and AutoAugment.
- **Metric:** top-1 (and top-5 on ImageNet) test error / accuracy, reported as mean and 95% confidence interval over several independent replicas.
- **Protocol notes:** if a procedure costs more gradient evaluations per step than the baseline, it should be compared against a baseline allowed proportionally more epochs, to equalize gradient-computation budget. Diagnostics include the Hessian eigenvalue spectrum at convergence (approximated by the Lanczos algorithm) as a direct readout of sharpness.

## Code framework

The primitives that already exist: a data pipeline yielding minibatches, a model with a per-batch loss, automatic differentiation for gradients, and a base optimizer (SGD with momentum, or Adam) that consumes a gradient and applies an update. The new procedure occupies the slot between "we have a batch gradient" and "we apply an update." A natural way to express it is as a wrapper around the base optimizer; to allow more general update rules, the training loop exposes the forward/backward as a closure that the optimizer may invoke.

```python
import torch

class GeometryAwareOptimizer(torch.optim.Optimizer):
    """Wraps a base optimizer; the update rule is the slot to be filled."""
    def __init__(self, params, base_optimizer, **kwargs):
        defaults = dict(**kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        # TODO: define the update rule here.
        raise NotImplementedError


def train_step(model, batch, optimizer, loss_fn):
    x, y = batch

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        return loss

    # standard baseline would be: closure(); optimizer.step()
    loss = closure()
    optimizer.step(closure)
    return loss
```
