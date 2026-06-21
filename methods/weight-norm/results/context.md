# Context: conditioning and parameterization of first-order training for deep networks

## Research question

Deep neural networks are trained, overwhelmingly, by first-order stochastic gradient
methods. The practical success of those methods is highly sensitive to the *curvature* of
the loss surface: when the Hessian at a solution is ill-conditioned (a large ratio between
its largest and smallest eigenvalues), gradient descent zig-zags across the steep
directions and crawls along the flat ones, and progress per step is poor. A crucial fact is
that this curvature is **not invariant to how the model is parameterized**. The very same
input-output function can be written with different internal parameters, and some of those
parameterizations give a better-conditioned optimization problem than others. Two
parameterizations that compute identical functions can therefore differ in how fast SGD
converges on them.

The question: take the elementary building block that almost every architecture is
made of — a neuron that computes a weighted sum of its inputs plus a bias, followed by a
pointwise nonlinearity, y = φ(w·x + b) — and find a *reparameterization* of its weights
that improves the conditioning of the gradient and so speeds up first-order optimization.

## Background

**Curvature, conditioning, and reparameterization.** First-order methods make fast progress
only when the loss is well-conditioned
(Martens 2010; Sutskever et al. 2013 on the importance of initialization and momentum).
The amount of curvature seen by the optimizer depends on the coordinates in which the
parameters are expressed (Amari 1997): there are many equivalent ways to parameterize the
same model, some far easier to optimize than others. This makes "find a better
parameterization" a legitimate and powerful lever, distinct from changing the optimizer.

**The natural gradient and its approximations.** The ideal preconditioner is the inverse
Fisher information matrix: left-multiplying the gradient by it yields the *natural
gradient*, which is invariant to reparameterization and effectively whitens the update. The
obstacle is cost — the Fisher is huge and must be estimated and inverted. A line of work
builds tractable approximations: a Kronecker-factored approximation to the Fisher that can
be inverted block-wise (KFAC, Martens & Grosse 2015); a sparse approximate Cholesky
factorization of the inverse Fisher (FANG, Grosse & Salakhutdinov 2015); and whitening the
input to each layer so that plain gradients become approximately natural (PRONG,
Desjardins et al. 2015). All of these pay a real price in computation and bookkeeping for
estimating curvature.

**Getting the natural-gradient effect by reparameterizing instead.** A cheaper alternative
is to leave the optimizer as plain first-order SGD but change the model's parameterization
so that ordinary gradients already look like whitened natural gradients. Raiko et al. (2012)
transform each neuron's output to have, on average, zero value and zero slope; they show
this approximately diagonalizes the Fisher information matrix and thereby whitens the
gradient, improving optimization. This establishes the template: a well-chosen
reparameterization can buy much of the conditioning benefit of natural-gradient methods at
almost no cost.

**Initialization controls early-training feature scales.** Analytic initialization schemes
(Glorot & Bengio 2010; He et al. 2015) set initial weight scales so that activations and
gradients are well-scaled at the start, derived under assumptions on the feature
distributions. A complementary idea is *data-dependent* initialization: run one minibatch
through the network and set the per-layer scales and offsets from the measured pre-activation
statistics, so that every layer starts with unit-variance, zero-mean pre-activations
(proposed concurrently by Mishkin & Matas 2015 (LSUV) and Krähenbühl et al. 2015).

## Baselines

**Batch normalization (Ioffe & Szegedy 2015).** For each neuron, take the pre-activation
t = v·x and standardize it using statistics computed over the current minibatch:

    t' = (t − μ[t]) / σ[t],

where μ[t] and σ[t] are the mean and standard deviation of t over the examples in the
minibatch; a learnable scale and shift are then reapplied. This reduces the shift in the
distribution of each neuron's inputs during training and is argued to bring the Fisher
matrix closer to the identity, which is why it accelerates training and tolerates larger
learning rates.

**Approximate-natural-gradient preconditioners (KFAC, FANG, PRONG).** Core idea: form a
tractable approximation to the (inverse) Fisher and use it to precondition the gradient,
approximating the natural gradient.

**Output-reparameterization (Raiko et al. 2012).** Core idea: transform neuron outputs to
zero average value and zero average slope, approximately diagonalizing the Fisher. This
operates on the activations rather than on the weight coordinates in which the optimizer
takes steps.

**Norm-constrained weights (max-norm; Srebro & Shraibman 2005).** Core idea: keep the norm
of each weight vector controlled. The optimization is carried out in the original weight
coordinates w, with the norm constraint applied after each SGD step as a projection.

## Evaluation settings

Natural yardsticks of the time, spanning the domains where the block appears:

- **Supervised image classification.** CIFAR-10 (Krizhevsky & Hinton 2009), 32×32 natural
  images, 10 classes, without data augmentation; an all-convolutional architecture in the
  style of ConvPool-CNN-C / ALL-CNN-C (Springenberg et al. 2015) with leaky-ReLU
  convolutions, max-pooling, dropout/Gaussian-noise regularization, global average pooling
  and a softmax head; optimized with Adam/Adamax (Kingma & Ba 2014). Metric: test
  classification error; also training-loss-versus-epoch curves to read off optimization
  speed, and wall-clock cost per epoch.
- **Generative modelling.** Convolutional variational autoencoders (Kingma & Welling 2013;
  Rezende et al. 2014; Salimans et al. 2015) with residual blocks (He et al. 2015) on MNIST
  and CIFAR-10; and DRAW (Gregor et al. 2015), a recurrent VAE built from LSTM units
  (Hochreiter & Schmidhuber 1997) on MNIST. Metric: the variational lower bound on the test
  log-likelihood versus training epoch, with error bars over random seeds.
- **Deep reinforcement learning.** Deep Q-Networks (Mnih et al. 2015) on Atari games in the
  Arcade Learning Environment (Bellemare et al. 2013) — Breakout, Enduro, Seaquest, Space
  Invaders. Metric: evaluation score versus training epoch and final score, a regime where
  the gradient noise from minibatch statistics is known to destabilize learning.

## Code framework

The ordinary `nn.Module` layer, optimizer, loss, and training loop are already enough: a
layer stores a raw weight tensor and bias, the optimizer steps whatever parameters the
layer registers, and a `register_forward_pre_hook` can replace the tensor a layer uses
before each forward call — so a layer's trainable parameters need not be the same as the
tensor that finally multiplies the input. That is the only affordance the implementation
needs; the slot below is where the proposed change goes.

```python
import torch
import torch.nn as nn


def apply_to_layer(module, name="weight", dim=0):
    # TODO: implement the proposed change to the layer here.
    pass
```
