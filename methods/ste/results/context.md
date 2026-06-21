# Context: gradients through hard, discrete decisions in a neural network (circa 2012-2013)

## Research question

We want a neural network to make a *hard, discrete decision* inside its computation — most
sharply, a unit whose output is a single bit, on or off — and we still want to train the whole
thing end to end by gradient descent. The cleanest example is a gating unit for **conditional
computation**: a cheap unit decides, per example, whether a much more expensive block of the
network is computed at all. If those gates produce genuine zeros most of the time, large chunks of
the network can be skipped, and the cost per example drops from "visit every parameter" to "visit
only the activated subset." The same hard-decision unit is wanted elsewhere — binary codes as hash
keys, hard decisions about temporal events at different time scales in recurrent nets, sparse
distributed representations that act as a regularizer.

A bit-valued output is the indicator of a threshold being crossed. That operation is **flat almost
everywhere** (its output does not change as the pre-activation wiggles, except at the single
threshold where it jumps), so its honest derivative with respect to the pre-activation is **zero
almost everywhere** (and undefined at the threshold). Composed into the chain rule, a zero factor
annihilates the gradient: no signal reaches the unit's incoming weights and bias, and no signal
reaches anything feeding into it. The question is how to get a training signal to the parameters
driving a hard, non-smooth, possibly stochastic discrete unit.

## Background

The field's working assumption, inherited from the move that made deep learning trainable, is that
the relationship between parameters and the training objective should be **continuous and generally
smooth** for exact gradient-based learning to work well. This is exactly why early networks of
hard-threshold "formal neurons" were abandoned in favor of sigmoidal units, which the
back-propagation algorithm (Rumelhart, Hinton & Williams 1986) can differentiate cleanly.

By 2012-2013 that landscape was shifting. Networks built on **rectifiers** — `h = max(0, a)`,
non-smooth at the kink — were winning (Glorot, Bordes & Bengio 2011; Krizhevsky, Sutskever & Hinton
2012; the maxout nets of Goodfellow et al. 2013). The rectifier's "derivative" is the indicator
`1_{a>0}`: on the backward sweep it zeroes exactly the units that were off on the forward sweep, and
otherwise passes the gradient through with slope one. A non-smooth unit with a piecewise-constant
*slope* trains fine in practice.

A second relevant body of practice is **noise injection in an otherwise smooth graph**. Dropout
(Hinton et al. 2012) multiplies a unit's output by independent binomial noise; denoising autoencoders
(Vincent et al. 2008) corrupt inputs with masking noise; semantic hashing (Salakhutdinov & Hinton
2009) adds Gaussian noise before a sigmoid and lets the growing weights push the sigmoid toward
saturation, yielding near-binary codes. In all of these, gradients flow as usual, because the noise
enters a part of the graph that is still a smooth function of the activation — the noise is just a
fixed multiplier or addend on the backward path. Noise also seems to help: dropout slows training
only mildly (~2x) and acts as a regularizer; the symmetry-breaking and induced sparsity may even ease
ill-conditioning.

Some structural facts about the hard-binary unit are worth stating. Take the unit to fire as a
sigmoid-parameterized Bernoulli: with `z ~ U[0,1]`, output `h = 1_{z < sigm(a)}`, so
`P(h=1) = sigm(a)` (a smooth, differentiable function of the pre-activation `a`, even though the
realized sample is hard). The *firing probability* has a clean derivative
`d sigm(a)/da = sigm(a)(1 - sigm(a))`; the *realized bit* does not. A useful identity: because
`h ∈ {0,1}`, any loss factors as `L(h) = h·L(1) + (1-h)·L(0)`, and `h² = h`, `h(1-h) = 0`. A unit
that is off contributes a hard zero to whatever it gates, and a hard zero is what makes conditional
computation pay off — so the forward output should be a true zero a large fraction of the time, not
a small continuous value.

## Baselines

These are the routes already available for getting a gradient-like signal to the parameters of a
hard or stochastic unit.

**Finite-difference gradient estimation.** Perturb one parameter at a time and measure the change in
loss: `(L(u + ε e_i) - L(u - ε e_i)) / 2ε`. With `N` parameters and `O(N)` cost per loss
evaluation, this is `O(N²)` in total.

**Simultaneous Perturbation Stochastic Approximation — SPSA (Spall 1992).** Instead of perturbing
coordinates one at a time, perturb *all* parameters at once with a single random vector `z` and
estimate `dL/du_i ≈ (L(u + z) - L(u - z)) / (2 z_i)`. Two function evaluations regardless of
dimension. A closely related spiking-neuron estimator appears in Fiete & Seung (2006).

**Likelihood-ratio / REINFORCE estimation (Williams 1992).** Treat the unit's output as a sampled
stochastic action `h` drawn with probability `p_θ(h)`, receiving a reward (negative loss) `R`. The
likelihood-ratio identity gives, for any baseline `b`,
`E_h[(R - b)·∂ log p_θ(h)/∂θ] = ∂ E_h[R]/∂θ` — an **unbiased** estimate of the gradient of the
expected reward. For a Bernoulli-logistic unit with `p_i = sigm(a_i)`, the characteristic eligibility
is `∂ log p / ∂a_i = (y_i - p_i)`, so the per-parameter update is proportional to
`(R - b)(y_i - p_i) x_j`. The baseline `b` shifts variance, not bias; its variance-optimal value is a
particular weighted average of the loss values.

**Anneal a smooth unit into a hard one.** Rather than confront the hard threshold directly, keep a
smooth (sigmoid) unit and gradually force it toward binary behavior. Semantic hashing (Salakhutdinov
& Hinton 2009) adds Gaussian noise before the sigmoid of a code unit; to avoid losing information
through the noise, the weights and biases grow, which pushes the sigmoid into saturation, so its
output drifts toward a near-0 or near-1 value that can be thresholded at the end of training.
Equivalently one can anneal the magnitude of the parameters.

## Evaluation settings

- **Conditional-computation gater on MNIST.** A network with one conditional layer over vectorized
  28×28 MNIST digits: a gater subnetwork (a small hidden layer, e.g. 400 tanh units, then an affine
  map to many gating units, e.g. 2000) whose outputs gate an equally wide expert path elementwise
  (`H_i · h_i`). A **sparsity constraint** drives the average gater firing rate to a target (e.g.
  10%), via a KL-divergence penalty for sigmoid gaters or an L1 penalty for rectifier gaters, with
  the penalty weight adapted to hold the target rate. Metric: classification error on validation/test,
  alongside the training criterion; the discrete gaters are evaluated deterministically at test time
  by thresholding to hit the same target firing rate. Computational protocol: compute the cheap gater
  first, and only compute the expensive expert `H_i` where `h_i ≠ 0`, so a 10% firing rate turns an
  `O(N²)` main path into roughly `O(0.1 N²)`.

## Code framework

The substrate is an ordinary autodiff network and training loop: an affine pre-activation
`a = W x + b`, a unit that maps `a` to an output `h`, downstream layers, a loss, and `loss.backward()`
filling parameter gradients for an optimizer step. Everything here already exists except the hard
decision unit itself. The scaffold leaves one generic slot for that unit and keeps the training loop
unchanged.

```python
import torch
import torch.nn as nn


def decision_unit(a):
    # TODO: the hard / discrete unit we will design
    raise NotImplementedError


class ConditionalGate(nn.Module):
    def __init__(self, in_features, num_gates):
        super().__init__()
        self.affine = nn.Linear(in_features, num_gates)

    def forward(self, x):
        a = self.affine(x)
        h = decision_unit(a)
        return h


# existing training loop the unit plugs into
def train(model, loss_fn, data_loader, optimizer):
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)          # forward; somewhere inside, decision_unit(a) is applied
        loss = loss_fn(outputs, targets) # task loss (may add a sparsity penalty on firing rates)
        loss.backward()                  # autodiff handles the existing differentiable parts
        optimizer.step()
```

The empty slot is the unit that produces the hard decision. The rest of the harness is untouched.
