# Context: keeping signal propagation well-conditioned in deep convolutional networks (circa 2013-2016)

## Research question

A deep convolutional or recurrent network is, structurally, a long composition of linear maps
interleaved with nonlinearities. A forward activation is multiplied by one weight matrix after
another on its way to the output; a backpropagated error is multiplied by the (transposed)
weight matrices on its way back to the early layers. Both passes are therefore governed by the
*spectrum* of the per-layer matrices and of their product: if those matrices systematically
shrink vectors, signals and gradients vanish with depth; if they systematically stretch
vectors, signals and gradients explode. The pain point is concrete and well documented — deep
and recurrent nets are hard to train precisely because this repeated multiplication drives the
signal away from an `O(1)` scale.

There is a known square-weight configuration that makes a layer exactly norm-preserving, and a
known way to *start* training from it. The open problem is different and sharper: that good
configuration is only imposed at initialization, and there is nothing holding the weights there.
As soon as the data gradient starts moving the weights, they drift, and the flat-spectrum
property erodes over the course of training. What is wanted is a way to keep convolutional
filter banks close to the feasible semi-orthogonal configuration *throughout* training, not just
at step zero — and to do it cheaply: as an extra term in the loss that the existing optimizer
already minimizes, differentiable or subdifferentiable, with no matrix factorization in the
inner loop, no change to the architecture, the base objective, or the training procedure, and
applicable to the rectangular weight matrices that convolutional layers actually have. Such a
method would, plausibly, also improve generalization, to the extent that a well-conditioned
weight spectrum keeps more of the model's nominal capacity usable rather than letting it degrade
over training.

## Background

**Why depth makes signal scale fragile.** Pascanu, Mikolov & Bengio (ICML 2013) made the
mechanism explicit for recurrent nets, which are feedforward nets with tied weights unrolled in
time. Writing the gradient of a loss at step `t` with respect to an earlier state `k` as a
product of Jacobians,

```
d x_t / d x_k  =  prod_{t >= i > k}  W_rec^T diag(sigma'(x_{i-1})),
```

the 2-norm of the whole product is bounded by the product of the per-factor 2-norms,
`|| W_rec^T || · || diag(sigma') ||`, and `|| diag(sigma') || <= gamma` where `gamma = sup |sigma'|`
(`gamma = 1` for `tanh`, `1/4` for the logistic sigmoid). Hence a *sufficient* contraction
condition for the long-term gradient to vanish is `||W_rec||_2 < 1/gamma`: then each factor
contracts by a fixed ratio `< 1` and the product decays geometrically with `t - k`. In the
linear/spectral-radius analysis this appears as `lambda_1 < 1` for vanishing, while
`lambda_1 > 1` (or, in the bounded-nonlinearity scaling, above `1/gamma`) is a *necessary* but
not by itself sufficient condition for exploding gradients. The same product-of-matrices logic
applies to the depth direction of a plain feedforward net. Their remedy for exploding gradients
is to clip the gradient norm; for vanishing gradients they suggest a soft term that encourages
the backpropagated signal to keep its norm. The lesson that carries over: the spectrum and
operator norms of the weight matrices are the levers that control whether deep signal
propagation is stable.

**The configuration with a flat spectrum.** A square matrix whose singular values are all exactly
`1` is exactly norm-preserving: `|| W x || = || x ||` for every `x`. These are the orthogonal
matrices, characterized equivalently by `W^T W = I` (columns orthonormal) or `W W^T = I` (rows
orthonormal). A rectangular matrix can only be semi-orthogonal: all singular values on the
feasible side are `1`, while the longer side has unavoidable null directions. Saxe, McClelland &
Ganguli (ICLR 2014) studied the learning dynamics of deep linear networks and drew out exactly
why the square flat-spectrum case matters at depth.
Glorot & Bengio (2010) had already proposed scaling random Gaussian weights so that, *on
average*, norms are preserved — for an `N x N` matrix, drawing entries i.i.d. from a zero-mean
Gaussian with standard deviation `1/sqrt(N)` gives `< v^T W^T W v > = v^T v`. Saxe et al.
pointed out the catch: norm preservation *in expectation* is not the same as a flat spectrum.
The squared singular values of a scaled Gaussian matrix follow the Marchenko-Pastur
distribution, which has a nontrivial spread that does not vanish even as `N -> infinity`; and a
*product* of such matrices across many layers develops a highly kurtotic singular spectrum —
most singular values collapse toward zero while a long tail grows very large. Such a product
still preserves the norm of a typical vector, but anisotropically: it strongly amplifies a few
directions and crushes all the others, so error vectors projected onto the crushed subspace are
attenuated to nothing. A product of *orthogonal* matrices, by contrast, is itself orthogonal,
so every singular value stays exactly `1` no matter the depth — what they named *dynamical
isometry*. Empirically, initializing each layer with a random orthogonal matrix (e.g. the `Q`
factor of a QR decomposition of a random Gaussian) gave depth-independent learning times on
MNIST, matching greedy layer-wise pretraining, while scaled Gaussian init did not. The property
survives into nonlinear networks operating just past the "edge of chaos" (gain `g_c = 1` for
`tanh`), where the end-to-end Jacobian still acts as a near-isometry.

**The diagnostic gap these findings leave.** Both of the above are facts about the *spectrum of
the weights*, and the orthogonal-initialization result is a fact about *step zero only*. A good
initial condition does not pin the weights in place: once the data-driven gradient starts
updating them, there is no force keeping the matrices orthogonal, and the flat-spectrum property
that made early learning fast erodes as training proceeds. This is an observed limitation of the
initialization approach, not a prescription — the weights leave the orthogonal configuration and
nothing measures or resists that drift.

**Adjacent regularizers of the time.** Standard L2 weight decay adds `(lambda/2) || W ||^2` to
the loss, i.e. it shrinks the Frobenius norm of the weights toward zero. It is the default
generalization aid, but it pulls every singular value *down*, which is the opposite of holding
them at `1`. Batch Normalization (Ioffe & Szegedy, 2015) stabilizes the *distribution* of
layer activations by normalizing them, indirectly easing signal scale, but it operates on
activations rather than on the weight spectrum and does not by itself flatten the singular
values of a weight matrix. None of these tools maintains the specific property — singular values
near `1` — that the propagation analysis identifies as the thing that matters.

## Baselines

These are the prior approaches a new method for sustaining a flat weight spectrum would be
measured against and would react to.

**Random orthogonal initialization (Saxe et al. 2014; Glorot & Bengio 2010 for the scaled
Gaussian it improves on).** Set each weight matrix at the start of training to a random
orthogonal matrix, obtained as the orthogonal factor of a QR (or SVD) decomposition of a random
Gaussian matrix. Core idea: enter training already on the orthogonal manifold, with all singular
values exactly `1`, so the initial forward/backward map is exactly norm-preserving and learning
is depth-independent. *Gap:* it is a one-time condition. There is no mechanism that keeps the
weights orthogonal after the first gradient step; the matrices drift off the manifold as the
data loss pulls them, and the flat spectrum it bought at step zero is gone by mid-training. It
constrains nothing during the part of training where most of the weight movement happens.

**Hard orthogonality via Stiefel-manifold / Riemannian optimization.** Treat orthogonality as a
hard constraint: confine each weight matrix to a Stiefel manifold such as `{ W : W^T W = I }` or
`{ W : W W^T = I }`, whichever side is feasible, and optimize on the manifold using Riemannian
gradients, or re-orthogonalize the weights after each update by an explicit factorization (QR or
SVD) projecting back onto the manifold. Core idea: never let the weights leave the selected
semi-orthogonal set at all, so the feasible side's spectrum is exact at every step. *Gap:* the
per-step cost is heavy — a manifold step or an explicit SVD/QR of every weight matrix on every
iteration is expensive on a GPU and grows badly with matrix size. And the hard constraint is
awkward for convolutional layers, whose weights, when reshaped to a matrix, are strongly
rectangular (`out_channels` versus `in_channels * k * k`); square orthogonality cannot hold
simultaneously for the rows and the columns of a non-square matrix, so the constraint has to be
restricted to whichever side is feasible, complicating the procedure.

**Gradient clipping (Pascanu et al. 2013).** Rescale the gradient when its norm exceeds a
threshold, capping explosive steps. Core idea: directly bound the size of the parameter update so
an exploding-gradient event cannot blow up training. *Gap:* it is a reactive cap on the
*gradient*, applied after the fact, not a property of the *weights*; it does nothing about
vanishing gradients and does nothing to flatten or maintain the singular value spectrum of the
weight matrices that caused the instability.

**L2 weight decay.** Add `(lambda/2) || W ||_F^2` to the objective; the optimizer shrinks weights
toward the origin. Core idea: penalize weight magnitude to improve generalization. *Gap:* it
drives the spectrum toward zero (all singular values smaller), the opposite of the target of
keeping them near `1`; it controls overall scale, not the *shape* of the spectrum, and on its own
allows filters to become small or mutually redundant.

## Evaluation settings

The natural yardsticks already in use for a weight regularizer on convolutional image
classifiers, all pre-existing facts about the world:

- **CIFAR-100** (Krizhevsky & Hinton, 2009): 32x32 natural images, 100 classes, standard data
  augmentation (random crops with reflection padding, horizontal flips). Test error of a
  convolutional classifier trained for a fixed schedule is the headline metric.
- **CIFAR-10 / SVHN / CelebA** for adjacent image tasks (image classification, attribute
  classification): same 32x32-scale convolutional regime; validation/test error as the metric.
- Backbone architectures of the period: deep convolutional classifiers such as residual and
  densely-connected networks, with Batch Normalization and an Adam or momentum-SGD optimizer, a
  cross-entropy objective, a learning-rate schedule with step or cosine annealing, trained for a
  fixed epoch budget.
- Protocol for assessing a regularizer: hold the architecture, base loss, optimizer, schedule,
  data augmentation, and evaluation fixed; add the candidate penalty (sweeping its coefficient
  over a small grid, e.g. across a few orders of magnitude); compare best test accuracy reached.

## Code framework

The regularizer plugs into an existing image-classification training loop. The data pipeline,
the convolutional model, the cross-entropy loss, the optimizer (with its own built-in L2 weight
decay), and the schedule all already exist and stay fixed. The one slot to be filled is a
function called every training step that, given the model and the current batch, returns a scalar
that is added to the cross-entropy loss before backpropagation. What that scalar should be — what
quantity computed from the weights (or activations, or outputs) deserves to be penalized — is
exactly what is to be designed.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_regularization(model, inputs, outputs, targets, config):
    """Return a scalar tensor added to the cross-entropy loss every training step.

    model:   the full nn.Module; iterate model.named_parameters() / named_modules().
    inputs:  [B, 3, H, W] batch.    outputs: [B, num_classes] logits.   targets: [B] labels.
    config:  dict with num_classes, epoch, total_epochs.
    Must backpropagate through PyTorch (differentiable or subdifferentiable). Standard L2
    weight decay is ALREADY applied by the optimizer; this term is additional.
    """
    # TODO: the quantity computed from the model that we will penalize.
    pass


# existing, fixed training loop the regularizer plugs into
def train_step(model, inputs, targets, optimizer, criterion, config):
    optimizer.zero_grad()
    outputs = model(inputs)                                   # forward through the fixed model
    loss = criterion(outputs, targets)                        # fixed cross-entropy objective
    loss = loss + compute_regularization(model, inputs, outputs, targets, config)
    loss.backward()                                           # backprop through loss + penalty
    optimizer.step()                                          # fixed optimizer (built-in weight decay)
    return loss
```

The outer loop supplies the model and batch; `compute_regularization` is where the additional
penalty term will live.
