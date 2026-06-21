## Research question

The setting is supervised learning of a mapping from a high-dimensional, *structured* input to
a target, where the structure is the whole point: the input coordinates are not an unordered bag
of numbers but lie on an axis (or grid) along which nearby coordinates are strongly correlated,
and the features that matter for the prediction are *local* (they involve a handful of
neighboring coordinates) yet can appear at *different positions* from one example to the next. A
handwritten digit is the canonical case — a stroke endpoint or a corner is a local feature, but
where it lands in the image shifts with the writer. The same shape recurs in any signal with a
real ordered axis: a temporal waveform, or a physical profile sampled along a spatial
coordinate, where adjacent samples are coupled by the underlying process and a salient local
motif can occur at different positions along the axis.

The goal is to learn this mapping end-to-end by gradient descent on a differentiable loss, from a
finite training set, given an input whose coordinates carry a known local structure and ordering.

## Background

The field state is "gradient-based learning works, if you give it the right architecture."

**Gradient-based learning and back-propagation.** The learning machine computes a function
`Y = F(Z, W)` with adjustable parameters `W`; a differentiable loss `E(W)` measures the
discrepancy between `Y` and the desired output, averaged over a labeled training set. Minimizing
`E` by gradient descent `W ← W − ε·∂E/∂W` reduces to computing the gradient efficiently, which
back-propagation (Rumelhart, Hinton & Williams 1986) does for any cascade of differentiable
modules `X_n = F_n(W_n, X_{n-1})` via the backward recurrence

```
∂E/∂W_n     = (∂F/∂W)(W_n, X_{n-1}) · ∂E/∂X_n
∂E/∂X_{n-1} = (∂F/∂X)(W_n, X_{n-1}) · ∂E/∂X_n.
```

The stochastic ("on-line") version, updating `W` from one example's gradient at a time, `W_k =
W_{k-1} − ε·∂E^{p_k}/∂W`, converges considerably faster than full-batch descent on large,
redundant datasets — the regime that matters here.

**Capacity and generalization.** The gap between test and training error is governed, to a
useful approximation, by

```
E_test − E_train ≈ k · (h / P)^α,    0.5 ≲ α ≲ 1,
```

with `h` the effective capacity of the machine (roughly its VC dimension, which grows with the
number of free parameters) and `P` the number of training samples. For a fixed dataset size
`P`, fewer free parameters means a smaller `h` and a smaller generalization gap.

**The biological and computational template for local-then-pool processing.** Hubel & Wiesel's
(1962) study of the cat's visual cortex described two cell classes arranged in a feedforward
hierarchy: *simple cells*, with small, oriented, localized receptive fields that act like local
oriented-edge detectors, and *complex cells*, which respond to the same oriented feature over a
*range of positions* — i.e. they tolerate where the feature is. The motif is two stages: detect a
local feature, then become insensitive to its exact location. Fukushima's neocognitron (1980)
turned this into a multilayer artificial network: alternating layers of feature-extracting units
whose weights are *replicated across position* (the same local detector applied everywhere) and
pooling units that grant tolerance to small shifts, cascaded so that local features are integrated
into higher-order features and the whole network recognizes patterns regardless of position. The
neocognitron is trained by *unsupervised self-organization*, with no globally supervised
objective.

**Tying parameters together.** Connections in a network need not each carry their own weight: a
group of connections can be constrained to share a single parameter (Rumelhart et al. 1986, in
the "T-versus-C" shape-discrimination problem). Such an equality constraint among connection
strengths is cheap to enforce and, by construction, reduces the count of independent parameters.

**The unstructured baseline.** When a fully-connected network is trained directly on raw,
size-normalized character images, it fits the training set while leaving a larger held-out error.
The capacity relation ties held-out error to `h`, which grows with the number of independent
parameters.

## Baselines

**Fully-connected multilayer network on raw inputs (the standard classifier).** Alternate
layers of matrix multiplication and a component-wise squashing nonlinearity; train end-to-end by
back-propagation; output through a softmax or RBF layer. It is a universal approximator and the
default supervised learner. A first fully-connected layer with, say, a hundred hidden units on an
input of a few hundred coordinates carries tens of thousands of independent weights; on a 2D image
the count is far larger. The architecture is invariant to a fixed permutation of the input
coordinates — it treats them as an unordered vector, with no built-in notion of which coordinates
are neighbors and no built-in invariance to translations of the input.

**Hand-designed local feature extractor followed by a trainable classifier (traditional pattern
recognition).** A fixed front end computes local features engineered by hand (oriented-gradient
estimators, on-center/off-surround detectors, and the like), and only the back-end classifier is
trained. This respects locality and can build in some invariance. The front end is frozen and
human-designed: its features are fixed in advance rather than optimized against the prediction
error.

**Self-organized hierarchical local-feature network (neocognitron line).** A cascade of
position-replicated local-feature layers and pooling layers, achieving shift-tolerant recognition
(above). Its detectors are set by unsupervised self-organization rather than driven by a
supervised, end-to-end objective.

## Evaluation settings

The natural yardsticks already in use for this kind of problem:

- **Isolated handwritten-digit recognition** on the modified NIST set (MNIST): 28×28 (or
  size-normalized 32×32) grayscale digit images, ten classes, with a standard train/test split;
  metric is test misclassification rate (%). Inputs are normalized so the background and
  foreground sit at fixed values, roughly zero-mean and unit-variance, to keep them in the
  network's operating range. The protocol of choice is on-line (stochastic) gradient descent
  over many passes through the training set.
- **A diagnostic comparison against the unstructured baseline**: train fully-connected models and
  structured candidates under matched data, loss, and comparable parameter or connection budgets,
  then compare training and test error to separate memorization capacity from useful inductive
  bias.
- For structured signals with a single ordered axis, the corresponding yardsticks are sequence-
  or profile-regression / classification tasks: an input organized as a number of variables
  measured along an ordered coordinate, predicting a per-position and/or aggregate target;
  metric a regression error (e.g. normalized mean-squared error) or classification rate. The
  loss is the mean squared error or a maximum-likelihood criterion, and the optimizer is
  mini-batch stochastic gradient descent — all fixed before any architecture is chosen.

## Code framework

A candidate model plugs into the fixed supervised-learning harness: normalized mini-batches of
flat column inputs and flat targets, a differentiable model, mean-squared-error training loss,
AdamW with a scheduled learning rate, gradient clipping, validation-based early stopping, and
held-out metrics. The editable surface is a single model class. The task interface is already
settled: the first `9 * 60 = 540` input coordinates are multi-level profiles, the remaining
coordinates are whole-column scalars, the first `6 * 60 = 360` outputs are multi-level
tendencies, and the remaining `8` outputs are scalar diagnostics. What is not settled is the
internal architecture of that model: given an ordered vertical axis, locally correlated profiles,
and scalar column conditions, how should the layers be wired so the model uses the structure
instead of treating the input as an unordered vector?

```python
import torch
import torch.nn as nn


N_LEVELS = 60
N_PROFILE_IN = 9
N_PROFILE_OUT = 6


class Custom(nn.Module):
    """Architecture slot for a column emulator.

    Input `x` has shape (batch_size, input_dim). Its first 9*60 entries are
    profile variables laid out on the ordered vertical axis; the rest are scalar
    column variables. The output must have shape (batch_size, output_dim), with
    the first 6*60 entries reserved for profile tendencies and the rest for
    scalar diagnostics.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_levels = N_LEVELS
        self.n_profile_in = N_PROFILE_IN
        self.n_profile_out = N_PROFILE_OUT
        self.n_scalar_in = input_dim - self.n_profile_in * self.n_levels
        self.n_scalar_out = output_dim - self.n_profile_out * self.n_levels
        # TODO: the architecture we will design.

    def forward(self, x):
        # x: (batch_size, input_dim)
        # TODO: produce a (batch_size, output_dim) prediction.
        raise NotImplementedError


# fixed mini-batch stochastic-gradient training loop the model plugs into
def train(model, loss_fn, data_loader, optimizer):
    for inputs, targets in data_loader:          # one structured input per example
        optimizer.zero_grad()
        outputs = model(inputs)                  # forward through the model
        loss = loss_fn(outputs, targets)         # MSE / NMSE regression loss
        loss.backward()                          # back-prop through the model
        optimizer.step()                         # AdamW / SGD-family step
```

The data pipeline, loss, optimizer, metrics, and training loop are fixed. The single open piece
is `Custom`: the architecture that fills the empty slot while preserving this exact input/output
contract.
