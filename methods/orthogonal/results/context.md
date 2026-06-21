## Research question

A deep feedforward network repeatedly multiplies activations on the forward pass
and error signals on the backward pass. If each layer scales a signal by even a
slightly wrong factor, depth raises that factor to a large power, so early layers
receive either vanishingly small gradients or unstable amplified ones. The
practical symptom is familiar: deeper networks can sit on long plateaus, then
move suddenly, and the number of gradient steps needed for useful learning can
grow with depth.

The goal is a data-independent initialization rule for a freshly constructed
network. It may use only the parameter shapes and the existing module types. It
must not use a calibration batch, pretrain on the data, change the architecture,
change the optimizer, or alter the loss.

## Background

The standard variance-propagation analysis treats a layer with fan-in `n` and
i.i.d. weights of variance `Var[W]`. Near initialization, a forward signal picks
up a factor about `n Var[W]` per layer, and the back-propagated gradient picks up
the corresponding fan-out factor. Across depth `d`, a gradient variance contains
a product like `(n Var[W])^{d-i}`. Scaled rules such as the Glorot and Bengio
compromise `Var[W] = 2/(n_in + n_out)` try to keep the per-layer average scale
close to one in both directions.

For an `N x N` Gaussian matrix with entries of variance `1/N`,
`E[v^T W^T W v] = v^T v`. Its squared singular values follow a Marchenko-Pastur
distribution. In a product of many independent matrices the full singular
spectrum, not only the mean squared norm, shapes which directions are propagated
through depth.

Deep linear networks provide a tractable place to study these dynamics. Although
their input-output map is a single linear transformation, gradient descent in the
layer weights is nonlinear because every layer's gradient contains the product of
all other layers. For whitened inputs, the input-output correlation matrix can be
decomposed by SVD, and learning can be studied mode by mode. The error surface of
a linear network has no spurious local minima: the non-global critical points are
saddles, and the stable solution is the best low-rank approximation of the
input-output correlation.

Greedy layer-wise unsupervised pretraining was the strongest practical answer to
deep optimization difficulty in this period. In a linear autoencoder, pretraining
amounts to learning principal directions of the input distribution before
supervised fine-tuning. Its advantage is optimization speed: it can place the
subsequent supervised problem closer to a well-aligned, high-strength starting
regime. It requires data and an extra training phase.

Standard linear-algebra and tensor primitives are available in the software
substrate: matrix factorizations for analysis, tensor views for reshaping
parameters, and in-place parameter fills. At the initialization hook, the only
permitted operation is to write parameter values in place before training starts.

## Baselines

**Small fixed-std Gaussian.** Draw every weight independently from a zero-mean
Gaussian with a small standard deviation such as `0.01`. It is simple and
data-free.

**Scaled Gaussian / Glorot-style variance matching.** Choose the variance so that
a typical signal has preserved squared norm in expectation; for a square
`N x N` layer, use entries with variance `1/N`, and for unequal fan-in/fan-out
use a compromise such as `2/(n_in + n_out)`.

**Greedy unsupervised pretraining.** Train layers as autoencoders, then
fine-tune on the supervised task. It can speed optimization by aligning the
network with important input directions before supervised training.

**Second-order or Hessian-free optimization.** Use a more powerful optimizer to
move through flat or ill-conditioned regions.

## Evaluation settings

The analytical testbed is a deep linear classifier trained by batch gradient
descent on whitened or controlled input statistics. The natural measurements are
the time for each input-output mode to reach a fixed fraction of its limiting
strength, the dependence of that time on the singular value of the data mode, and
the dependence of learning time and stable learning rate on depth.

The empirical linear-network setting uses MNIST-style inputs: a `784`-dimensional
input, a `10`-dimensional one-hot target, batch gradient descent, overcomplete
hidden layers, and depths ranging from shallow to about one hundred layers. The
input-output correlation can be precomputed, so very deep linear networks can be
trained by propagating correlation matrices rather than all examples. Learning
rate is tuned per depth over a logarithmic grid, and learning time is the first
iteration at which training error crosses a fixed threshold.

The nonlinear diagnostic setting uses deep feedforward networks with a saturating
nonlinearity and tracks the population variance
`q^l = (1/N) sum_i (x_i^l)^2` across layers, along with the singular-value
distribution of the end-to-end input-output Jacobian. The questions are whether
activity decays, grows without bound, or propagates at an `O(1)` scale, and
whether backpropagated gradients retain usable singular directions through many
layers.

The code-facing benchmark exposes a single hook, `initialize_weights(model,
config)`, inside a fixed training pipeline. The hook must initialize convolution,
linear, and normalization modules in place, without data access or calibration.
The fixed downstream pipeline includes SGD with momentum, cosine scheduling,
standard image augmentations, and architectures such as ResNet, VGG with batch
normalization, and MobileNet-style convolutional networks.

## Code framework

The initialization rule plugs into an existing PyTorch training harness. The
model is already constructed; the hook may iterate through modules and write
parameter tensors in place. Convolution kernels are ordinary tensors of shape
`out_channels x in_channels x kH x kW`, and linear weights are two-dimensional
`out_features x in_features` tensors. In the vision scaffold, convolution
modules are built without biases under batch normalization; linear biases and
normalization affine parameters can be filled with constants. The open slot is
the data-independent rule for the trainable weight tensors.

```python
import torch.nn as nn


def initialize_weights(model, config):
    """Set the parameters of a freshly built model in place.

    The rule may inspect module types and tensor shapes, but it must not use
    training data, run a calibration pass, or alter the model graph.
    """
    # TODO: choose and apply the data-independent rule for Conv2d and Linear
    # weights; set neutral constants for biases and normalization affine terms.
    pass
```
