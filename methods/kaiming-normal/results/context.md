# Context: initializing the weights of very deep rectifier CNNs (circa 2014)

## Research question

When you stack many convolutional layers and train them from scratch with stochastic
gradient descent, the random numbers you put in the weight tensors *before* the first
gradient step turn out to decide whether the network trains at all. The reason is purely
multiplicative: a signal passing through `L` layers is transformed by a product of `L`
per-layer factors, and a forward or backward signal scaled by a constant `β` in each layer
ends up scaled by `β^L` after `L` layers. If `β > 1` the responses (or the gradients)
amplify exponentially and the run diverges to infinity; if `β < 1` they shrink exponentially
and learning stalls because nothing reaches the early layers. For a handful of layers either
failure is mild, but as `L` grows past roughly eight convolutional layers the exponent makes
the difference between a network that learns and one that does nothing.

The concrete pain point of the time is exactly this. The dominant recipe is to fill every
weight with i.i.d. Gaussian noise of a single fixed standard deviation — `0.01` is the value
that traveled with the AlexNet recipe (Krizhevsky, Sutskever & Hinton 2012). With that fixed
std, networks deeper than about eight convolutional layers are reported not to converge: the
VGG team (Simonyan & Zisserman 2014) say so explicitly, and the same is seen elsewhere. The
workarounds in circulation are indirect — pre-train a shallow (8-conv) model and use it to
initialize a deeper one (Simonyan & Zisserman 2014), or bolt auxiliary classifiers onto
intermediate layers to inject gradient closer to the input (Szegedy et al. 2014; Lee et al.
2014). Both add training cost and complexity, and pre-training a shallow model first may land
in a poorer optimum. The precise goal is a *data-independent, one-shot* rule for the standard
deviation of each layer's initial Gaussian weights — a rule that depends only on the layer's
shape — such that neither the forward responses nor the backward gradients shrink or grow
exponentially with depth, so that a very deep rectifier network can be trained directly from
scratch.

## Background

The relevant unit is the rectifier. The Rectified Linear Unit `f(y) = max(0, y)` (Nair &
Hinton 2010; Glorot, Bordes & Bengio 2011; used at scale by Krizhevsky et al. 2012) has
become the default nonlinearity for deep CNNs because it speeds convergence and reaches
better solutions than sigmoid-like units. A close relative keeps a small slope on the
negative side, the Leaky ReLU `f(y) = max(0, y) + a·min(0, y)` with a small fixed `a` (Maas,
Hannun & Ng 2013), introduced to avoid exactly-zero gradients. The single property of the
rectifier that matters for initialization is that it is *not* odd and *not* linear: it deletes
the negative half of its input. That asymmetry leaves a scale effect that the existing linear
rules do not model.

The load-bearing tool is variance analysis of signal propagation, due to Glorot & Bengio
(2010). The idea is to track how the variance of a layer's responses (forward) and of the
back-propagated gradients (backward) changes from layer to layer, and to pick the weight
variance so neither changes. Write a dense layer as `s = z·W + b`, `z = f(s)`. Treating the
weights as i.i.d., the inputs as i.i.d. and independent of the weights, and — this is the key
assumption of that analysis — treating the network as operating in a *linear regime* where
the activation derivative `f'(s) ≈ 1`, the forward variance recursion across a layer of
fan-in `n_in` is `Var[z_out] = n_in·Var[W]·Var[z_in]`, and the backward (gradient) variance
recursion across a layer of fan-out `n_out` is `Var[∂C/∂s_in] = n_out·Var[W]·Var[∂C/∂s_out]`.
Demanding that the forward variance be preserved gives the condition `n_in·Var[W] = 1`;
demanding that the backward variance be preserved gives `n_out·Var[W] = 1`. The two cannot
hold simultaneously unless every layer has the same width, so as a compromise one takes
`Var[W] = 2/(n_in + n_out)` — the "normalized initialization," implemented as a uniform draw
`W ~ U[−√(6/(n_in+n_out)), +√(6/(n_in+n_out))]`. This is the recipe known as "Xavier"
initialization (Jia et al. 2014). For comparison, the older fixed heuristic `W ~ U[−1/√n,
1/√n]` corresponds to `n·Var[W] = 1/3`, which makes the back-propagated gradient variance
depend on the layer and decrease with depth.

A second theoretical guideline came from the exact learning-dynamics analysis of deep *linear*
networks by Saxe, McClelland & Ganguli (2013). They show that initializing each weight matrix
to a random *orthogonal* matrix gives "dynamical isometry": the input–output Jacobian acts
like an isometry, all of its singular values near one, so signals and gradients propagate
through arbitrarily many layers without changing norm, yielding depth-independent learning
times. Saxe et al. note that in a nonlinear network the orthogonal weights must additionally
be scaled by a gain tuned so that the linear amplification of the weight matrix balances the
dampening caused by the nonlinearity — i.e. even the orthogonal route needs a scalar gain that
depends on the activation.

Two diagnostic facts about existing systems frame the problem sharply. First, the magnitude of
the per-layer std really does control deep convergence: a fixed `0.01` may be tolerable in a
shallow model but can leave early-layer gradients extremely small once many convolutional
layers are stacked. Second, the existing variance analyses were derived under the
linear-regime assumption `f'(s) ≈ 1`, which is exactly the assumption a rectifier breaks; the
gap between the assumed linear unit and the actual rectifier is itself a pre-method fact about
why the off-the-shelf recipe is not quite right for these networks.

## Baselines

These are the initialization recipes a new rule would be measured against and would react to.

**Fixed-std Gaussian (Krizhevsky, Sutskever & Hinton 2012).** Draw every weight from
`N(0, σ²)` with a single hand-chosen `σ` (commonly `σ = 0.01`) for all layers, biases set to a
small constant or zero. Simple and width-agnostic. **Limitation:** because `σ` ignores each
layer's fan, the per-layer variance factor `n·σ²` is not one, so over many layers the forward
responses and backward gradients drift exponentially; deep stacks (more than ~8 conv layers)
are observed not to converge.

**Xavier / normalized initialization (Glorot & Bengio 2010).** Choose the weight variance from
the layer's shape so that, *in the linear regime*, forward responses and backward gradients
keep their variance: forward wants `n_in·Var[W] = 1`, backward wants `n_out·Var[W] = 1`, and
the compromise `Var[W] = 2/(n_in + n_out)` is realized as `W ~ U[−√(6/(n_in+n_out)),
+√(6/(n_in+n_out))]` (a Gaussian with the same variance works identically). This makes
convergence far less sensitive to depth than a fixed std. **Limitation:** the derivation
assumes the activation is in a linear regime, `f'(s) ≈ 1` and the unit symmetric around zero.
That assumption is false for a rectifier, whose output is one-sided and whose derivative is
gated. The recipe therefore has no activation-specific term for that one-sided, gated
behavior; if the resulting scale mismatch compounds across many layers, the forward and
backward scales can drift even though the rule uses the fan.

**Random orthogonal initialization (Saxe, McClelland & Ganguli 2013).** Initialize each weight
matrix as a random (semi-)orthogonal matrix — draw a Gaussian matrix and take the `Q` of its
QR factorization — optionally times a scalar gain. In a deep linear net this gives exact
norm-preservation of both signals and gradients (dynamical isometry), independent of depth.
**Limitation:** the isometry result is derived for linear networks, and the analysis itself
only norm-preserves once a gain is chosen to offset the nonlinearity's dampening — but it does
not say *what* that gain should be for a rectifier. Operationally it also needs a matrix
factorization rather than a single i.i.d. draw, and for a convolution weight tensor
`[d, c, k, k]` the four-dimensional tensor must first be flattened to a matrix.

## Evaluation settings

The natural yardsticks in use at the time, all pre-existing:

- **ImageNet 2012 classification** (Russakovsky et al. 2014): ~1.2M training images, 1000
  classes; metric is top-1 / top-5 error on the validation set; the official ranking metric is
  top-5 error. Standard training is mini-batch SGD with momentum 0.9, weight decay `5e-4`, a
  step-decayed learning rate, random `224×224` crops with per-pixel mean subtraction and
  horizontal flips.
- **Very deep CNN architectures** as the stress test for initialization: VGG-style stacks of
  `3×3` convolutions (16–19 weight layers), and deliberately deeper rectifier stacks (up to
  ~30 conv+fc layers) built by adding small `2×2` conv layers, used to probe whether a network
  trains *from scratch* at all.
- The diagnostic readout for an initialization is the *training/validation error versus
  epoch* curve at the start of training — specifically whether error begins to fall at all,
  how early it starts falling, and whether the back-propagated gradients at the early layers
  are diminishing (which can be checked by whether the gradient is modulated only by weight
  decay).

## Code framework

An initialization rule is a function handed the fully constructed model; it walks the modules
and writes initial values into each parameter in place, with no access to data and no
calibration forward passes. The model, the convolution / batch-norm / linear module types,
the optimizer, the loss, and the training loop all already exist; the only thing not yet
decided is *what numbers to write into each weight tensor as a function of that tensor's
shape*. So the substrate is the generic "iterate the modules and fill the parameters" harness,
with one empty slot per module type for the rule we are about to design.

```python
import torch.nn as nn


def initialize_weights(model, config):
    """Walk every module and set its initial parameters in place.

    Data-independent: depends only on each parameter tensor's shape (fan), never on
    training data, and runs no calibration forward passes. `config` carries `arch`,
    `num_classes`, `depth` (number of Conv2d + Linear layers).
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # TODO: choose this conv's weight distribution from its shape;
            #       set its bias.
            pass
        elif isinstance(m, nn.BatchNorm2d):
            # TODO: choose the affine parameters of normalization at init.
            pass
        elif isinstance(m, nn.Linear):
            # TODO: choose this linear layer's weight distribution from its shape;
            #       set its bias.
            pass


# existing training harness the initialized model is dropped into
def train(model, loss_fn, data_loader, optimizer, scheduler, epochs):
    initialize_weights(model, config)              # one-shot, before any data is seen
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)                 # forward through the existing model
            loss = loss_fn(outputs, targets)        # existing loss
            loss.backward()                         # backprop fills each p.grad
            optimizer.step()
        scheduler.step()
```

The shape information each slot can use is fixed by the tensor layout: a conv weight is
`[out_channels, in_channels, k, k]`, a linear weight is `[out_features, in_features]`. From
those one can read off the number of incoming connections per response, `in_channels·k²` (the
"fan-in"), and the number of outgoing connections, `out_channels·k²` (the "fan-out"). The
empty slots are exactly where the per-layer rule will go.
