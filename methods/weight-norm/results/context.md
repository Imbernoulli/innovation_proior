# Context: conditioning and parameterization of first-order training for deep networks

## Research question

Deep neural networks are trained, overwhelmingly, by first-order stochastic gradient
methods. The practical success of those methods is highly sensitive to the *curvature* of
the loss surface: when the Hessian at a solution is ill-conditioned (a large ratio between
its largest and smallest eigenvalues), gradient descent zig-zags across the steep
directions and crawls along the flat ones, and progress per step is poor. A crucial and
under-exploited fact is that this curvature is **not invariant to how the model is
parameterized**. The very same input-output function can be written with different internal
parameters, and some of those parameterizations give a far better-conditioned optimization
problem than others. Two parameterizations that compute identical functions can therefore
differ enormously in how fast SGD converges on them.

The precise problem: take the elementary building block that almost every architecture is
made of — a neuron that computes a weighted sum of its inputs plus a bias, followed by a
pointwise nonlinearity, y = φ(w·x + b) — and find a *reparameterization* of its weights
that improves the conditioning of the gradient and so speeds up first-order optimization.
Because this block is shared across feed-forward nets, convolutional nets, recurrent nets,
generative models and value/policy networks, a change at this level would help a very wide
range of models at once. The solution must be cheap (little extra computation or memory),
must not introduce dependencies between the examples in a minibatch, and must not inject
stochastic noise into the gradient, so that it remains usable in recurrent models and in
noise-sensitive settings such as reinforcement learning and generative modelling.

## Background

**Curvature, conditioning, and reparameterization.** First-order methods make fast progress
only when the loss is well-conditioned; pathological curvature stalls them
(Martens 2010; Sutskever et al. 2013 on the importance of initialization and momentum).
Crucially, the amount of curvature seen by the optimizer depends on the coordinates in
which the parameters are expressed (Amari 1997): there are many equivalent ways to
parameterize the same model, some far easier to optimize than others. This makes "find a
better parameterization" a legitimate and powerful lever, distinct from changing the
optimizer.

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
distributions. Those assumptions hold at initialization but drift as training moves the
weights, so initialization alone cannot keep activations well-behaved throughout training.
A complementary idea is *data-dependent* initialization: run one minibatch through the
network and set the per-layer scales and offsets from the measured pre-activation
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
learning rates. Its limitations, and the gap it leaves open: (1) it makes the output for
one example depend on the other examples in the minibatch — the examples are coupled;
(2) μ[t] and σ[t] are stochastic estimates, so the layer injects noise into the gradients,
and that noise has high variance when the minibatch is small; (3) train and test compute
different functions — at test time, when there is no representative minibatch, frozen
running averages are substituted for μ[t], σ[t]; (4) it is awkward in recurrent networks,
where the same weights are reused at every timestep and normalizing the recurrent cell
states diminishes their ability to carry information across time; and (5) it carries real
overhead in time and memory. These together make it ill-suited to recurrent models and to
noise-sensitive applications such as deep reinforcement learning and generative models.

**Approximate-natural-gradient preconditioners (KFAC, FANG, PRONG).** Core idea: form a
tractable approximation to the (inverse) Fisher and use it to precondition the gradient,
approximating the natural gradient. Gap: they require estimating and inverting curvature,
which adds substantial computation, memory, and implementation complexity, and ties the
method to a particular optimizer rather than improving the model itself.

**Output-reparameterization (Raiko et al. 2012).** Core idea: transform neuron outputs to
zero average value and zero average slope, approximately diagonalizing the Fisher. Gap: it
operates on the outputs and centers around average behaviour; it does not directly
decouple the magnitude of a weight vector from its direction, which (as becomes clear) is
the axis along which the conditioning is worst.

**Norm-constrained weights (max-norm; Srebro & Shraibman 2005).** Core idea: keep the norm
of each weight vector controlled. Gap: the optimization is still carried out in the
original weight coordinates w, with the norm constraint merely *applied after* each SGD
step (a projection). It never changes what the optimizer sees — the gradient is still the
plain gradient in w — so it does not improve conditioning the way a genuine
reparameterization would.

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

The primitives that already exist, and the empty slots a new method would fill. A layer
stores its own raw weight tensor and bias and exposes a forward; an optimizer (SGD with
momentum, or Adam/Adamax) steps the registered parameters; a loss and a training loop
already work. The two open slots: a hook that may *recompute* a layer's effective weight
from some other set of trainable parameters before each forward, and a one-shot routine
that may *set* parameters from the statistics of a single minibatch passed through the
network before training begins.

```python
import torch
import torch.nn as nn

# ---- existing primitives -------------------------------------------------

class Linear(nn.Module):
    """A standard affine layer: y = phi(W x + b). Owns its raw weight and bias."""
    def __init__(self, in_features, out_features, act=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_normal_(self.weight)
        self.act = act

    def effective_weight(self):
        # TODO: by default the effective weight IS the stored weight.
        #       This is the slot where a weight reparameterization will plug in.
        return self.weight

    def forward(self, x):
        y = x @ self.effective_weight().t() + self.bias
        return self.act(y) if self.act is not None else y


def reparameterize_weight(layer):
    # TODO: replace a layer's single weight Parameter by an equivalent set of
    #       trainable parameters, and make effective_weight() rebuild the weight
    #       from them on every forward. Empty for now.
    pass


def data_dependent_init(model, x_init):
    # TODO: run one minibatch x_init through the model and set per-layer
    #       scale/offset parameters from the measured pre-activation statistics
    #       so pre-activations start standardized. Empty for now.
    pass


def make_optimizer(params, lr):
    return torch.optim.Adam(params, lr=lr)


def train(model, data, lr, epochs):
    opt = make_optimizer(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in data:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
```
