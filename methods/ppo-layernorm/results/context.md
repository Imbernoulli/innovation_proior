# Context: normalizing neuron activations to speed up deep-net training (circa 2015-2016)

## Research question

Training state-of-the-art deep networks with stochastic gradient descent takes many days. Two of the
obvious ways to go faster — splitting the data across machines or splitting the model across machines —
buy speed at the cost of communication and software complexity, and they hit diminishing returns as the
degree of parallelism grows. An orthogonal lever is to change the *computation in the forward pass itself*
so that each gradient step makes more progress. The concrete observation that makes this lever attractive is
that the gradient with respect to the weights of one layer depends heavily on the outputs of the layer below,
and when those outputs shift together in a correlated way during training, every layer above is chasing a
moving target; this is sharply worse with ReLU units, whose outputs can swing by a lot. So the goal is a
cheap, in-the-forward-pass transformation, applied at each layer, that keeps the distribution of a layer's
summed inputs stable as training proceeds, lets the optimizer use a larger learning rate without diverging,
and thereby converges in fewer steps.

But the solution has to clear a set of constraints that the existing approach (below) does not. It must work
when the minibatch is small or even of size one (online learning, very large distributed models where each
worker holds only a few cases), it must perform the *same* computation at training and at test time so that
there is no train/test mismatch to reconcile, and — this is the case that breaks the existing approach
outright — it must apply cleanly to recurrent networks, where the same weights are reused at every time step
and different training sequences have different lengths. A method that needs a separate set of normalization
statistics for each position in the sequence cannot handle a test sequence longer than any training sequence,
and that is a routine occurrence in language and handwriting data. Closing the gap between "normalization
that accelerates feed-forward training" and "normalization that also satisfies all of the above" is the
problem.

## Background

By this time stochastic gradient descent (usually with momentum) is the engine behind the rapid progress in
vision (Krizhevsky et al. 2012) and speech (Hinton et al. 2012). A deep feed-forward network is a chain of
layers, each computing a summed input through a linear projection and then an element-wise nonlinearity. For
the `l`-th layer, with bottom-up input `h^l`, incoming weights `w_i^l` for unit `i`, and scalar bias
`b_i^l`,

```
a_i^l = (w_i^l)^T h^l ,        h_i^{l+1} = f( a_i^l + b_i^l ) ,
```

where `f` is an element-wise nonlinearity (sigmoid, tanh, ReLU). The parameters are learned by
backpropagation. The recurring pain is that the per-layer gradient is entangled with the activations of the
layer below: as the lower layers change, the *distribution* of each `a_i^l` drifts, and the upper layers
spend capacity re-adapting to that drift instead of learning. The standard partial fixes — ReLU units,
careful initialization, and small learning rates — attack the symptoms but cap the learning rate, which is
exactly what limits training speed.

A second, load-bearing background fact concerns the geometry of learning. For a model whose output is a
probability distribution, the natural way to measure how far an update moves the *function* (not the
parameters) is the Kullback-Leibler divergence between the output distributions before and after the update.
To second order this gives a Riemannian metric on parameter space (Amari 1998),

```
ds^2 = KL[ P(y|x;theta) || P(y|x;theta+delta) ] ≈ (1/2) delta^T F(theta) delta ,
F(theta) = E[ (d log P(y|x;theta)/d theta) (d log P(y|x;theta)/d theta)^T ] ,
```

the Fisher information matrix. The point of this frame, here, is diagnostic: it lets one ask how a change to
the *parameterization* of a network — leaving the function it computes unchanged — alters the curvature the
optimizer sees, and therefore how stable and how fast learning is. Different parameterizations of the same
function can be wildly easier or harder to optimize (low vs high condition number of the Hessian/Fisher), and
the success of first-order optimization is not invariant to reparameterization.

A third background fact is the diagnostic phenomenon that motivates normalization at all: when one inserts a
normalization that fixes the mean and variance of a layer's summed inputs but lets a downstream bias absorb
the mean, the bias can grow without bound while the normalized output stays fixed, and the model has been
*observed to blow up* — i.e. a normalization that the gradient step ignores is unstable. The cure observed in
practice is to make the normalization a differentiable part of the model so the gradient accounts for it.

## Baselines

**Batch normalization (Ioffe & Szegedy, ICML 2015).** Normalize each summed input *per feature, across the
training cases*. For the `i`-th summed input in layer `l`,

```
abar_i^l = (g_i^l / sigma_i^l) ( a_i^l - mu_i^l ) ,
mu_i^l = E_{x~P(x)}[ a_i^l ] ,
sigma_i^l = sqrt( E_{x~P(x)}[ (a_i^l - mu_i^l)^2 ] ) ,
```

with a learned per-neuron gain `g_i` and bias `b_i` applied after normalization, before the nonlinearity. The
expectations over the data are impractical, so in practice `mu` and `sigma` are estimated from the *current
minibatch*; making the minibatch statistics part of the computation is what keeps the transform
differentiable so the gradient can flow through it. The learned affine `g_i x̂ + b_i` restores
representational power — setting `g_i = sigma_i`, `b_i = E[a_i]` recovers the original activations, so the
transform can represent the identity and never costs capacity. Two further properties matter. First, BN lets
you use a much higher learning rate: it can be shown that `BN(Wu) = BN((aW)u)`, and
`d BN((aW)u)/d(aW) = (1/a) · d BN(Wu)/dW`, so scaling up a weight leaves the layer Jacobian unchanged but
*shrinks* the gradient to that weight — large weights self-stabilize, which is why higher learning rates stop
diverging. Second, the per-minibatch sampling noise in `mu, sigma` acts as a regularizer. BN trains
feed-forward nets dramatically faster with plain SGD. **Gaps:** (1) the statistics are computed over the
minibatch, so their quality degrades as the batch shrinks, and the method cannot be used at batch size one or
in the pure online regime; (2) at test time there is no minibatch, so BN must switch to running averages of
the statistics collected during training — the train-time and test-time computations differ, and those
running averages must be stored per layer; (3) on a recurrent network the summed inputs to the recurrent
units depend on the time step, so applying BN in the obvious way requires separate statistics for each time
step, which leaves a test sequence longer than every training sequence with no statistics to use.

**Weight normalization (Salimans & Kingma, 2016).** Instead of normalizing activations, reparameterize each
weight vector by decoupling its length from its direction,

```
w = (g / ||v||) v ,
```

so the optimizer learns a magnitude `g` and a direction `v` separately. This improves the conditioning of the
optimization and, because it touches only the weights, introduces no dependence between training cases — so
unlike BN it applies to recurrent models, reinforcement learning, and small/noisy-batch settings, and it is
much cheaper. It can be read as normalizing the summed input by the L2 norm of the incoming weights: in the
common framing where a normalization divides `a_i` by a scalar `sigma_i` and subtracts a scalar `mu_i`,
weight normalization corresponds to `mu_i = 0` and `sigma_i = ||w_i||_2`. **Gap:** it never subtracts a mean
(`mu = 0`, no re-centering), and it normalizes by a quantity computed purely from the *weights*, not from the
actual distribution of activations the layer sees — so it does not stabilize the activation distribution the
way an activation-statistic method does, and it is blind to the data.

**Recurrent batch normalization (Cooijmans et al. 2016; Laurent et al. 2015).** Extends BN into RNNs by
keeping independent normalization statistics *for each time step*, and reports that the gain parameter of the
recurrent BN layer must be initialized small (e.g. `0.1`) to train well. **Gap:** the per-time-step
statistics inherit BN's sequence-length problem, and the strong sensitivity to the gain initialization is a
fragility — getting that one constant wrong significantly degrades the final model.

## Evaluation settings

The natural yardsticks, all of which exist before any new method:

- **Recurrent NLP / embedding tasks** where stable hidden dynamics over long sequences matter:
  image-sentence ranking with order-embeddings on MS-COCO (GRU sentence encoder, pre-trained VGG image
  features; metric Recall@K and mean rank); a question-answering / reading-comprehension "attentive reader"
  on the CNN corpus; unsupervised sentence representations (skip-thoughts) trained on BookCorpus and
  evaluated downstream on semantic-relatedness and sentiment/subjectivity classification. These are the
  settings where per-time-step statistics and small batches are the natural stress test.
- **Generative / long-sequence recurrent modeling:** binarized MNIST density modeling with a recurrent
  attention writer (test variational bound in nats); online handwriting prediction on the IAM-OnDB database,
  where the combination of very long sequences (~700 steps) and a small minibatch (size 8) makes stable
  hidden dynamics essential (negative log likelihood vs. updates).
- **Feed-forward classification:** permutation-invariant MNIST with a 784-1000-1000-10 fully-connected net,
  trained at batch size 128 and at the much smaller batch size 4, comparing convergence and test error — the
  setting that isolates robustness to batch size. Training uses the Adam optimizer.
- **Convolutional networks** as a stress case where the units in a layer do *not* all contribute similarly
  (boundary-receptive-field units fire rarely and carry different statistics).
- Protocol throughout: match the published baseline's architecture and hyperparameters, insert the candidate
  forward-pass transform in the hidden computation, and read per-iteration and wall-clock convergence as well
  as final generalization. Metrics are task-specific (Recall@K, accuracy, NLL / variational bound), measured
  against iterations or epochs.

## Code framework

The existing substrate is just a hidden-layer builder: affine map, optional same-shape transform, nonlinearity,
repeated, followed by a task-specific output head. The open design object is the transform applied to the
summed inputs of a hidden layer and the placement rule that says where it should be used. The scaffold leaves
that object empty without committing to a statistic, an axis, or an affine parameterization.

```python
import torch
import torch.nn as nn


class HiddenTransform(nn.Module):
    """Same-shape transform for hidden summed inputs.

    It must be usable at any batch size, including one, and its train-time and test-time
    computations should not silently diverge.
    """

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        # TODO: parameters, if the chosen transform needs them.

    def forward(self, x):                     # x: (..., num_features), post-affine hidden inputs
        # TODO: return a tensor of the same shape as x.
        raise NotImplementedError


def build_hidden_stack(input_dim, hidden_dims, output_dim, activation=nn.Tanh()):
    """Affine -> optional transform -> activation, repeated, then an uncommitted output head."""
    layers = []
    dims = [input_dim] + list(hidden_dims)
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(d_in, d_out))
        # TODO: optionally insert HiddenTransform(d_out) here.
        layers.append(activation)
    layers.append(nn.Linear(dims[-1], output_dim))   # output head
    return nn.Sequential(*layers)
```

The empty slot is `HiddenTransform.forward`, plus the rule for which hidden computations receive it and which
task-output computations should remain untouched.
