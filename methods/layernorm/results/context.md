# Context

## Research question

State-of-the-art deep networks trained with stochastic gradient descent take days to converge, and the dominant lever for speeding this up — data/model parallelism across machines — has rapidly diminishing returns and heavy communication cost. An orthogonal lever is to change the computation in the forward pass so that optimization itself becomes easier: if the distribution of the signals flowing into each layer can be kept stable as the parameters below shift during training, gradients become better-conditioned and larger learning rates become usable.

A mini-batch-based normalization scheme already delivers exactly this speedup for feed-forward networks, but it carries three coupled liabilities. First, its normalization statistics are estimated from the current mini-batch, so their quality degrades as the batch shrinks, and the pure online regime (batch size 1) is impossible. Second, it does not transfer cleanly to recurrent networks: the same weights are reused at every time-step, yet the distribution of summed inputs differs per step and sequences have variable length, so the obvious application needs separate stored statistics per time-step and breaks when a test sequence is longer than any training sequence. Third, it computes one thing at training time (batch statistics) and a different thing at test time (stored running averages), requiring extra bookkeeping and creating a train/test mismatch.

The question: is there a normalization that delivers the same conditioning benefit — stable summed-input distributions, faster convergence — while being independent of batch size, identical at train and test time, and naturally applicable to recurrent networks of arbitrary length?

## Background

The relevant unit of analysis is the *summed input* (pre-activation) to a neuron. In layer l, neuron i computes a_i = w_i^T h, and the hidden output is f(a_i + b_i) for an element-wise nonlinearity f. A recognized difficulty of deep learning is that the gradient with respect to the weights in one layer depends strongly on the outputs of the previous layer, and those outputs can shift in a highly correlated way as training proceeds — especially with ReLU units, whose outputs can change by large amounts. This correlated drift of the input distribution to a layer ("covariate shift") forces small learning rates and careful initialization.

The standard mini-batch normalization attacks this by standardizing each summed input over the data. For the i-th summed input in layer l:

  abar_i = (g_i / sigma_i)(a_i - mu_i),  mu_i = E_{x~P(x)}[a_i],  sigma_i = sqrt(E_{x~P(x)}[(a_i - mu_i)^2]),

with a learned gain g_i scaling the normalized value before the nonlinearity. The expectation over the full data distribution is impractical (it would require a forward pass over the whole dataset at the current weights), so mu and sigma are estimated from the current mini-batch — which is exactly what ties the method to batch size and to the per-time-step problem in recurrent nets. Feed-forward networks trained this way converge far faster even with plain SGD, and the sampling noise in the batch statistics additionally acts as a regularizer.

Two empirical/diagnostic facts about the existing schemes set up the problem. Extending mini-batch normalization to recurrent networks was found to work best only when independent normalization statistics are kept for each time-step, and only when the gain parameter inside the recurrent normalization is initialized to a small value (0.1); a default-scale initialization fails. Both observations point to fragility that comes from forcing a batch-statistics scheme onto a setting it was not designed for.

A second relevant strand reparameterizes the weights instead of normalizing activations: write each weight vector as w = g · v/||v||₂, decoupling its length g from its direction v. This is equivalent to normalizing the summed input with mu = 0 and sigma = ||w||₂, and it improves conditioning while being cheap and batch-independent.

A geometric lens underlies the analysis of all these schemes. For a model whose output is a distribution P(y|x;theta), the natural way to measure how far apart two parameter settings are is the KL divergence between their output distributions, which makes parameter space a Riemannian manifold. To second order, KL[P(y|x;theta) || P(y|x;theta+delta)] ≈ (1/2) delta^T F(theta) delta, where F is the Fisher information matrix F(theta) = E[(∂ log P/∂theta)(∂ log P/∂theta)^T]. The Fisher matrix is the local metric: it says how much a parameter step actually moves the function. Reading off this metric for a normalized model is what reveals whether a normalization changes *learning dynamics*, not just the *function* — two parameterizations can express the same function yet train very differently.

## Baselines

**Mini-batch normalization (Ioffe & Szegedy, 2015).** Normalizes each summed input per-neuron using mean and variance computed across the mini-batch, then applies a learned per-neuron gain and bias before the nonlinearity. Core effect: keeps each neuron's pre-activation distribution stable across updates, enabling large learning rates and fast convergence with plain SGD; batch noise regularizes. Gap it leaves: the statistic is computed *across the batch*, so it is batch-size dependent (noisy or undefined for very small / size-1 batches), it requires running averages so that test-time computation differs from training, and it has no clean recurrent form because each time-step has its own summed-input distribution and sequences vary in length.

**Recurrent mini-batch normalization (Laurent et al. 2015; Amodei et al. 2015; Cooijmans et al. 2016).** Attempts to carry batch normalization into recurrent networks. The most effective variant keeps independent statistics for every time-step and initializes the recurrent gain to 0.1. Core idea: per-time-step batch statistics inside the recurrent loop. Gap: storing per-time-step statistics fails for sequences longer than those seen in training, and the strong sensitivity to gain initialization signals an unstable interaction between batch statistics and the recurrent nonlinearity.

**Weight normalization (Salimans & Kingma, 2016).** Reparameterizes each weight vector as w = g · v/||v||₂, separating length from direction; equivalent to normalizing the summed input with mu = 0 and sigma = ||w||₂. Core idea: improve conditioning by a cheap, batch-independent reparameterization. Gap: with mu = 0 it does not re-center, and being a pure reparameterization of the original network it inherits that network's invariances and nothing more — in particular it is not invariant to re-scaling or re-centering of the input data, nor to per-example rescaling.

**Path-normalized SGD (Neyshabur et al. 2015).** Studies reparameterization invariance in ReLU networks and optimizes a path-based norm. Relevant as part of the reparameterization-invariance line that frames how to think about which transformations of the weights leave a network's function unchanged.

A unifying observation across these: batch normalization (using expected statistics) and weight normalization are both *reparameterizations* of the original feed-forward network — they leave the represented function reachable by the original net and only change its coordinates. A different normalization that is not merely a coordinate change would need its own invariance and geometry checks, because equal function classes do not imply equal learning dynamics.

## Evaluation settings

The natural yardsticks are recurrent-network tasks where batch normalization is awkward, plus a feed-forward control where it is strong.

- **Image–sentence ranking:** order-embeddings on Microsoft COCO; a GRU encodes sentences, a pre-trained VGG ConvNet (10-crop) encodes images, into a shared space; metric Recall@K (R@1/5/10) and mean rank, evaluated over 5 test splits. Optimizer Adam; Theano implementation modifying the public order-embedding code.
- **Question answering / reading comprehension:** unidirectional attentive reader on the CNN corpus (passages truncated to 4 sentences, entities anonymized with consistently permuted tokens); validation error over training; Theano. This setting is the direct head-to-head against recurrent batch normalization.
- **Contextual sentence representations:** skip-thought vectors trained on BookCorpus, a 2400-dim encoder; downstream evaluation on semantic relatedness (SICK; Pearson, Spearman, MSE) and sentence classification (MR, CR, SUBJ, MPQA accuracy), checkpointed periodically; Adam; Theano.
- **Generative modeling:** DRAW on fixed-binarization MNIST (50k/10k/10k split), 64 glimpses, 256 LSTM units, minibatch 128, Adam; metric test variational bound / negative log-likelihood in nats.
- **Handwriting sequence generation:** the IAM-OnDB online handwriting database (handwritten lines from 221 writers; average line length ~700, input strings >25 characters); a 3-layer, 400-cell LSTM emitting 20 bivariate Gaussian mixture components, with a soft-window attention over one-hot character input; minibatch size 8, Adam; metric negative log-likelihood. Chosen for long sequences and small batch.
- **Permutation-invariant MNIST:** a 784-1000-1000-10 fully-connected classifier, Adam, batch sizes 128 and 4; metric negative log-likelihood and test error. The feed-forward control where mini-batch normalization is the incumbent; the small-batch run probes batch-size sensitivity.
- **Convolutional networks:** standard image-classification ConvNets, as a setting where the per-layer-statistics assumption is stressed.

## Code framework

The primitives that already exist: a tensor library with autodiff (Theano-era / PyTorch-style), an Adam optimizer, standard losses, and base recurrent cells (LSTM, GRU). A small transformation slot before a nonlinearity is the new primitive to design; once it exists, it can be dropped into recurrent cells and feed-forward layers.

```python
from typing import List, Union

import torch
from torch import Size, nn


class Normalizer(nn.Module):
    """Placeholder transform inserted before a nonlinearity."""
    def __init__(self, feature_shape: Union[int, List[int], Size], *,
                 eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        # TODO: decide what statistics and parameters this transform needs.
        pass

    def forward(self, x: torch.Tensor):
        # TODO: transform x before the layer nonlinearity.
        pass


class RecurrentCell(nn.Module):
    """A standard gated recurrent step built from existing linear projections."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.W_x = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.b = nn.Parameter(torch.zeros(4 * hidden_size))
        # TODO: normalization sub-modules, if this step is to be normalized.
        pass

    def forward(self, x_t, state):
        h_prev, c_prev = state
        # TODO: combine recurrent and input contributions, compute gates, update
        #       cell state, and return the new hidden state and recurrent state.
        pass


def train_step(model, batch, optimizer, loss_fn):
    optimizer.zero_grad()
    out = model(batch.inputs)
    loss = loss_fn(out, batch.targets)
    loss.backward()
    optimizer.step()
    return loss.item()
```
