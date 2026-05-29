# Layer Normalization

## Problem

Training deep networks with SGD is slow, and the obvious speedup — distributing the batch or model across machines — gives diminishing returns. An orthogonal lever is to change the forward computation so each gradient step is worth more, by keeping the distribution of summed inputs (pre-activations) to each layer stable as the layers below shift during training. Batch normalization achieves this for feed-forward nets, but it estimates per-neuron mean and variance *across the mini-batch*. That ties it to batch size (noisy or undefined at batch size 1), forces a train/test mismatch (training uses batch statistics, test uses stored running averages), and has no clean recurrent form: each time-step has a different summed-input distribution, so the obvious extension stores separate statistics per step and breaks when a test sequence is longer than any training sequence.

## Key idea

Transpose the normalization axis. Instead of computing statistics per-neuron over the batch dimension, compute them per-example over the *feature* dimension — the mean and variance over all `H` units in a layer, for a single training case. Because a shift in the layer below moves all summed inputs in the next layer in a highly correlated way, that drift shows up largely as a change in the layer's own mean and spread; subtracting them damps the same "covariate shift" batch norm targets, without ever touching the batch.

This removes all three liabilities at once: the statistics use only the current example, so batch size is irrelevant (size 1 works), training and test do identical computation, and an RNN normalizes with statistics computed from the `H` summed inputs at the current step `t`, with one shared gain and bias across all steps regardless of sequence length.

## Final method

For a single training case, over the `H` units of a layer:

    mu    = (1/H) * sum_i a_i
    sigma = sqrt( (1/H) * sum_i (a_i - mu)^2 )
    h_i   = f( (g_i / sigma) * (a_i - mu) + b_i )

where `a_i = w_i^T h` is the summed input, `f` the nonlinearity, and `g_i`, `b_i` are a learned per-neuron gain and bias applied after standardization and before the nonlinearity (they give each neuron a learned scale and operating point after the shared normalization; init `g=1`, `b=0` so the affine restore starts as identity on the standardized signal).

For a recurrent layer with `a^t = W_hh h^{t-1} + W_xh x^t`:

    h^t = f[ (g / sigma^t) ⊙ (a^t - mu^t) + b ]

with `mu^t`, `sigma^t` over the `H` components of `a^t`, and a single `g`, `b` shared over time. Dividing `a^t` by its own scale makes the step invariant to positive re-scaling of the whole summed-input vector, which is the degree of freedom that compounds into exploding/vanishing dynamics over long sequences — so hidden-to-hidden dynamics are held at a fixed scale every step.

Properties that distinguish it: it is *not* a reparameterization of the original network (its statistics depend on the data through the per-example summed inputs), so it has its own invariances — invariant to positive re-scaling and re-centering of the whole weight matrix, and to positive re-scaling of an individual training case; not invariant to re-scaling a single weight vector or to re-centering the dataset. A Fisher-information analysis shows `sigma` growing in proportion to an invariant weight scale damps the derivative by `1/sigma`: a mixed block halves when the scale doubles, while the pure diagonal KL quadratic for a fixed absolute weight update quarters. This gives an automatic per-direction learning-rate decay ("implicit early stopping"), while making the geometry of magnitude-learning (the gain) depend on prediction error and scale-free normalized activations rather than raw input/weight scale.

Application notes: leave the final logit/softmax layer un-normalized (prediction confidence lives in the logit scale, which LN's scale-invariance would erase); in an RNN cell, normalize the recurrent contribution `W_h h` and the input contribution `W_x x` *separately* (they live at different scales; normalizing their sum jointly lets the larger dominate). Fully-connected layers satisfy the "all units contribute similarly" assumption that makes a single layer-wide statistic meaningful; convolutional layers do not (boundary units are rarely active and have different statistics), so LN is less suited there.

## Code

A PyTorch-style implementation matching `torch.nn.LayerNorm` and the labml annotated version:

```python
from typing import Union, List

import torch
from torch import nn, Size


class LayerNorm(nn.Module):
    """Standardize over the trailing feature axes of a single example, then
    restore a learned per-feature gain and bias. No batch dimension enters,
    so the computation is identical at train and test and valid at batch size 1.
    """
    def __init__(self, normalized_shape: Union[int, List[int], Size], *,
                 eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        assert isinstance(normalized_shape, torch.Size)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.gain = nn.Parameter(torch.ones(normalized_shape))   # init 1 -> identity start
            self.bias = nn.Parameter(torch.zeros(normalized_shape))  # init 0

    def forward(self, x: torch.Tensor):
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]
        # reduce over the trailing feature axes (per example), never the batch axis
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        mean = x.mean(dim=dims, keepdim=True)
        mean_x2 = (x ** 2).mean(dim=dims, keepdim=True)
        var = mean_x2 - mean ** 2                          # E[x^2] - E[x]^2
        x_norm = (x - mean) / torch.sqrt(var + self.eps)   # eps inside sqrt for stability
        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias        # affine restore
        return x_norm
```

The layer-normalized LSTM cell, applying separate LN to the recurrent and input
contributions and to the cell state before the output `tanh`:

```python
class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.W_x = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.b = nn.Parameter(torch.zeros(4 * hidden_size))
        self.ln_h = LayerNorm(4 * hidden_size)   # normalize recurrent contribution
        self.ln_x = LayerNorm(4 * hidden_size)   # normalize input contribution
        self.ln_c = LayerNorm(hidden_size)       # normalize cell before output tanh

    def forward(self, x_t, state):
        h_prev, c_prev = state
        # separate LN on each contribution, then add bias -> stable scale per step
        gates = self.ln_h(self.W_h(h_prev)) + self.ln_x(self.W_x(x_t)) + self.b
        f, i, o, g = gates.chunk(4, dim=-1)
        c_t = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
        h_t = torch.sigmoid(o) * torch.tanh(self.ln_c(c_t))
        return h_t, (h_t, c_t)
```
