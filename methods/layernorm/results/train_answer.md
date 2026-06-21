What I actually care about is wall-clock training time: the networks that matter take days to converge with SGD, and the easy lever — splitting the batch or model across more machines — keeps disappointing me, because the communication cost rises and the returns fall off fast once I parallelize past a point. So I want the orthogonal lever: not "compute the same gradients faster on more hardware" but "change the forward computation so each gradient step is worth more." The difficulty lives inside a single layer. Neuron $i$ forms a summed input $a_i = w_i^\top h$ and fires $f(a_i + b_i)$; when I update the weights below, the distribution of $h$ moves, so the distribution of every $a_i$ in this layer moves too — and in a highly correlated way, since all the $a_i$ are driven by the same shifting $h$. With ReLU units below, those swings are large. Every lurch in a layer's input distribution forces the layer above to chase a moving target, which is exactly what pins me to tiny learning rates and fragile initialization. If I could hold the mean and spread of the summed inputs roughly fixed as the layers below churn, I could push the learning rate up.

There is a recipe that does this for feed-forward nets: for each neuron, standardize its summed input across the data, $\bar a_i = (g_i/\sigma_i)(a_i - \mu_i)$ with $\mu_i = \mathbb{E}_x[a_i]$ and $\sigma_i = \sqrt{\mathbb{E}_x[(a_i-\mu_i)^2]}$, then restore a learned gain and bias before the nonlinearity. The full-data expectation is impractical — it would mean a forward pass over the whole dataset at the current weights — so $\mu_i$ and $\sigma_i$ are estimated from the current mini-batch. It works: plain SGD converges much faster, and the jitter from estimating on a random batch even regularizes a little for free. But look at what the estimate is: $\mu_i$ and $\sigma_i$ are per-neuron, averaged down the batch dimension. That single fact is the source of everything that goes wrong. The estimate is only as good as the batch is big, so it gets noisy as the batch shrinks and is meaningless at batch size 1, the pure online regime. Training normalizes with batch statistics but test has no batch, so I must keep running averages and swap them in at test, which makes train and test compute different functions and adds bookkeeping. And the recurrent case really bites: the layer reuses the same weights at every step, $a^t = W_{hh}h^{t-1} + W_{xh}x^t$, but the distribution of $a^t$ differs at every $t$, so "per-neuron across the batch" becomes "per-neuron across the batch, separately for each $t$." I would have to store a $\mu,\sigma$ for $t=1,2,3,\dots$, and at test time a sequence longer than anything I trained on has no stored statistic to normalize with. The scheme assumes a fixed, enumerable set of slots, and an unbounded recurrence does not give me one. The reparameterization alternative — write $w = g\cdot v/\lVert v\rVert_2$, which is normalizing with $\mu=0$, $\sigma=\lVert w\rVert_2$ — is cheap and batch-independent but never re-centers and, being a pure relabeling of the original net, inherits exactly that net's invariances and nothing more.

I propose Layer Normalization. The fix is to turn the computation ninety degrees: instead of reducing over the batch dimension, reduce over the *feature* dimension — compute the mean and variance over the $H$ units of a layer, for one training case at a time. The summed inputs form a matrix whose rows are cases and columns are neurons; the batch recipe computes a statistic down each column, pooled over cases, while I compute a statistic along each row, pooled over neurons, so that all neurons in a layer share an example's statistics and different examples get different $\mu,\sigma$. This is not an arbitrary transpose: the disease was that a shift below moves all the $a_i$ in this layer together, and because that drift is largely common across the units it shows up precisely as a change in the *layer's* own mean and spread — the two quantities I now compute. Cure and disease line up, and I can see the drift inside a single example without ever consulting the batch. Concretely, for one training case over the $H$ units,
$$\mu = \frac{1}{H}\sum_{i=1}^{H} a_i,\qquad \sigma = \sqrt{\frac{1}{H}\sum_{i=1}^{H}(a_i-\mu)^2},\qquad h_i = f\!\left(\frac{g_i}{\sigma}(a_i-\mu) + b_i\right).$$
All three liabilities fall away at once: the statistics use only the current example, so batch size is irrelevant and size 1 is fine; training and test do literally the same computation; and an RNN normalizes at step $t$ with $\mu^t,\sigma^t$ taken over the $H$ components of $a^t$, with a single gain and bias shared across all steps regardless of sequence length.

I keep the per-neuron gain $g_i$ and bias $b_i$ and apply them after standardizing and before the nonlinearity, because standardizing pins the normalized vector to a centered, unit-scale coordinate system while the network may genuinely want each neuron at a different operating point going into $f$. The learned $g_i,b_i$ hand back that per-neuron scale and offset without re-introducing the common drift I just removed; placing them after normalization and before the nonlinearity keeps the nonlinearity always seeing a controlled range, whereas standardizing *after* the nonlinearity would fight saturating units and miss the point. I initialize $g_i=1$ and $b_i=0$ so the affine restore starts as the identity on the standardized signal and does not disrupt the initial dynamics — which is also why I do not need the tiny gain initialization (around $0.1$) that the recurrent batch scheme had to resort to. For the recurrent step the normalized form is
$$h^t = f\!\left[\frac{g}{\sigma^t}\odot(a^t-\mu^t) + b\right],$$
and dividing $a^t$ by its own scale buys a crucial invariance: scale the whole summed-input vector by any $c>0$ and both $(a^t-\mu^t)$ and $\sigma^t$ scale by $c$, so the ratio is unchanged. That overall magnitude is exactly the degree of freedom the recurrence compounds into exploding or vanishing dynamics over a long sequence; quotienting it out holds the hidden-to-hidden dynamics at a fixed scale every step, which for long sequences with tiny batches — where the batch recipe is useless anyway — is the whole game.

What makes this more than a coordinate change is that it is genuinely *not* a reparameterization of the original network. Its $\mu$ and $\sigma$ depend on this example's actual summed inputs through the data, not on a fixed function of the weights, so re-centering by the layer's own per-example mean is something the original net cannot reproduce by relabeling its weights. Working out the invariance table makes the trade explicit. Rescale one neuron's incoming weights $w_i \to \delta w_i$: under the batch and weight recipes the statistics for neuron $i$ are functions of $w_i$ alone, so $a_i,\mu_i,\sigma_i$ all scale together and the ratio is fixed, but under Layer Normalization $\mu,\sigma$ are pooled over all units, so a single row's rescaling shifts the ratio — I lose that invariance. What I gain instead is invariance to scaling and re-centering the *whole* matrix: take $W' = \delta W + \mathbf{1}\gamma^\top$, so every row becomes $\delta w_i + \gamma$ and $a'_i = \delta a_i + \gamma^\top x$ with the same scalar $\gamma^\top x$ added to every unit; then $\mu' = \delta\mu + \gamma^\top x$, the centered values $a'_i - \mu' = \delta(a_i-\mu)$ cancel the common additive term exactly, $\sigma' = \delta\sigma$, and $(a'_i-\mu')/\sigma' = (a_i-\mu)/\sigma$, so the output is identical. This works only because I normalize *after* the weights, not before. On the data side I uniquely gain per-example invariance: rescale a single case $x' = \delta x$ and, since this example's $\mu,\sigma$ scale with it, $h_i' = f\big((g_i/\sigma')(w_i^\top(\delta x) - \mu') + b_i\big) = h_i$ — the batch recipe cannot do this because its statistics pool over data and the weight recipe cannot because its normalizer is tied to $\lVert w_i\rVert$ rather than this case's activation scale. I trade away single-weight-vector and dataset-recentering invariance for whole-matrix rescale/recenter and per-example rescale invariance, and that per-example invariance is a real asset for inner hidden layers where a single example's overall input magnitude should not matter.

Equal function classes do not imply equal learning dynamics, so I also read off the local geometry — how far a small parameter step actually moves the model's output, measured by the KL divergence between output distributions, which to second order is the quadratic form $\frac{1}{2}\delta^\top F(\theta)\delta$ with $F(\theta) = \mathbb{E}[(\partial\log P/\partial\theta)(\partial\log P/\partial\theta)^\top]$ the Fisher information matrix. Treating a neuron as a generalized linear model and a layer with a block-diagonal $F$, the key step is that under normalization a move in $w_i$ changes the standardized $z_i=(a_i-\mu_i)/\sigma_i$ by $dz_i = (1/\sigma_i)\chi_i^\top dw_i$, where the raw input is replaced by $\chi_i = x - \partial\mu_i/\partial w_i - \big((a_i-\mu_i)/\sigma_i\big)\partial\sigma_i/\partial w_i$ because $w_i$ acts on the normalized input three ways at once — directly, through the mean, and through the standard deviation. The derivative therefore carries a factor $g_i/\sigma_i$, so when an invariant rescaling doubles the relevant weight scale, $\sigma_i$ doubles: a one-sided sensitivity halves, a mixed Fisher block halves, and the pure $w_i$–$w_i$ diagonal quadratic for a fixed absolute $dw_i$ drops to a quarter. The norm has quietly become a per-direction step-size control — a larger-norm weight vector is harder to rotate because a fixed optimizer step is a smaller angular change, an automatic per-direction learning-rate decay. The zero directions confirm the function-level story: $w_i^\top\chi_i = 0$ along the batch recipe's rescaling direction and along the weight recipe's, while for Layer Normalization the single-row direction is not a zero direction but summing the blocks along the whole-matrix rescaling direction $\mathrm{vec}([W,0,0]^\top)$ and along the pure recentering direction $\mathrm{vec}([\mathbf{1}\gamma^\top,0,0]^\top)$ both give zero. And the explicit gain direction behaves well: under Layer Normalization a pure gain update sees $\frac{1}{2}\delta_g^\top \frac{1}{\phi^2}\mathbb{E}_x\big[\{\mathrm{Cov}(y_i,y_j\mid x)\,(a_i-\mu)(a_j-\mu)/\sigma^2\}_{ij}\big]\delta_g$, i.e. the prediction-error covariance weighted by scale-free normalized activations, whereas in the un-normalized model the equivalent magnitude move depends on $a_i a_j = (w_i^\top x)(w_j^\top x)$, the raw input and weight scale. Learning the magnitude of the incoming weights is therefore far more robust to input and parameter scaling — the other half of why this trains faster and more reliably.

A few boundaries the geometry tells me to respect. I leave the final logit/softmax layer un-normalized: the *scale* of the logits is the prediction confidence, and Layer Normalization is by construction scale-invariant, so normalizing the logits would erase exactly the information the classifier needs. In a recurrent cell I do not standardize the summed input $a^t = W_h h^{t-1} + W_x x^t$ as one lump, because the recurrent and input contributions generally live at different scales and a joint standardization lets the larger drown out the smaller in the shared $\mu,\sigma$; I apply a separate normalization to each contribution and only then add the bias. And I expect convolutional layers to be a poor fit: my whole justification rested on the units of a layer contributing similarly so that one layer-wide moment summarizes them, but in a conv layer the boundary units are rarely active and carry statistics unlike the rest, so a single layer-wide statistic mixes very different populations — fully-connected layers, where the assumption holds, are the natural home.

The core operator standardizes a vector along its trailing feature axes and restores a per-feature affine, computing the variance as $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ in one pass with $\varepsilon$ placed inside the square root so a near-constant layer does not blow up:

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

The layer-normalized LSTM cell applies separate normalization to the recurrent and input contributions and to the cell state before the output `tanh`:

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
