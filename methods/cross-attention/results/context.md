# Context: learning over heterogeneous gridded geophysical fields (circa 2021-2022)

## Research question

Weather and climate data is a stack of many distinct physical fields defined on the same
spatial grid: 2 m temperature, 10 m wind components, geopotential, temperature, wind,
relative and specific humidity at a range of pressure levels, plus static surface fields
like land-sea mask and orography. A single training example is therefore a tensor of shape
`V × H × W`: `V` physical variables, each a map over an `H × W` latitude-longitude grid.
The number and identity of these variables is *not* fixed across data sources. Different
reanalysis products, and especially the many climate-simulation datasets that would be the
natural fuel for large-scale pretraining, each carry a *different set* of variables — one
collection might provide temperature and humidity, another wind and geopotential, a third
some overlapping-but-not-identical mixture. At deployment one often wants to feed the model
only a *subset* of variables, or a variable that a particular source simply does not record.

The precise goal is an architecture that can ingest an *arbitrary set* of these variables —
any count, any subset drawn from a known vocabulary of possible variables, including a
variable not seen in a given pretraining run — and turn the per-variable spatial content at
each grid location into one fixed-width representation that a shared sequence backbone can
process. It must do this (1) without re-architecting or retraining a new input layer every
time the variable set changes; (2) at a compute cost that does not explode as the number of
variables grows toward the few dozen present in realistic data; and (3) in a way that lets
the backbone learn cleanly despite the variables having wildly different physical units,
scales, and dynamics. None of the architectures on the table below achieves all three at
once. Closing that gap is the problem.

## Background

By this time, data-driven weather prediction has become a serious alternative to numerical
weather prediction. The dominant recipe is to train a deep network on decades of historical
gridded reanalysis (ERA5; Hersbach et al. 2020) to map the current atmospheric state to a
future one, and there is mounting evidence (Rasp & Thuerey 2021; Weyn et al. 2020; Pathak
et al. 2022; Bi et al. 2022) that such networks can rival operational numerical systems on
medium-range forecasting while running orders of magnitude faster at inference. The pain
points that shape the architecture question:

- **The variable axis is treated as a channel axis, which hard-wires the variable set.**
  In every standard image model, the `V` physical fields are simply stacked as the input
  channels of a `V × H × W` tensor. The very first layer — a convolution or a patch
  projection — has its input dimension fixed to exactly `V`. Change which variables you
  feed, and that layer no longer fits; the model is welded to one variable set. This is the
  single fact that blocks pretraining across heterogeneous data sources and blocks ingesting
  a subset of variables at finetune time.
- **Heterogeneous physical groundings.** The variables are not interchangeable channels like
  the R, G, B of a photograph. Geopotential at 500 hPa, specific humidity at 925 hPa, and
  10 m wind live on different scales, carry different units, and obey different dynamics.
  After normalization they coexist numerically, but a token that summarizes "temperature
  here" and a token that summarizes "humidity here" are semantically very different objects.
- **Sequence-length / cost pressure if variables become tokens.** The flexibility one wants
  comes from treating data as a *set of tokens* rather than a fixed-channel image, because a
  token sequence can be any length. But if every variable contributes its own tokens at
  every spatial patch, the token count multiplies by `V`. With attention as the sequence
  model and its quadratic cost in sequence length, multiplying the length by `V` multiplies
  the cost by `V²`. Realistic inputs reach a few dozen variables, so this is not a constant
  factor one can ignore.
- **The foundation-model paradigm raises the stakes.** Pretraining one large Transformer on
  broad data with a self-supervised objective and finetuning it to many downstream tasks
  (Bommasani et al. 2021; Devlin et al. 2018; He et al. 2022) has reshaped language and
  vision. To bring it to this domain, the single architecture must be able to *consume* the
  broad, heterogeneous data — which is exactly what the fixed-channel input layer forbids.

Two background frames are load-bearing.

First, the **attention** mechanism (Vaswani et al. 2017). An attention layer maps a query
and a set of key-value pairs to an output that is a weighted average of the values, the
weight on each value being a learned compatibility between the query and that value's key.
The scaled dot-product form is

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V,
```

with the `1/√d_k` scale present because, for query/key components that are independent with
unit variance, the dot product `q·k = Σ_{i=1}^{d_k} q_i k_i` has mean 0 and variance `d_k`,
so its magnitude grows like `√d_k`; left unscaled, large logits push the softmax into a
near-one-hot regime where its gradient is tiny, and dividing by `√d_k` restores unit-variance
logits. The decisive structural property for the present problem: the softmax runs over the
*set of keys*, so the output is invariant to the ordering of the key-value pairs and is
defined for *any number* of them. Multi-head attention runs `h` of these in parallel over
learned projections to `d_k = d_model/h` dimensions and concatenates,
`MultiHead(Q,K,V) = Concat(head_1,…,head_h) Wᴼ`,
`head_i = Attention(Q Wᵢ^Q, K Wᵢ^K, V Wᵢ^V)`, so that different heads can pick up different
relations at once instead of being blurred into a single averaged pattern; with
`d_k = d_v = d_model/h` the total cost matches one full-width head. The queries need not come
from the same place as the keys and values — in the original sequence-transduction setting the
decoder's queries attend over the encoder's keys and values — so the query side and the
key/value side are independent inputs to the layer.

Second, the **Vision Transformer** (Dosovitskiy et al. 2020) showed how to make an image
into a token sequence: cut it into non-overlapping `p × p` patches, flatten each, and
linearly project it to a `D`-dimensional embedding (a single convolution with kernel and
stride `p`), giving `(H/p)·(W/p)` patch tokens. Because the model then operates on a token
sequence, it accepts variable sequence lengths and variable spatial resolution (adjusted by
interpolating a positional embedding). ViT also introduced a **learnable `[class]` token**:
an extra trainable embedding prepended to the patch sequence whose output state, after the
Transformer mixes it with the patches, serves as the pooled representation of the whole
image — a single trainable vector that gathers a set of tokens into one summary. ViT still
folds the channels `C` into the patch vector `p²·C`, so it inherits the same fixed-channel
limitation as a CNN; its contribution here is the token-sequence view and the trainable
summarizing token, not a way to handle a variable channel set.

## Baselines

These are the prior approaches a new design would be measured against and would react to.

**Channel-stacking image models (UNet, ResNet, ViT applied to `V × H × W`).** Treat the
forecasting problem as image-to-image translation with `V` input channels and `V'` output
channels (Rasp & Thuerey 2021 used a ResNet pretrained on climate simulations; Pathak et al.
2022 and Bi et al. 2022 stack variables as channels of a single tensor). Core mechanism: the
first layer maps `V` input channels to feature maps; everything downstream is channel-agnostic.
This is simple and strong on a fixed variable set. **Gap:** the input layer's channel count
is fixed at `V`, so the model is bound to exactly the variable set it was built for. It cannot
be pretrained across sources with different variable sets, cannot accept a subset of variables,
and cannot take in a variable absent from its original channel list — and convolutional
variants additionally require a complete, regular grid. The variable identity is also lost the
moment the channels are summed in the first convolution; there is no representation of "which
variable contributed what."

**Uniform pooling across the variable tokens.** Once each variable is given its own token at
a spatial location, the cheapest way to collapse the `V` tokens into one is to average them:
take the per-location mean over the variable axis. It adds no parameters and trivially accepts
any number of variables. **Gap:** the average weights every variable identically at every
location and in every atmospheric state, so it cannot give more weight to whichever variable
is informative for a given place and condition; it also blends the differently-grounded
variable tokens into one muddied vector rather than re-expressing them in a shared form.

**Fixed learned weighting across the variable tokens.** A small step up: attach one trainable
scalar per variable, normalize the scalars across variables with a softmax, and take that
weighted sum over the variable axis. More expressive than the plain mean and still cheap.
**Gap:** the weights are *static* — once trained they are the same regardless of the actual
token contents at a location or the current state of the atmosphere — so the combination
remains content-independent, and like the mean it only rescales-and-adds the raw variable
tokens without re-projecting them into a common representation.

**Self-attention over the full variable-and-space token sequence.** Keep every variable's
tokens at every patch and let the Transformer's self-attention mix the whole `V·h·w`-length
sequence directly. This is maximally expressive and content-dependent. **Gap:** the sequence
length carries the `V` factor, so attention cost is `O((V·h·w)²)` — quadratic in the variable
count on top of the spatial cost — which is impractical at the few-dozen variables of real
inputs, and it leaves the backbone to sort out the heterogeneous-token soup with no prior
reduction.

## Evaluation settings

The yardsticks already established in the field:

- **Reanalysis benchmark data.** ERA5 (Hersbach et al. 2020), regridded to the commonly used
  WeatherBench resolutions of 5.625° (a 32 × 64 grid) and 1.40625° (128 × 256), is the
  standard source for training and benchmarking data-driven forecasting (Rasp et al. 2020,
  WeatherBench; Rasp & Thuerey 2021). Large simulation archives (the CMIP6 collection of
  climate-model runs) are the candidate broad data for pretraining.
- **Forecasting targets and protocol.** Map the current atmospheric state to a future one at
  a given lead time; evaluate held-out forecasts of key variables at several lead-time
  horizons following the WeatherBench protocol (Rasp et al. 2020; Rasp & Thuerey 2021).
  Representative targets: 500 hPa geopotential at a 3-day lead, 850 hPa temperature at 5 days,
  10 m wind speed at 7 days.
- **Metric — latitude-weighted RMSE.** Equal-degree grid cells cover more area near the
  equator than near the poles, so each cell's error is weighted by the cosine of its latitude.
  With `lat(i)` the latitude of grid row `i`, the per-row weight is

  ```
  L(i) = cos(lat(i)) / ( (1/H) Σ_{i'} cos(lat(i')) ),
  ```

  normalized so the weights average to one, and the score is the square root of the
  latitude-weighted mean squared error over the grid, averaged over forecasts (Rasp et al.
  2020). The same weighting defines the latitude-weighted MSE used as the training loss.
- **Normalization and harness.** Each variable (each pressure level for atmospheric fields)
  is standardized to zero mean and unit variance before training and de-normalized before the
  metric is computed; models are trained with AdamW and the data managed with the standard
  PyTorch/`timm`/`xarray` stack.

## Code framework

A sequence backbone (a Vision Transformer over spatial patch tokens) and the rest of the
pipeline — data loading, normalization, optimizer, the latitude-weighted loss — already exist
and are fixed. What is not settled is how a *set of per-variable tokens at each spatial
location* should be turned into a *single token per location* that the backbone consumes. So
the substrate is the generic machinery that already exists: per-variable patch tokenization
that produces, for each spatial patch, one `D`-dimensional token per input variable; standard
PyTorch layer and tensor primitives; and the backbone that wants one token per spatial
location. The single empty slot is the module that performs that reduction.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariableAggregator(nn.Module):
    """Reduce a set of per-variable tokens at each spatial location to one token.

    Input:  x : [B, V, L, D]  — for each of B examples, V variable tokens at each of
                                L spatial patches, each token a D-vector.
    Output:     [B, L, D]     — one token per spatial location for the sequence backbone.

    V (the number of variables) is read from the input at runtime and may differ between
    inputs; the module must accept any V. Standard PyTorch primitives such as nn.Linear,
    nn.LayerNorm, tensor reshaping, and F.softmax are available.
    """

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vars = num_vars
        # TODO: the per-location set-reduction we will design.

    def forward(self, x):
        b, v, l, d = x.shape
        # TODO: produce one backbone token for each spatial location, returning [B, L, D].
        raise NotImplementedError


# existing pipeline the module plugs into
def forward_encoder(tokenize, aggregate, backbone, pos_embed, imgs, variables):
    x = tokenize(imgs, variables)   # [B, V, L, D] : per-variable patch tokens (already exists)
    x = aggregate(x)                # [B, L, D]    : the slot to fill
    x = x + pos_embed               # spatial position embedding (already exists)
    return backbone(x)              # ViT blocks -> [B, L, D] (already exists)
```

The encoder supplies one `D`-vector per variable per spatial patch; `aggregate` is where
the per-location reduction rule will live.
