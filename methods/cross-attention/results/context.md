# Context: learning over heterogeneous gridded geophysical fields (circa 2021-2022)

## Research question

Weather and climate data is a stack of many distinct physical fields defined on the same
spatial grid: 2 m temperature, 10 m wind components, geopotential, temperature, wind,
relative and specific humidity at a range of pressure levels, plus static surface fields
like land-sea mask and orography. A single training example is therefore a tensor of shape
`V × H × W`: `V` physical variables, each a map over an `H × W` latitude-longitude grid.
The number and identity of these variables is *not* fixed across data sources. Different
reanalysis products, and the many climate-simulation datasets that are candidate fuel for
large-scale pretraining, each carry a *different set* of variables — one collection might
provide temperature and humidity, another wind and geopotential, a third some
overlapping-but-not-identical mixture. At deployment one often wants to feed the model only
a *subset* of variables, or a variable that a particular source simply does not record.

The question is how to turn the per-variable spatial content at each grid location into one
fixed-width representation that a shared sequence backbone can process, when the set of
variables presented to the model — any count, any subset drawn from a known vocabulary,
possibly a variable not seen in a given pretraining run — varies from input to input.

## Background

By this time, data-driven weather prediction has become a serious alternative to numerical
weather prediction. The dominant recipe is to train a deep network on decades of historical
gridded reanalysis (ERA5; Hersbach et al. 2020) to map the current atmospheric state to a
future one, and there is mounting evidence (Rasp & Thuerey 2021; Weyn et al. 2020; Pathak
et al. 2022; Bi et al. 2022) that such networks can rival operational numerical systems on
medium-range forecasting while running orders of magnitude faster at inference. Several
facts about the setting shape the architecture question:

- **The variable axis is treated as a channel axis.** In standard image models, the `V`
  physical fields are stacked as the input channels of a `V × H × W` tensor. The first layer
  — a convolution or a patch projection — has its input dimension set to exactly `V`.
- **Heterogeneous physical groundings.** The variables are not interchangeable channels like
  the R, G, B of a photograph. Geopotential at 500 hPa, specific humidity at 925 hPa, and
  10 m wind live on different scales, carry different units, and obey different dynamics.
  After normalization they coexist numerically, but a token that summarizes "temperature
  here" and a token that summarizes "humidity here" are semantically distinct objects.
- **Set-of-tokens view and cost.** A token sequence can be any length, so treating the data
  as a *set of tokens* rather than a fixed-channel image admits an arbitrary variable set. If
  every variable contributes its own tokens at every spatial patch, the token count is `V`
  times the number of patches. With attention as the sequence model and its quadratic cost in
  sequence length, multiplying the length by `V` multiplies the cost by `V²`. Realistic
  inputs reach a few dozen variables.
- **The foundation-model paradigm.** Pretraining one large Transformer on broad data with a
  self-supervised objective and finetuning it to many downstream tasks (Bommasani et al. 2021;
  Devlin et al. 2018; He et al. 2022) has reshaped language and vision, motivating a single
  architecture that can consume broad, heterogeneous data in this domain.

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
logits. A structural property: the softmax runs over the *set of keys*, so the output is
invariant to the ordering of the key-value pairs and is defined for *any number* of them.
Multi-head attention runs `h` of these in parallel over learned projections to
`d_k = d_model/h` dimensions and concatenates,
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
interpolating a positional embedding). ViT also uses a **learnable `[class]` token**: an
extra trainable embedding prepended to the patch sequence whose output state, after the
Transformer mixes it with the patches, serves as the pooled representation of the whole
image. ViT folds the channels `C` into the patch vector `p²·C`.

## Baselines

These are the prior approaches a new design would be measured against and would react to.

**Channel-stacking image models (UNet, ResNet, ViT applied to `V × H × W`).** Treat the
forecasting problem as image-to-image translation with `V` input channels and `V'` output
channels (Rasp & Thuerey 2021 used a ResNet pretrained on climate simulations; Pathak et al.
2022 and Bi et al. 2022 stack variables as channels of a single tensor). Core mechanism: the
first layer maps `V` input channels to feature maps; everything downstream is
channel-agnostic. Convolutional variants operate on a complete, regular grid.

**Uniform pooling across the variable tokens.** Once each variable is given its own token at
a spatial location, average the `V` tokens: take the per-location mean over the variable axis.
It adds no parameters and accepts any number of variables.

**Fixed learned weighting across the variable tokens.** Attach one trainable scalar per
variable, normalize the scalars across variables with a softmax, and take that weighted sum
over the variable axis. More expressive than the plain mean and still cheap; once trained the
weights are the same regardless of the token contents at a location.

**Self-attention over the full variable-and-space token sequence.** Keep every variable's
tokens at every patch and let the Transformer's self-attention mix the whole `V·h·w`-length
sequence directly. This is maximally expressive and content-dependent; the sequence length
carries the `V` factor, so attention cost is `O((V·h·w)²)`.

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
