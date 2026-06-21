# Context: efficient cross-channel modeling for multivariate time series forecasting (circa 2023-2024)

## Research question

Multivariate time series forecasting (MTSF) takes a lookback window of `C` channels (variables) over
`L` past time steps, `X ∈ R^{C×L}`, and must predict the next `H` steps, `Y ∈ R^{C×H}`. Real deployments
in traffic, energy, and electricity routinely have hundreds to nearly a thousand channels (a city's traffic
sensors, a grid's meters), and the signal at each channel is non-stationary: its mean and scale drift over
days and seasons, and individual channels go anomalous (a stuck sensor, a missing meter). Channels are
correlated — nearby road sensors rise and fall together — so one channel's history can inform another's
forecast. The question is how to design a forecasting component that captures cross-channel information
while keeping computational cost manageable as `C` grows into the hundreds.

## Background

**The temporal-modeling backbone and the move to long horizons.** Early deep forecasters were RNNs (LSTM,
GRU) and temporal convolutional networks. Transformers entered the field to capture long-range temporal
dependence through attention, and a line of efficient variants — Informer (probabilistic sparse attention),
Autoformer (autocorrelation + FFT), FEDformer (frequency-domain attention) — pushed long-horizon accuracy
forward. In these models a *token is a timestamp*: at each time step the `C` channel values are embedded
together into one vector, and attention runs along the time axis.

**The robustness finding that reshaped the field.** A simple one-layer linear model (DLinear; Zeng et al.,
AAAI 2023, "Are Transformers Effective for Time Series Forecasting?") matched or beat elaborate
channel-mixing Transformers on the standard benchmarks. The diagnosis (Han, Ye & Zhan, 2023, "The capacity
and robustness trade-off") is that multivariate series are *non-stationary*: the joint distribution over
channels drifts between train and test, so a model that fits cross-channel structure can overfit to
correlations that no longer hold at test time. This is the load-bearing empirical fact for everything that
follows: **cross-channel correlation is informative but, on real non-stationary data, the per-pair
correlations a model extracts are not always trustworthy.**

**Channel independence and what it offers.** The response was the *channel-independent* (CI) strategy:
process each channel's series on its own with a shared-weight model (PatchTST; Nie et al., ICLR 2023;
SegRNN; the linear models). CI is robust — there is no cross-channel structure to overfit — and it is
cheap. On highly correlated data (dense sensor grids), genuine inter-channel correlation also exists as
real signal.

**Reversible instance normalization (RevIN; Kim et al., ICLR 2021).** A standard tool for fighting
distribution shift: before forecasting, subtract each instance's per-channel mean over the lookback and
divide by its per-channel standard deviation, so the model sees a zero-mean, unit-variance series; after
forecasting, undo the transform on the prediction. This removes the slowly drifting local level/scale (the
part most responsible for train/test mismatch) and lets the network model the normalized shape.

**Permutation-invariant set aggregation.** A separate body of theory governs functions of an unordered
*set*. The Kolmogorov–Arnold representation theorem (1961) writes any continuous multivariate function as
`ρ(Σ_m λ_m φ(x_m))` — an inner map `φ` summed with per-coordinate weights `λ_m`, then an outer map `ρ`; the
`λ_m` make it depend on the *position* of each argument. DeepSets (Zaheer et al., NeurIPS 2017) gives the
permutation-*invariant* version: any continuous permutation-invariant set function can be approximated as
`ρ(Σ_{x∈X} φ(x))` — same `φ` applied to every element, summed with equal weight, then `ρ`. The two differ
only in whether the summands carry coordinate-specific weights, i.e. whether the output depends on element
ordering. This is the mathematics of "summarize a collection of channels into one vector without privileging
any channel's index."

**Stochastic pooling (Zeiler & Fergus, ICLR 2013).** A pooling operator introduced as a regularizer for
conv nets. Over the activations `a_i` to be pooled, form probabilities from the activation magnitudes; in the
original nonnegative-activation setting this is `p_i = a_i / Σ_k a_k`, and for unconstrained learned
activations the same idea can be implemented by a softmax over the items being pooled. During *training*,
sample one element `c ∼ Multinomial(p)` and output `a_c`; at *test* time output the expectation
`Σ_i p_i a_i`. It sits between max pooling (which always takes the largest) and mean pooling (which always
averages): training-time sampling injects noise tied to activation magnitude, and the test-time expectation
is a magnitude-weighted average.

## Baselines

**Channel-mixing Transformers (Informer, Zhou et al. 2021; Autoformer, Wu et al. 2021; FEDformer, Zhou et al.
2022).** Token = timestamp: embed the `C` channels at each step into one vector, attend along time.
Informer sparsifies attention probabilistically, Autoformer replaces the attention map with
autocorrelation accelerated by FFT, FEDformer attends in the frequency domain on selected components — all to
tame the `O(L^2)` cost of long sequences.

**DLinear / linear forecasters (Zeng et al., AAAI 2023).** Decompose the series into a moving-average trend
and a residual seasonal component, run one shared linear layer per component mapping `L → H`, channel by
channel.

**PatchTST / channel-independent Transformers (Nie et al., ICLR 2023).** Cut each channel's series into
patches, embed patches as tokens, run a Transformer along time *within each channel separately* with shared
weights, then a flatten-and-linear head to `H`.

**Cross-channel-attention forecasters (Crossformer, Zhang & Yan 2023; iTransformer, Liu et al., ICLR 2024).**
The most direct precursors. iTransformer *inverts* the tokenization: a token is now a whole channel's series
— the `L` past values of one variable are embedded into one `d`-dim "variate token" by a single linear map.
Self-attention then runs *across the channel tokens* (so it explicitly models inter-channel correlation),
while a position-wise feed-forward network acts *within* each channel token to learn its temporal nonlinear
representation; LayerNorm and residuals wrap each sublayer; a linear head maps each refined token to the `H`
forecast.

**MLP mixers for time series (TSMixer; LightTS; FreTS).** Replace attention with cheap MLP mixing along the
time and/or channel axes.

## Evaluation settings

The established yardstick for long-term MTSF at this time:

- **Datasets** spanning channel counts from a handful to ~900: ETT (4 subsets, 7 channels), Weather
  (21), Electricity/ECL (321), Solar-Energy (137), Traffic (862), and the PEMS subsets (PEMS03/04/07/08,
  ranging ~170–883 channels). The large-`C` sets (Traffic, PEMS) are the ones that stress scalability.
- **Protocol:** fixed lookback `L = 96`; prediction horizon `H ∈ {12, 24, 48, 96}` for PEMS and
  `{96, 192, 336, 720}` for the rest; report averaged over horizons. Series are z-score normalized; for the
  PEMS sets, whether to apply instance normalization is treated as a tuned choice.
- **Metrics:** Mean Squared Error and Mean Absolute Error, lower is better, in the original scale after the
  inverse normalization.
- **Efficiency axis:** memory and wall-clock vs. number of channels (e.g. on Traffic at `L=96, H=720`, small
  batch) — the curve that exposes scaling behavior as `C` grows.
- **Optimization:** Adam; MSE training loss; identical data splits across methods so accuracy differences are
  attributable to the model.

## Code framework

The forecasting component plugs into a fixed PyTorch training/evaluation harness: the pipeline owns the data
loader, normalization used by the dataset, Adam optimization, MSE training loss, and the train/eval loop. What
is not settled is the model itself — the mechanism that turns the lookback window into the forecast is exactly
what is to be designed — so the substrate is only the generic pieces that already exist: a reversible
per-instance normalization step, a way to turn each channel's lookback into a vector, an empty stack of
interaction layers, and a linear head to the horizon. The cross-channel operation inside each layer is the one
big open slot.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible instance normalization: remove each instance's per-channel
    level/scale over the lookback, restore it on the prediction."""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            x = x - self.mean
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps)
            return x / self.stdev
        else:
            return x * self.stdev + self.mean


class InteractionLayer(nn.Module):
    """One layer that refines the per-channel tokens. The interaction mechanism
    is the open question."""
    def __init__(self, hidden_size):
        super().__init__()
        # TODO: the architecture we will design
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):                       # x: [B, C, d], one token per channel
        # TODO: x = self.norm1(x + <interaction>(x))
        x = self.norm2(x + self.ffn(x))
        return x


class Forecaster(nn.Module):
    def __init__(self, input_len, output_len, hidden_size, num_layers):
        super().__init__()
        self.revin = RevIN()
        # each channel's whole lookback -> one d-dim token
        self.embed = nn.Linear(input_len, hidden_size)
        self.layers = nn.ModuleList([
            InteractionLayer(hidden_size) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, output_len)

    def forward(self, inputs, inputs_timestamps):
        # inputs: [B, L, C]
        x = self.revin(inputs, "norm")
        N = x.size(-1)
        h = self.embed(x.transpose(1, 2))       # [B, C, d]
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        pred = self.head(h).transpose(1, 2)[:, :, :N]   # [B, H, C]
        return self.revin(pred, "denorm")
```

The token-per-channel layout and the linear head are settled; the body of the interaction module — how the
`C` channel tokens should exchange information — is the single empty slot the method will fill.
