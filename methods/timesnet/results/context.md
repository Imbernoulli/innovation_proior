# Context: temporal-variation modeling for general time series analysis (circa 2021-2022)

## Research question

Time series drive a huge range of applications — forecasting weather, imputing missing sensor
readings, classifying activity from trajectories, detecting anomalies in monitoring data. What
they have in common, and what makes them different from language or video, is that each
recorded time point is just a handful of scalars with almost no standalone semantic content.
The information lives in the *temporal variation* — how the signal rises, falls, oscillates,
trends. The hard part is that real-world variation is a tangle: multiple patterns of different
character overlap and interact within a single series.

A particularly load-bearing structural fact is *multi-periodicity*. A weather series carries a
daily cycle and a yearly cycle; an electricity-load series carries a weekly cycle and a
quarterly cycle; these periods coexist and interfere. Worse, within any one period the signal
has short-term structure (how it moves from one step to the next inside the cycle), and across
consecutive cycles it has longer-term structure (how the same phase drifts from one cycle to
the next). A single one-dimensional sequence can only ever expose adjacency between
consecutive time points; the cross-cycle relationships are buried far apart in the sequence.

The immediate task is unsupervised anomaly detection on multivariate monitoring data: train on
normal windows, flag the time points where the model fails to reproduce the signal. But the
goal is not an anomaly-specific trick. The point is a backbone for temporal-variation modeling
that (1) captures both the short-range structure inside a cycle and the long-range structure
across cycles; (2) copes with several overlapping periods at once rather than assuming a single
dominant one; (3) is not bottlenecked by the locality of 1D convolution or the sequential
dependence of recurrence; and (4) is *task-general* — the same representation should serve
forecasting, imputation, classification, and reconstruction-based anomaly detection, rather
than being hand-built for one of them. A strong variation model gives a strong reconstruction,
and a strong reconstruction makes the residual a clean anomaly signal.

## Background

By this time deep models have largely displaced the classical pattern-fitting approaches —
ARIMA (Anderson 1976), Holt-Winters (Hyndman & Athanasopoulos 2018), Prophet (Taylor &
Letham 2017) — which assume the variation follows a small family of pre-defined templates and
break when real variation is too complex to fit any template. The deep landscape is organized
by how each family models temporal variation:

- **Recurrence (RNN/LSTM; Hochreiter & Schmidhuber 1997; later state-space variants).** Model
  the series as a Markov chain of hidden-state transitions. The hidden state is meant to carry
  history forward, but in practice long-term dependencies fade, and the strictly sequential
  computation makes both training and inference slow.
- **Temporal convolution (TCN; Franceschi et al. 2019; Bai et al. 2018).** Slide 1D kernels
  along the time axis. A kernel of width w only sees a window of w adjacent time points, so a
  single convolution captures local variation; long-range structure needs many stacked,
  dilated layers, and even then the model only ever sees variation as *adjacency along one
  axis*.
- **Attention over time points (Informer, Zhou et al. 2021; Autoformer, Wu et al. 2021;
  FEDformer, Zhou et al. 2022).** Compute pairwise dependencies between time points directly.
  This removes the locality constraint, but reading reliable dependencies off scattered,
  individually-uninformative time points is hard — the genuine structure is obscured by the
  intricate mixture of variations, and the cost is quadratic in sequence length.

Two background frames matter especially. First, the **frequency view**: a real-valued
length-T series can be written through the Fast Fourier Transform as a sum of sinusoidal basis
functions; the amplitude of the frequency-j component measures how strongly a periodic pattern
of length ceil(T/j) is present. The frequency spectrum of real series is sparse — a few
frequencies carry most of the energy — and high frequencies are usually noise (Chatfield 1981).
By the conjugate symmetry of the FFT of a real signal, only frequencies 1..floor(T/2) are
independent. Second, the **2D-locality view** familiar from vision: convolutional kernels are
cheap and effective precisely when the structure they must capture is *local in the grid they
slide over*; a pattern that is non-local in the input layout is invisible to a small kernel.

A diagnostic observation about real data sets the stage: running an FFT over fixed-length
segments of the standard benchmarks and recording the most significant period lengths shows a
spread of several coexisting periods per series, not a single one — empirical confirmation
that multi-periodicity is the rule, not the exception. And on the anomaly-detection side, a
known failure mode anchors the problem: a plain attention encoder trained to reconstruct
windows scores worst among deep backbones, because its pairwise similarity is dominated by the
many normal points and the rare abnormal pattern gets washed out.

## Baselines

These are the prior backbones a new temporal-variation model would be measured against and
reacts to. For anomaly detection they are all dropped into the same reconstruction harness:
each plays the role of the base model whose reconstruction error is the anomaly criterion.

**Autoformer (Wu et al. 2021).** Introduces an Auto-Correlation mechanism: instead of
point-wise attention, it estimates the dominant period of the series (via the autocorrelation
function, computed efficiently through the FFT), rolls the series by those lags, and aggregates
sub-series that are one period apart — weighting the rolled copies by a softmax over their
correlation confidences, so dependencies are captured *series-wise* at the period scale rather
than point-wise. It also adds a deep decomposition architecture that progressively separates
the input into trend and seasonal parts. **Gap:** the aggregation is still carried out in the
original 1D layout — sub-series separated by a period are combined, but the short-range
structure *inside* a period and the structure *across* periods are not represented in a single
object that a local operator can process jointly; and it commits to essentially one period
scale per decomposition step.

**FEDformer (Zhou et al. 2022).** Builds on the decomposition idea with a mixture-of-experts
seasonal-trend decomposition and a sparse attention performed in the frequency domain (keeping
a random subset of Fourier modes), giving linear complexity. **Gap:** like Autoformer it
operates on 1D sequence/spectrum representations; periodicity is used to select frequency modes
but the intraperiod-versus-interperiod structure is never laid out so that a single local
operator sees both, and the frequency-mode subsampling is a fixed heuristic.

**TCN-based backbones (Franceschi et al. 2019; Bai et al. 2018).** Stacked dilated 1D
convolutions over the time axis; efficient and parallel. **Gap:** each kernel is local along
the single temporal axis, so capturing variation between points that are a full period apart
requires very deep or very dilated stacks, and the cross-period relationship is never made
adjacent to the kernel — it is always "p steps away" in the 1D layout.

**MLP-temporal backbones (N-BEATS, Oreshkin et al. 2019; DLinear/LightTS, Zeng et al. 2022;
Zhang et al. 2022).** Apply MLPs along the temporal dimension, baking the temporal dependency
into fixed weights; DLinear splits the window into a moving-average trend and a residual
seasonal part and applies a separate linear layer to each. **Gap:** the temporal-mixing pattern
is fixed by the learned weight matrix and tied to a fixed input length; it represents variation
as a flat function of absolute position and has no explicit handle on which periodicities a
given series carries, so it cannot separate the within-cycle and across-cycle components.

**Transformer (Vaswani et al. 2017) applied to reconstruction.** A plain encoder over the
window, trained to reconstruct it; reconstruction error is the anomaly score. **Gap (observed
on anomaly detection):** vanilla attention compares every pair of time points, and the
similarity is dominated by the many *normal* points, so the rare abnormal pattern the task
cares about gets washed out — the worst average F1 among the deep backbones.

The reusable vision pieces on the table: the **Inception block** (Szegedy et al. 2015), which
runs several 2D convolution kernels of different sizes in parallel and combines them, giving a
multi-scale receptive field in a single layer; and **residual connections** (He et al. 2016),
which let deep stacks of blocks be optimized stably. Both are mature, well-understood 2D-vision
primitives — but nothing has yet connected a *1D time series* to them in a way that lets a 2D
kernel do useful work.

A further reusable component is **Series Stationarization** (Liu et al. 2022): normalize each
input window by subtracting its mean and dividing by its standard deviation before the model,
and de-normalize the model output by the same statistics, to remove the per-window
distribution shift that otherwise destabilizes training.

## Evaluation settings

The natural yardsticks already in use, with anomaly detection as the primary target:

- **Unsupervised anomaly detection** on multivariate monitoring benchmarks: SMD (server
  machine, Su et al. 2019), MSL and SMAP (spacecraft / satellite telemetry, Hundman et al.
  2018), SWaT (water treatment, Mathur & Tippenhauer 2016), PSM (server, Abdulaal et al. 2021).
  Following the Anomaly Transformer pre-processing (Xu et al. 2021), each series is split into
  consecutive non-overlapping windows by a sliding window (window length 100). Labels are not
  used in training (unsupervised); the model is trained on normal data to reconstruct windows.
  Inputs are Z-score normalized per dataset. The shared anomaly criterion is the reconstruction
  error per time point; the threshold is set at the anomaly-ratio percentile (e.g. 1%) of the
  combined train/test score distribution so that fixed fraction of points is flagged; the
  metric is the point-adjusted F1-score (harmonic mean of precision and recall), F1 primary.
- **Long- and short-term forecasting** (ETT, Electricity, Traffic, Weather, Exchange, ILI; M4),
  metrics MSE/MAE and SMAPE/MASE/OWA respectively.
- **Imputation** (ETT, Electricity, Weather; random masking 12.5–50%), metrics MSE/MAE.
- **Classification** (10 UEA multivariate datasets), metric accuracy.
- Protocol: input embedding and final projection head are held identical across the compared
  backbones so only the backbone's representation power is measured; experiments use the Adam
  optimizer with (beta_1, beta_2) = (0.9, 0.999); for anomaly detection the model width is
  sized to the input dimension C by d_model = min(max(2^ceil(log C), d_min), d_max), batch
  size 128, learning rate 1e-4, up to 10 epochs.

## Code framework

The backbone plugs into the standard reconstruction harness used for the anomaly-detection
baselines. What already exists: a data pipeline that yields fixed-length windows `x_enc` of
shape `[batch, seq_len, enc_in]`; an Adam optimizer; an MSE reconstruction loss; the
per-window normalize/de-normalize utility (Series Stationarization); a generic value+position
input embedding; and the outer training loop. The framework computes the anomaly score
(per-point reconstruction MSE), the threshold (the anomaly-ratio percentile), and the metrics —
none of that is the backbone's job. What is *not* settled is the backbone that turns an
embedded window into a representation good enough to reconstruct: that is exactly the slot to
design. The scaffold below has one big empty slot for that backbone block, plus the empty
`Model.anomaly_detection` body that wires it up.

```python
import torch
import torch.nn as nn
from layers.Embed import DataEmbedding   # value + position (+ optional time) embedding (exists)


class TemporalBlock(nn.Module):
    """A single backbone block: maps an embedded window to a refined representation.
    [batch, seq_len, d_model] -> [batch, seq_len, d_model].  The architecture is
    exactly what we have to design."""

    def __init__(self, configs):
        super().__init__()
        # TODO: the block we will design.
        pass

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # TODO: produce a refined [batch, seq_len, d_model] representation.
        pass


class Model(nn.Module):
    """Reconstruction model for unsupervised anomaly detection.
    Embeds the window, runs a stack of backbone blocks, projects back to the input
    space; the framework turns the reconstruction error into an anomaly score."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.layer = configs.e_layers
        # generic value + position embedding that already exists
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model,
                                           configs.embed, configs.freq, configs.dropout)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.model = nn.ModuleList([TemporalBlock(configs) for _ in range(configs.e_layers)])
        # project the representation back to input channels for reconstruction
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def anomaly_detection(self, x_enc):
        # x_enc: [batch, seq_len, enc_in] -> reconstruction [batch, seq_len, c_out]
        # per-window normalize (Series Stationarization) is available; embed; run the
        # backbone stack; project back; de-normalize.
        # TODO: fill in once the backbone block above is designed.
        pass

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        return None


# existing reconstruction training loop the backbone plugs into
def train(model, data_loader, optimizer):
    mse = nn.MSELoss()
    for x_enc, x_mark_enc in data_loader:        # fixed-length normal windows
        optimizer.zero_grad()
        recon = model(x_enc, x_mark_enc, None, None)
        loss = mse(recon, x_enc)                 # reconstruct the normal window
        loss.backward()
        optimizer.step()
```

The outer loop supplies normalized windows and scores reconstruction error; `TemporalBlock` is
the empty slot where the backbone will live.
