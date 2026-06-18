## Research question

Can one classification component — a temporal encoder, a way of relating the channels, and a
padding-aware way of collapsing the time axis into a fixed-width feature — generalize across
heterogeneous multivariate time series when training and evaluation are pinned to a fixed protocol?
The three datasets deliberately stress different structure: **EthanolConcentration** is spectral
chemistry (4 classes, long smooth absorbance traces where the discriminative signal is the shape of
the curve, not any one wavelength), **FaceDetection** is MEG brain imaging (binary, ~144 noisy
channels, short windows, the signal lives in cross-channel covariation), and **Handwriting** is
tri-axial accelerometer character recognition (26 classes, the signal is a temporal gesture). The
single thing being designed is the `Model` that maps a padded window plus its padding mask to class
logits. Everything around it — the data loader, the per-dataset padding length, the optimizer, the
loss, early stopping — is frozen.

## Prior art before the first rung (sequence-classification lineage)

The first rung reacts to a specific, embarrassing fact about deep multivariate time-series models, so
the lineage worth naming is the one that produced that fact.

- **Distance-based classifiers (DTW-1NN; the UEA archive itself, Bagnall et al. 2018).** The archive's
  reference baseline is one-nearest-neighbour under dynamic time warping — align two series elastically,
  classify by the nearest training example. Strong and assumption-light, but it is lazy (no learned
  representation), scales badly, and treats each channel's warping independently. Gap: no shared
  representation, no cross-channel modeling, expensive at test time.
- **Recurrent encoders (LSTM/GRU classifiers).** Carry a hidden state along time and read the final
  state into a classifier. They model temporal order natively, but the discriminative cue at step *t*
  can sit a full period earlier, and that dependency must survive the recurrence — the long-range
  washout LSTMs are known for — and the sequential scan is slow. Gap: long-range dependence is fragile,
  training is serial.
- **Temporal convolution (TCN/InceptionTime, Fawaz et al. 2020).** Slide 1-D kernels along time, widen
  the receptive field with dilation or parallel kernel sizes. Fast and a strong TSC baseline, but a
  kernel only relates points that are *near along the time axis*; same-phase points one period apart
  are never in one kernel's window. Gap: cross-period structure is hard to reach.
- **Per-step Transformer encoders (the "vanilla" deep TSC recipe).** Embed the whole channel-vector at
  each timestamp into one token and run self-attention over the `seq_len` tokens. This is the recipe the
  first rung is built to undercut: a single timestamp has no standalone meaning (unlike a word), so
  point-wise attention compares the wrong objects, and the cost is quadratic in `seq_len`. The
  uncomfortable observation that motivates the whole ladder is that a *plain linear map* over the
  flattened window is competitive with these encoders on the standard benchmarks — which says the
  elaborate encoder may not be earning its keep, and that the input representation, not the attention
  kernel, is where the leverage is.

## The fixed substrate

The Time-Series-Library classification harness is frozen and must not be touched. It provides: the UEA
loader that reads each dataset, pads every window in a batch to the dataset's maximum length
`seq_len = max(train.max_seq_len, test.max_seq_len)`, and hands the model a binary mask
`x_mark_enc[b, t] ∈ {0, 1}` (1 for a real timestep, 0 for right-padding); dynamic config injection that
sets `configs.seq_len`, `configs.enc_in` (channel count), `configs.num_class`, and `configs.pred_len =
0` from the data before the model is built; RAdam optimization with `CrossEntropyLoss`; early stopping
with `patience = 10`; and per-dataset accuracy reporting. The library's `layers/` modules are importable
read-only building blocks (`Autoformer_EncDec.series_decomp`, `Embed.DataEmbedding` /
`Embed.PatchEmbedding` / `Embed.DataEmbedding_inverted`, `Conv_Blocks.Inception_Block_V1`,
`Transformer_EncDec.Encoder` / `EncoderLayer`, `SelfAttention_Family.FullAttention` / `AttentionLayer`).
The fixed run flags are `d_model 128`, `d_ff 256`, `e_layers 3`, `n_heads 16`, `dropout 0.1`,
`batch_size 16`, `learning_rate 1e-3`, `train_epochs 100` (early-stopped at patience 10).

## The editable interface

Exactly one file is editable: `Time-Series-Library/models/Custom.py`. It must define a `Model(nn.Module)`
whose `__init__(self, configs)` reads `configs.seq_len`, `configs.enc_in`, `configs.num_class`, and whose
`classification(self, x_enc, x_mark_enc)` maps `x_enc` of shape `[batch, seq_len, enc_in]` and the mask
`x_mark_enc` of shape `[batch, seq_len]` to `[batch, num_class]` pre-softmax logits, dispatched from
`forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)`. Every method on the ladder is a fill of this
same contract. The starting point is the scaffold default: it returns zeros — a no-op classifier that
predicts a constant for every input, i.e. chance-level accuracy. Each rung replaces this body with a
real temporal encoder and head.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Default scaffold fill: a no-op classifier (returns zero logits)."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.num_class = configs.num_class
        # no parameters yet — the rungs add the encoder + head here

    def classification(self, x_enc, x_mark_enc):
        # x_enc:      [batch, seq_len, enc_in]  padded window
        # x_mark_enc: [batch, seq_len]          1 = valid timestep, 0 = padding
        batch_size = x_enc.shape[0]
        return torch.zeros(batch_size, self.num_class).to(x_enc.device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
```

## Evaluation settings

Three UEA datasets — **EthanolConcentration** (4 classes, spectral), **FaceDetection** (2 classes, MEG,
many channels), **Handwriting** (26 classes, accelerometer) — under the fixed protocol above, seed 42.
One metric, **test accuracy, higher is better**, reported per dataset. The padding length, train/test
split, optimizer, loss, and early-stopping budget are identical for every rung; only `models/Custom.py`
changes.
