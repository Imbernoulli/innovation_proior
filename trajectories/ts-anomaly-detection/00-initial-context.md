## Research question

Unsupervised anomaly detection on heterogeneous multivariate time series — server metrics (PSM, 25
channels), spacecraft telemetry (MSL, 55 channels), satellite telemetry (SMAP, 25 channels). Labels
are too rare and too expensive to train on, so the model only ever sees data that is overwhelmingly
normal. The operational recipe is **reconstruction**: train a model to reproduce normal windows, then
flag the points it fails to reproduce, because a model that has only ever seen normal variation should
choke on the abnormal. The anomaly score is the per-point reconstruction MSE; the harness then sets a
threshold at the per-dataset `anomaly_ratio` percentile of the score distribution and scores
point-adjusted F1 / precision / recall against ground-truth intervals. So the *only* thing being
designed is the reconstruction model `Model` in `models/Custom.py`: window in, same-length
reconstruction out. A better variation model gives a tighter reconstruction on normal points and a
cleaner spike on anomalies — that is the whole game.

## Prior art before the first rung (reconstruction-backbone lineage)

The ladder is a sequence of *reconstruction backbones*: each step replaces the architecture that maps a
normal window to its reconstruction, and the reconstruction error becomes the score. These are the
methods the first rung reacts to.

- **Dense autoencoders / LSTM-AE (Malhotra et al. 2016; Hundman et al. 2018).** Encode the window to a
  bottleneck and decode it back, or roll an LSTM forward and reconstruct/predict. The LSTM carries its
  state step by step, so a dependency on the same phase one period back must survive that many state
  transitions — the long-range washout LSTMs are known for — and the recurrence is slow. Gap: weak at
  long-range periodic structure, and sequential.
- **Point-wise Transformer reconstructors (Anomaly Transformer lineage, Xu et al. 2021).** Attention
  scores similarity between every pair of *time steps*. But a reconstruction window is mostly normal
  points, so the attention is dominated by them and the rare abnormal pattern — exactly what the score
  must keep sharp — gets washed out. On this very task the plain attention encoder is the *worst*
  reconstruction backbone. Gap: the token is a single step (no standalone meaning), and the dominant
  normal points blur the signal.
- **Frequency-/decomposition-aware models (Autoformer, FEDformer, 2021–2022).** Take periodicity
  seriously — estimate likely period lengths from the autocorrelation/spectrum and aggregate
  same-phase copies. A real acknowledgment that periodicity is the structure to exploit, but they still
  operate on the 1D series, folding the within-cycle shape and the across-cycle relation into one
  mechanism rather than treating them as two axes that can be modeled independently. Gap: intra- and
  inter-period variation are not separated.

## The fixed substrate

The Time-Series-Library anomaly-detection harness is frozen and must not be touched. It supplies: the
dataset loaders (windows of `seq_len=100`, Z-score normalized per dataset), the training loop (Adam,
`lr=1e-4`, MSE reconstruction loss over normal windows), the scoring pipeline (per-point squared
reconstruction error → `anomaly_ratio=1`-percentile threshold over combined train/test scores →
point-adjusted F1/precision/recall), and a library of reusable layers (`layers.Embed.DataEmbedding`
and `PatchEmbedding`, `layers.Conv_Blocks.Inception_Block_V1`, `layers.Autoformer_EncDec.series_decomp`,
`layers.Transformer_EncDec.Encoder/EncoderLayer`, `layers.SelfAttention_Family.FullAttention/AttentionLayer`).
The run scripts pass `d_model=512`, `d_ff=512`, `e_layers=2`, `n_heads=8`, `dropout=0.1`,
`batch_size=32`, `train_epochs=3`, `seq_len=100`, `pred_len=100`, with `enc_in == c_out` equal to the
dataset's channel count.

## The editable interface

Exactly one region is editable — the `Model` class in `models/Custom.py`. The contract: `__init__(self,
configs)` reads `configs.task_name == "anomaly_detection"`, `configs.seq_len`, `configs.enc_in`,
`configs.c_out` (`== enc_in` for reconstruction), and the width/depth knobs above; `anomaly_detection(self,
x_enc)` maps a window `x_enc` of shape `[batch, seq_len, enc_in]` to a reconstruction `[batch, seq_len,
c_out]`; `forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)` dispatches to it. The mark/dec
arguments are unused for anomaly detection. Every method on the ladder is a fill of this same contract —
the architecture inside `anomaly_detection` is the only thing that changes.

The starting point is the scaffold default: **identity reconstruction** (return the input unchanged),
which scores nothing useful and exists only to define the contract.

```python
import torch
import torch.nn as nn


class Model(nn.Module):
    """Default scaffold: identity reconstruction (placeholder)."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len      # for anomaly detection, pred_len = seq_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

    def anomaly_detection(self, x_enc):
        # x_enc: [batch, seq_len, enc_in] -> [batch, seq_len, c_out]
        return x_enc                          # placeholder: identity reconstruction

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        return None
```

## Evaluation settings

Three datasets spanning the difficulty range — **PSM** (server metrics, the easiest, near-saturated
F1), **MSL** (Mars rover telemetry, 55 channels, intermediate), **SMAP** (satellite telemetry, the
hardest, F1 in the high 0.6s) — each at a single seed (42), `seq_len=100`, `anomaly_ratio=1`. Three
metrics per dataset, higher is better: `f_score` (the primary metric — point-adjusted F1),
`precision`, `recall`. The reported number is the per-dataset F1; a backbone is stronger when it lifts
the mean F1 across the three.
