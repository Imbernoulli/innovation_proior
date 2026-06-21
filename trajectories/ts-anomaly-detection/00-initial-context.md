## Research question

Unsupervised anomaly detection on heterogeneous multivariate time series — server metrics (PSM, 25 channels), spacecraft telemetry (MSL, 55 channels), and satellite telemetry (SMAP, 25 channels). Labels are too scarce and expensive to train on, so the model only ever sees overwhelmingly normal data. The operational recipe is **reconstruction**: train a model to reproduce normal windows, then flag the points it fails to reproduce. The anomaly score is the per-point reconstruction MSE; the harness sets a threshold at the per-dataset `anomaly_ratio` percentile of the score distribution and reports point-adjusted F1, precision, and recall. The only design freedom is the reconstruction model `Model` in `models/Custom.py`: window in, same-length reconstruction out. A better model reconstructs normal points tightly and produces a clean spike on anomalies.

## Prior art / Background / Baselines

The first rung reacts to prior reconstruction backbones. Each maps a normal window to its reconstruction, and the reconstruction error becomes the anomaly score.

- **Dense autoencoders / LSTM-AE.** Map the window through a bottleneck or roll an LSTM encoder-decoder to reproduce it.
- **Point-wise Transformer reconstructors (Anomaly Transformer).** Apply self-attention over every pair of individual time steps and reconstruct each point from the rest.
- **Frequency-/decomposition-aware models (Autoformer, FEDformer).** Estimate period lengths from the spectrum or autocorrelation and aggregate same-phase information.

## Fixed substrate / Code framework

The Time-Series-Library anomaly-detection harness is frozen. It supplies dataset loaders (windows of `seq_len=100`, Z-score normalized per dataset), the training loop (Adam, `lr=1e-4`, MSE reconstruction loss over normal windows), the scoring pipeline (per-point squared reconstruction error → `anomaly_ratio=1`-percentile threshold over combined train/test scores → point-adjusted F1/precision/recall), and reusable layers (`layers.Embed.DataEmbedding`, `PatchEmbedding`, `layers.Conv_Blocks.Inception_Block_V1`, `layers.Autoformer_EncDec.series_decomp`, `layers.Transformer_EncDec.Encoder/EncoderLayer`, `layers.SelfAttention_Family.FullAttention/AttentionLayer`).

Run scripts pass `d_model=512`, `d_ff=512`, `e_layers=2`, `n_heads=8`, `dropout=0.1`, `batch_size=32`, `train_epochs=3`, `seq_len=100`, `pred_len=100`, with `enc_in == c_out` equal to the dataset's channel count.

## Editable interface

Only the `Model` class in `models/Custom.py` is editable. The contract: `__init__(self, configs)` reads `configs.task_name == "anomaly_detection"`, `configs.seq_len`, `configs.enc_in`, `configs.c_out` (`== enc_in` for reconstruction), and the width/depth knobs above; `anomaly_detection(self, x_enc)` maps a window `x_enc` of shape `[batch, seq_len, enc_in]` to a reconstruction `[batch, seq_len, c_out]`; `forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)` dispatches to it. The mark/dec arguments are unused for anomaly detection.

The starting point is the scaffold default: **identity reconstruction** (return the input unchanged), which defines the contract but scores nothing useful.

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

Three benchmark datasets — **PSM** (server metrics), **MSL** (spacecraft telemetry, 55 channels), and **SMAP** (satellite telemetry) — each evaluated at a single seed (42), `seq_len=100`, `anomaly_ratio=1`. Three metrics per dataset, higher is better: `f_score` (primary — point-adjusted F1), `precision`, `recall`. The reported number is the per-dataset F1; a stronger backbone lifts the mean F1 across the three.
