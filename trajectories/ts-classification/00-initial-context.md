## Research question

Can one classification component — a temporal encoder, a way of relating the channels, and a padding-aware way of collapsing the time axis into a fixed-width feature — generalize across heterogeneous multivariate time series when training and evaluation are pinned to a fixed protocol? The three datasets deliberately stress different structure: **EthanolConcentration** is spectral chemistry (4 classes, long smooth absorbance traces), **FaceDetection** is MEG brain imaging (binary, ~144 noisy channels, the signal lives in cross-channel covariation), and **Handwriting** is tri-axial accelerometer character recognition (26 classes). The only designed piece is the `Model` that maps a padded window plus its padding mask to class logits. Everything around it — the data loader, the per-dataset padding length, the optimizer, the loss, and early stopping — is frozen.

## Prior art / Background / Baselines

Current sequence-classification baselines each make a different representational choice, and each leaves a concrete gap.

- **Distance-based classifiers (DTW-1NN; Bagnall et al. 2018).** They align a test series to every training series with dynamic time warping and predict the nearest neighbor's label. The result is lazy, stores the whole training set, scales poorly at test time, and warps channels independently, so there is no shared learned representation and no cross-channel modeling.

- **Recurrent encoders (LSTM/GRU classifiers).** They carry a hidden state through time and read the final state into a classifier. Long-range cues must survive every recurrent transition, which is fragile, and the sequential scan makes training slow.

- **Temporal convolution (TCN / InceptionTime; Fawaz et al. 2020).** They slide 1-D kernels along time and widen the receptive field with dilation or parallel kernel sizes. A finite kernel only relates points inside a local time window, so structure between points farther apart than the receptive field is hard to reach.

- **Per-step Transformer encoders.** They embed the full channel vector at each timestep as one token and run self-attention over the sequence. A raw timestep vector is not a semantically coherent unit, and attending over all timesteps costs quadratically in sequence length.

## Fixed substrate / Code framework

The Time-Series-Library classification harness is frozen and must not be touched. It provides: the UEA loader that pads every batch to the dataset's maximum length `seq_len = max(train.max_seq_len, test.max_seq_len)` and passes a binary mask `x_mark_enc[b, t] ∈ {0, 1}` (1 for a real timestep, 0 for right-padding); dynamic config injection that sets `configs.seq_len`, `configs.enc_in`, `configs.num_class`, and `configs.pred_len = 0` before the model is built; RAdam optimization with `CrossEntropyLoss`; early stopping with `patience = 10`; and per-dataset accuracy reporting. The fixed run flags are `d_model 128`, `d_ff 256`, `e_layers 3`, `n_heads 16`, `dropout 0.1`, `batch_size 16`, `learning_rate 1e-3`, `train_epochs 100` (early-stopped at patience 10).

## Editable interface

Exactly one file is editable: `Time-Series-Library/models/Custom.py`. It must define a `Model(nn.Module)` whose `__init__(self, configs)` reads `configs.seq_len`, `configs.enc_in`, and `configs.num_class`, and whose `classification(self, x_enc, x_mark_enc)` maps `x_enc` of shape `[batch, seq_len, enc_in]` and mask `x_mark_enc` of shape `[batch, seq_len]` to `[batch, num_class]` pre-softmax logits, dispatched from `forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)`. The starting point is a no-op scaffold that returns zero logits for every input.

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
        # no parameters yet — the encoder + head are added here

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

Three UEA datasets — **EthanolConcentration** (4 classes, spectral), **FaceDetection** (2 classes, MEG, many channels), **Handwriting** (26 classes, accelerometer) — under the fixed protocol above, seed 42. One metric, **test accuracy, higher is better**, reported per dataset. The padding length, train/test split, optimizer, loss, and early-stopping budget are identical for every run; only `models/Custom.py` changes.
