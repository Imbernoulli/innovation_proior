# ULMFiT: Universal Language Model Fine-tuning for Text Classification

## Method

ULMFiT transfers a whole pretrained language model to a text classifier in three
stages:

1. Pretrain a regular AWD-LSTM language model on a large general corpus
   (WikiText-103 in the paper).
2. Fine-tune that language model on the target task's unlabeled text.
3. Replace the decoder with a concat-pooling classifier head and fine-tune the
   classifier with top-down unfreezing.

The method is designed around three failures of naive transfer: small target
corpora overfit, full classifier fine-tuning causes catastrophic forgetting, and
last-layer-only fine-tuning underfits shallow NLP models.

## Core formulas

**Discriminative fine-tuning.** For layer groups θ^1,...,θ^L with rates
η^1,...,η^L:

```text
θ_t^l = θ_{t-1}^l - η^l * ∇_{θ^l} J(θ)
η^{l-1} = η^l / 2.6
```

The sign is ordinary gradient descent. Lower layers get smaller rates because
they hold more general features.

**Slanted triangular learning rates.** For T updates:

```text
cut = floor(T * cut_frac)
p = t / cut                                      if t < cut
p = 1 - (t - cut) / (cut * (1/cut_frac - 1))    otherwise
η_t = η_max * (1 + p * (ratio - 1)) / ratio
```

Paper defaults: `cut_frac=0.1`, `ratio=32`, `η_max=0.01`. The schedule rises
quickly from `η_max/ratio` to `η_max`, then decays linearly back near the floor.

**Concat pooling.**

```text
h_c = [h_T, maxpool(H), meanpool(H)]
H = {h_1, ..., h_T}
```

**Gradual unfreezing.** Train the classifier/top group first, then unfreeze one
lower group per epoch until all groups train together. This protects lower
language features from the first random-head gradients.

## Configuration

Paper constants: AWD-LSTM with embedding size `400`, hidden activations `1150`,
`3` recurrent layers, BPTT length `70`, batch size `64`; classifier hidden size
`50`; dropout rates `0.4` layer/output, `0.3` recurrent, `0.4` input, `0.05`
embedding, and `0.5` recurrent weight-drop; base learning rates `0.004` for LM
fine-tuning and `0.01` for classifier fine-tuning; Adam reported with
`β1=0.7`, `β2=0.99`.

Reference-code caveats from the fastai v0.7.2 release snapshot:

- The IMDb scripts set `optim.Adam(betas=(0.8, 0.99))`, not the paper's
  reported `β1=0.7`.
- The classifier script implements five discriminative groups as
  `[lr/2.6^4, lr/2.6^3, lr/2.6^2, lr/2.6, lr]`.
- The LM fine-tuning script uses four rates `[lr/6, lr/3, lr, lr/2]`.
- The later fastai1 library default uses `n_hid=1152`; the paper and v0.7.2
  IMDb scripts use `1150`.

## Reference-faithful sketch

```python
import math
import torch
import torch.nn as nn

def stlr(t, T, eta_max=0.01, cut_frac=0.1, ratio=32):
    cut = math.floor(T * cut_frac)
    if cut <= 0:
        return eta_max
    if t < cut:
        p = t / cut
    else:
        p = 1 - (t - cut) / (cut * (1 / cut_frac - 1))
    return eta_max * (1 + p * (ratio - 1)) / ratio

def classifier_group_lrs(lr, factor=2.6):
    return [lr / (factor ** 4), lr / (factor ** 3), lr / (factor ** 2), lr / factor, lr]

def lm_group_lrs_v072(lr):
    return [lr / 6, lr / 3, lr, lr / 2]

def concat_pool(output):
    # fastai v0.7 uses sequence-first tensors: [time, batch, hidden].
    avg_pool = output.mean(dim=0)
    max_pool = output.max(dim=0).values
    return torch.cat([output[-1], max_pool, avg_pool], dim=1)

class LinearBlock(nn.Module):
    def __init__(self, n_in, n_out, p):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_in)
        self.drop = nn.Dropout(p)
        self.lin = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.lin(self.drop(self.bn(x)))

class PoolingLinearClassifier(nn.Module):
    def __init__(self, hidden_size, n_classes, inner=50, drops=(0.4, 0.1)):
        super().__init__()
        self.layers = nn.ModuleList([
            LinearBlock(3 * hidden_size, inner, drops[0]),
            LinearBlock(inner, n_classes, drops[1]),
        ])

    def forward(self, raw_outputs, outputs):
        x = concat_pool(outputs[-1])
        for layer in self.layers:
            logits = layer(x)
            x = torch.relu(logits)
        return logits, raw_outputs, outputs

class MultiBatchEncoder(nn.Module):
    def __init__(self, encoder, bptt=70, max_seq=20 * 70):
        super().__init__()
        self.encoder, self.bptt, self.max_seq = encoder, bptt, max_seq

    def forward(self, tokens):
        # tokens: [time, batch]. The encoder carries hidden state across chunks.
        self.encoder.reset()
        raw_chunks, out_chunks = [], []
        sl = tokens.size(0)
        for i in range(0, sl, self.bptt):
            raw, out = self.encoder(tokens[i:min(i + self.bptt, sl)])
            if i > sl - self.max_seq:
                raw_chunks.append(raw)
                out_chunks.append(out)
        return self._concat(raw_chunks), self._concat(out_chunks)

    @staticmethod
    def _concat(chunks):
        return [torch.cat([chunk[layer] for chunk in chunks], dim=0)
                for layer in range(len(chunks[0]))]

def gradual_unfreezing_training(learner, lrs, final_cycles):
    learner.freeze_to(-1)
    learner.fit(lrs, 1, use_clr=(8, 3))
    learner.freeze_to(-2)
    learner.fit(lrs, 1, use_clr=(8, 3))
    learner.unfreeze()
    learner.fit(lrs, final_cycles, use_clr=(8, 8))
```

Run the same pipeline for forward and backward language models when bidirectional
evidence is wanted, then average the two classifiers' predicted probabilities.
