# ULMFiT: Universal Language Model Fine-tuning for Text Classification

## Problem

NLP lacked the vision-style recipe of pretraining a whole model once and
fine-tuning it to any task; word-embedding transfer only seeds the first layer.
Full language-model fine-tuning had failed because it overfits small target sets,
suffers catastrophic forgetting when a classifier is attached, and underfits if
(vision-style) only the last layer is tuned. ULMFiT makes whole-model fine-tuning
robust across diverse text-classification tasks with a single architecture and no
custom engineering.

## Key idea

Use a language model as the universal source task and transfer the *entire*
network in three stages, with fine-tuning techniques that defeat overfitting and
forgetting:

1. **General-domain LM pretraining** — pretrain an AWD-LSTM (3-layer LSTM, heavy
   regularization: DropConnect on recurrent weights, variational dropout, etc.) on
   a large general corpus (WikiText-103). Done once.

2. **Target-task LM fine-tuning** — adapt the LM to the target text using:
   - **Discriminative fine-tuning**: per-layer learning rates. The SGD update
     becomes θ_t^l = θ_{t-1}^l − η^l·∇_{θ^l}J(θ). Tune the last layer's rate η^L,
     then set lower layers η^{l-1} = η^l / 2.6 (general low layers move little;
     task-specific high layers adapt).
   - **Slanted triangular learning rates (STLR)**: short linear increase, long
     linear decay. With cut = ⌊T·cut_frac⌋, p = t/cut for t<cut else
     1 − (t−cut)/(cut·(1/cut_frac − 1)), and η_t = η_max·(1 + p·(ratio−1))/ratio.
     Defaults cut_frac=0.1, ratio=32, η_max=0.01.

3. **Target-task classifier fine-tuning** — attach two new linear blocks (batchnorm
   + dropout, ReLU then softmax; only these are learned from scratch), fed by:
   - **Concat pooling**: h_c = [h_T, maxpool(H), meanpool(H)] over hidden states
     H = {h_1..h_T}, so a class-deciding word anywhere in a long document survives.
   - **Gradual unfreezing**: unfreeze from the top, one layer per epoch (last layer
     first, as it holds the least general knowledge), adding a layer to the thawed
     set each epoch until all layers train — protecting general features through the
     unstable early epochs.
   - **BPT3C**: backprop-through-time over fixed-length chunks of long documents,
     carrying hidden state across chunks.

Optionally pretrain/fine-tune a **backward LM** as well and average the two
classifiers' predictions.

## Configuration

AWD-LSTM: embedding 400, 3 layers, 1150 hidden units/layer, BPTT length 70.
Dropouts: 0.4 (layers), 0.3 (RNN), 0.4 (input embedding), 0.05 (embedding), 0.5
weight-drop (recurrent hidden-to-hidden). Classifier hidden size 50. Adam with
β₁=0.7, β₂=0.99. Batch size 64; base LR 0.004 (LM fine-tune) and 0.01 (classifier
fine-tune). Special tokens for upper-case, elongation, repetition.

## Code

```python
import torch, torch.nn as nn

# ---- discriminative fine-tuning: eta^{l-1} = eta^l / 2.6 ----
def discriminative_lrs(eta_last, n_layers, factor=2.6):
    return [eta_last / (factor ** (n_layers - 1 - l)) for l in range(n_layers)]

def make_optimizer(layer_groups, eta_last):
    lrs = discriminative_lrs(eta_last, len(layer_groups))
    return torch.optim.Adam(
        [{"params": g.parameters(), "lr": lr} for g, lr in zip(layer_groups, lrs)],
        betas=(0.7, 0.99))

# ---- slanted triangular learning rates ----
def stlr(t, T, eta_max=0.01, cut_frac=0.1, ratio=32):
    cut = int(T * cut_frac)
    p = (t / cut) if t < cut else 1 - (t - cut) / (cut * (1/cut_frac - 1))
    return eta_max * (1 + p * (ratio - 1)) / ratio

# ---- concat pooling ----
def concat_pool(H):                                # H: [B, T, hidden]
    return torch.cat([H[:, -1, :], H.max(dim=1).values, H.mean(dim=1)], dim=-1)

class Classifier(nn.Module):
    def __init__(self, encoder, hidden, inner=50, n_classes=2, drop=0.4):
        super().__init__()
        self.encoder = encoder                     # pretrained AWD-LSTM
        self.block1 = nn.Sequential(nn.BatchNorm1d(3*hidden), nn.Dropout(drop),
                                    nn.Linear(3*hidden, inner), nn.ReLU())
        self.block2 = nn.Sequential(nn.BatchNorm1d(inner), nn.Dropout(drop),
                                    nn.Linear(inner, n_classes))
    def forward(self, tokens):
        H = self.encoder(tokens)
        return self.block2(self.block1(concat_pool(H)))   # softmax via cross-entropy

# ---- gradual unfreezing: thaw top-down, one layer per epoch ----
def gradual_unfreeze_schedule(model_layers):
    for k in range(1, len(model_layers) + 1):
        for L in model_layers[:-k]: set_requires_grad(L, False)
        for L in model_layers[-k:]: set_requires_grad(L, True)
        yield model_layers[-k:]                    # train one epoch, then release one more

# ---- BPT3C: chunked BPTT for long documents, state carried across chunks ----
def bpt3c(encoder, document, chunk=70):
    H_all, state = [], None
    for c in document.split(chunk, dim=1):
        H, state = encoder(c, state)
        H_all.append(H)
    return torch.cat(H_all, dim=1)
```

In practice the AWD-LSTM uses DropConnect/variational dropout, discriminative
learning rates via layer groups, the slanted-triangular (one-cycle) schedule, and
`freeze_to`-style gradual unfreezing.
