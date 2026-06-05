# Adapters: Parameter-Efficient Transfer Learning for NLP

## Problem

Adapt one pre-trained Transformer (BERT) to many downstream tasks while adding only a tiny number
of new parameters per task, with near-full-fine-tuning accuracy, and supporting sequential (online)
task arrival. Full fine-tuning matches accuracy but stores a complete model copy per task (100% of
parameters, N× total for N tasks) and shares nothing; tuning only normalization parameters is
compact but too weak.

## Key idea

Freeze the entire pre-trained network w and inject small trainable **bottleneck adapter modules**
v (with |v| ≪ |w|) into each Transformer layer, initialized so the adapted network starts equal to
the pre-trained one. Many tasks then cost ≈ one shared backbone plus a tiny per-task delta, and
because w is frozen, tasks never interfere (no catastrophic forgetting; perfect memory).

The adapter maps a d-dimensional activation through a bottleneck and adds the input:

  Adapter(x) = x + W_up · GeLU(W_down · x + b_down) + b_up,
  W_down ∈ R^{m×d},  W_up ∈ R^{d×m},  m ≪ d.

Parameters per adapter = 2md + d + m. Two design choices make it work:

- **Near-identity initialization.** The internal skip connection plus near-zero projection weights
  (zero-mean truncated Gaussian, std ≈1e-2) make W_up·GeLU(W_down·x) ≈ 0 at start, so Adapter(x) ≈ x.
  The adapted deep network is therefore unperturbed at init and trainable; the correction grows as v
  learns. Too-large init breaks training.
- **Bottleneck width m.** A direct d→d map costs d² parameters; down-projecting to m ≪ d, applying
  GeLU, and up-projecting back costs only 2md+d+m. m is a single knob trading accuracy for
  compactness; in practice the per-task footprint is ≈0.5–8% of the base model.

Placement: two adapters per Transformer layer, each inserted after a sub-layer's projection back to
d (and after dropout) but **before** the residual add and the (post-LN) layer normalization —
output = LayerNorm(x + Adapter(Sublayer(x))). One after multi-head attention, one after the
feed-forward network.

Trainable per task: the adapters, the layer-normalization affine parameters (cheap re-stabilization
of the activation statistics the adapters shift; 2d per LN), and a fresh classification head on the
[CLS] token. Everything else frozen. Optimizer: Adam, linear warmup over the first 10% of steps then
linear decay, batch size 32.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    """Bottleneck with an internal skip connection; near-identity at init."""
    def __init__(self, d_model, m, init_std=1e-2):
        super().__init__()
        self.down = nn.Linear(d_model, m)     # d -> m
        self.up = nn.Linear(m, d_model)        # m -> d
        for lin in (self.down, self.up):
            nn.init.trunc_normal_(lin.weight, std=init_std, a=-2 * init_std, b=2 * init_std)
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))   # x + correction (~0 at init)


def sublayer_output(sublayer_fn, x, layer_norm, adapter, dropout_p=0.1):
    h = sublayer_fn(x)                  # attention or FFN, projected back to d
    h = F.dropout(h, p=dropout_p)
    h = adapter(h)                       # adapter before residual + LayerNorm
    return layer_norm(h + x)


class AdaptedTransformerLayer(nn.Module):
    def __init__(self, layer, d_model, m):
        super().__init__()
        self.layer = layer               # frozen pre-trained sub-layers + LayerNorms
        self.attn_adapter = Adapter(d_model, m)
        self.ffn_adapter = Adapter(d_model, m)

    def forward(self, x):
        a = sublayer_output(self.layer.attention, x, self.layer.ln1, self.attn_adapter)
        h = sublayer_output(self.layer.feed_forward, a, self.layer.ln2, self.ffn_adapter)
        return h


def trainable_parameters(model):
    """Freeze pre-trained weights; train adapters, LayerNorm affines, and the head."""
    for p in model.parameters():
        p.requires_grad = False
    for module in model.modules():
        if isinstance(module, (Adapter, nn.LayerNorm)):
            for p in module.parameters():
                p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def train(model, loader, opt):
    optim = torch.optim.Adam(trainable_parameters(model), lr=opt.lr)  # warmup 10%, linear decay
    for batch in loader:
        logits = model(batch["input_ids"])
        loss = F.cross_entropy(logits, batch["labels"])
        loss.backward(); optim.step(); optim.zero_grad()
```
