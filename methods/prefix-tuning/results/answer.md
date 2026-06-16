# Prefix-Tuning

## Problem

Adapt one frozen pre-trained generation model (GPT-2 for table-to-text, BART for summarization) to
a downstream task by learning only a tiny task-specific increment, keeping accuracy close to full
fine-tuning. Full fine-tuning stores a complete model copy per task; discrete prompting is hard to
optimize and each slot is capped to a real word's embedding; tuning only input embeddings is too
weak.

## Key idea

Prepend |P| "virtual token" positions and make their **per-layer key/value tensors** trainable,
while freezing the entire LM. The prefix sits in the left context of every real token at every layer,
so through ordinary attention it steers both the encoding of the input x and the generation of the
output y — without changing any pre-trained weight.

For z = [P; x; y] (autoregressive) or [P; x; P'; y] (encoder-decoder), P_θ is the flattened
per-layer K/V prefix, equivalently tensors `{(K_l^P, V_l^P)}_{l=1}^n` with shape
`[batch, n_head, |P|, head_dim]` when supplied as a Hugging Face-style cache:

  h_i = P_θ[i, :]            if i ∈ prefix positions,
  h_i = LM_φ(z_i, h_{<i})    otherwise.

Train only the prefix-side parameters θ on the unchanged log-likelihood objective
max Σ_{i∈y} log p_φ(z_i | h_{<i}); φ is frozen.

Design choices:
- **Intervene at all layers, not just embeddings.** Trainable input embeddings alone (embedding-only)
  must propagate through the whole frozen stack — too long a dependency path, underpowered. Making
  the prefix's key/value tensors trainable at every layer gives short dependency paths and more
  capacity.
  Expressiveness order: discrete prompt < embedding-only < prefix-tuning.
- **Prefixing, not infixing.** At the front, the prefix can influence both x's encoding and y's
  generation; placed between x and y it can only affect y. Prefixing dominates.
- **MLP reparametrization for stable optimization.** Optimizing P_θ directly is unstable and LR/init
  sensitive. Instead optimize a smaller P'_θ ∈ R^{|P| × k} (k=512 table-to-text, k=800 summarization)
  and set P_θ[i,:] = MLP_θ(P'_θ[i,:]). After training, materialize P_θ once and drop the MLP/P'_θ —
  store only the per-layer K/V prefix.
- **Prefix length** trades expressiveness vs. overfitting; reported settings use short table-to-text
  prefixes and a longer summarization prefix, while attention over the prefix is parallelized.
- **Initialize from real-word activations** when data are scarce, keeping the prefix near activation
  regions the frozen LM already uses.

A bonus: because the task lives in the context rather than the weights, one batch can carry different
prefixes against the same frozen backbone (per-task / per-user batching). Implemented via the
attention cache (`past_key_values`). AdamW, linear LR schedule; default 10 epochs, batch 5, lr 5e-5.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrefixTuning(nn.Module):
    """Trainable per-layer key/value prefix steering a frozen LM, MLP-reparametrized."""
    def __init__(self, lm_config, prefix_len=10, reparam_dim=512, mlp_hidden_dim=512):
        super().__init__()
        self.prefix_len = prefix_len
        self.n_layer = lm_config.n_layer
        self.n_head = lm_config.n_head
        self.d_model = lm_config.n_embd
        self.head_dim = self.d_model // self.n_head
        self.register_buffer("prefix_positions", torch.arange(prefix_len), persistent=False)
        self.prefix_basis = nn.Embedding(prefix_len, reparam_dim)  # P'_theta, width k
        self.mlp = nn.Sequential(                                   # expand k -> flattened K/V cache
            nn.Linear(reparam_dim, mlp_hidden_dim),
            nn.Tanh(),
            nn.Linear(mlp_hidden_dim, self.n_layer * 2 * self.d_model),  # 2 = key + value
        )

    def materialize(self, batch_size, device):
        idx = self.prefix_positions.to(device)
        h = self.mlp(self.prefix_basis(idx))                       # [prefix_len, n_layer*2*d_model]
        h = h.unsqueeze(0).expand(batch_size, -1, -1)
        h = h.reshape(batch_size, self.prefix_len, self.n_layer * 2, self.n_head, self.head_dim)
        h = h.permute(2, 0, 3, 1, 4)                               # [n_layer*2, B, n_head, len, hd]
        return tuple((h[2 * l], h[2 * l + 1]) for l in range(self.n_layer))   # (key, value) per layer


def loss_fn(frozen_lm, prefix, x_ids, y_ids):
    z = torch.cat([x_ids, y_ids], dim=1)
    past = prefix.materialize(z.size(0), z.device)
    out = frozen_lm(input_ids=z, past_key_values=past)
    logits = out.logits[:, x_ids.size(1) - 1:-1, :]                # score the y positions
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_ids.reshape(-1))


def train(frozen_lm, prefix, loader, opt):
    for p in frozen_lm.parameters():
        p.requires_grad = False
    optim = torch.optim.AdamW(prefix.parameters(), lr=opt.lr)      # only prefix-side params train
    for batch in loader:
        loss = loss_fn(frozen_lm, prefix, batch["x_ids"], batch["y_ids"])
        loss.backward(); optim.step(); optim.zero_grad()

# After training: cache the materialized per-layer key/value tensors and discard the MLP/P'_theta.
```
