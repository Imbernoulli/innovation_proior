# Pre-LN Transformer residual connection

The Pre-Layer-Normalization Transformer updates the residual stream as

```text
x <- x + Sublayer(LayerNorm(x))
```

for each attention or MLP sub-layer, then applies one final `LayerNorm` before the output head. The contrast baseline is Post-LN:

```text
x <- LayerNorm(x + Sublayer(x)).
```

Pre-LN keeps the repeated residual through-path as a clean identity-plus-addition stream while still normalizing each branch input.

## Problem it solves

Post-LN Transformers need learning-rate warm-up when trained from scratch. The warm-up dependence remains with plain SGD, so the issue is not only Adam's early adaptive-step variance. The architecture creates large, imbalanced gradients at initialization, especially near the output layer, and a large initial learning rate can make the first updates unstable.

## Core derivation

A clean residual block has

```text
x_{l+1} = x_l + F(x_l)
x_L = x_l + sum_{i=l}^{L-1} F(x_i)
dE/dx_l = dE/dx_L (1 + d/dx_l sum_{i=l}^{L-1} F(x_i)).
```

The additive `1` sends gradient directly through depth. If the skip is scaled as `lambda_l x_l`, the direct term becomes `prod_i lambda_i`, which explodes or vanishes exponentially depending on whether `lambda` is above or below one. Post-LN puts a nonidentity transformation, `LayerNorm`, after every residual addition, so the backward highway contains a product of layer-normalization Jacobians.

At initialization:

```text
E||LN(v)||^2 = d
E||ReLU(X)||^2 = (1/2) sigma^2 d,  X ~ N(0, sigma^2 I_d)
||J_LN(x)||_2 = O(sqrt(d) / ||x||_2).
```

For Post-LN, each layer normalization resets the stream to radius `sqrt(d)`, and the input to the last normalization in a layer satisfies

```text
E||x^{post,5}_{l,i}||^2 = (3/2)d
```

for all `l > 0`, independent of depth. In the last FFN layer, each gradient entry passes through one `J_LN` whose squared norm is `O(d / ((3/2)d)) = O(1)` and one ReLU coordinate bounded by `O(ln d)`, so

```text
||dL/dW^{2,L}||_F = O(d sqrt(ln d)).
```

The same Post-LN Jacobian appears across layers. Since its norm is approximately `sqrt(2/3)`, gradients toward layer `l` carry the decay factor

```text
(2/3)^{(L-l)/2}.
```

For Pre-LN, the stream is not normalized between layers. Each layer adds between `d/2` and `3d/2` expected squared norm, so

```text
(1 + l/2)d <= E||x^{pre}_{l,i}||^2 <= (1 + 3l/2)d.
```

The final normalization before the head receives an input of norm squared `Theta(Ld)`, so its Jacobian squared norm contributes `O(1/L)`. The last-layer gradient becomes

```text
||dL/dW^{2,L}||_F = O(d sqrt(ln d / L)).
```

Layer-to-layer Pre-LN Jacobians have the form `I + branch Jacobian`, and the branch layer-normalization factors shrink as `O(1/sqrt(j))` with depth, so their eigenvalues stay close to one and gradients are much more balanced across layers.

## Design

- Put `LayerNorm` inside each branch: `x + Sublayer(LayerNorm(x))`.
- Keep the skip path as a pure `+ x`; no scalar, gate, projection, or normalization on the repeated through-path.
- Add one final `LayerNorm` before the head, because the stream's expected squared norm grows linearly with depth.
- Initialize residual output projections with `std = 0.02 / sqrt(2 * n_layer)` to account for two residual branch writes per layer.
- Tie token embedding and output-head weights as in the standard GPT language-model harness.

## Faithful code

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0,
                                      std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
```
