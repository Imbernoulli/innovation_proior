# Full Attention Residuals (Full AttnRes)

Full Attention Residuals replaces the fixed additive residual accumulator with softmax
attention over earlier depth sources. For destination layer `l`,

```text
h_l = sum_{i=0}^{l-1} alpha_{i->l} v_i,
v_0 = h_1,          v_i = f_i(h_i) for i >= 1,
alpha_{i->l} = exp(w_l^T RMSNorm(v_i)) / sum_{j=0}^{l-1} exp(w_l^T RMSNorm(v_j)).
```

The learned pseudo-query `w_l in R^d` is one zero-initialized vector per destination layer.
Keys and values are the same earlier outputs; RMSNorm is applied only to the keys used for
scoring, while the weighted sum mixes the raw values.

## Main Points

- The standard recurrence `h_l = h_{l-1} + f_{l-1}(h_{l-1})` has the backward identity
  `dL/dh_l = dL/dh_L * prod_{j=l}^{L-1}(I + df_j/dh_j)`, which explains the residual
  gradient route, but its forward aggregation is a fixed unit-weight sum.
- In the depth mixing form `h_l = sum_i M_{i->l} v_i`, the valid causal entries of the plain
  residual form a rank-1 all-ones semiseparable kernel; Highway gates remain rank 1 through
  scalar carry products; Hyper-Connections produce an `m`-semiseparable kernel
  `M_{i->l} = beta_i^T A^x_{i+1->l} alpha_l`, equivalent to linear attention over depth.
- Full AttnRes uses the non-separable softmax kernel `exp(w_l^T RMSNorm(v_i))`, so the causal
  entries of `M` are dense and the `L x L` depth-mixing matrix is generically rank `L`.
- Zero initialization is exact: if `w_l = 0`, then every score is `exp(0) = 1`, so the source
  softmax is uniform. The model starts from an equal-weight average rather than an arbitrary
  random depth preference.
- Per token, the full method costs `O(L^2 d)` compute and `O(Ld)` source memory. In ordinary
  training the source memory overlaps with activations already retained for backpropagation.
- Query parameters get their own optimizer group with `0.1x` learning rate and no weight
  decay.

## Code

```python
import torch
import torch.nn as nn
from torch.nn import functional as F


class FullAttentionResiduals(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(2 * config.n_layer, config.n_embd))
        self.query_out = nn.Parameter(torch.zeros(config.n_embd))

    def attend(self, sources, query):
        stacked = torch.stack(sources, dim=0)                         # (S, B, T, D)
        keys = F.rms_norm(stacked, (stacked.size(-1),))               # RMSNorm keys
        logits = torch.einsum('d, s b t d -> s b t', query, keys)
        weights = logits.softmax(dim=0)
        return torch.einsum('s b t, s b t d -> b t d', weights, stacked)

    def start(self, x):
        return [x]                                                    # embedding is source 0

    def before_sublayer(self, sources, q_index):
        return self.attend(sources, self.queries[q_index])

    def after_sublayer(self, sources, output):
        sources.append(output)
        return sources

    def readout(self, sources):
        return self.attend(sources, self.query_out)

    def query_parameters(self):
        return [self.queries, self.query_out]

    def extra_parameter_ids(self):
        return {id(p) for p in self.query_parameters()}

    def optimizer_groups(self, learning_rate):
        return [{'params': self.query_parameters(), 'lr': learning_rate * 0.1, 'weight_decay': 0.0}]


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
        self.depth_flow = FullAttentionResiduals(config)
        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        x = self.transformer.drop(self.transformer.wte(idx))
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = x + self.transformer.wpe(pos)

        sources = self.depth_flow.start(x)
        q = 0
        for block in self.transformer.h:
            h = self.depth_flow.before_sublayer(sources, q)
            q += 1
            sources = self.depth_flow.after_sublayer(sources, block.attn(block.ln_1(h)))

            h = self.depth_flow.before_sublayer(sources, q)
            q += 1
            sources = self.depth_flow.after_sublayer(sources, block.mlp(block.ln_2(h)))

        x = self.depth_flow.readout(sources)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        query_ids = self.depth_flow.extra_parameter_ids()
        pd = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay = [p for n, p in pd.items() if p.dim() >= 2 and id(p) not in query_ids]
        nodecay = [p for n, p in pd.items() if p.dim() < 2 and id(p) not in query_ids]
        optim_groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': nodecay, 'weight_decay': 0.0},
        ]
        optim_groups.extend(self.depth_flow.optimizer_groups(learning_rate))
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
```
