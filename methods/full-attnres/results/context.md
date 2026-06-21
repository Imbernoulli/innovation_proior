## Research question

A deep pre-LN decoder carries information upward through a single residual stream. Each
transformation reads the current stream, produces an output, and the architecture decides how
that output and the earlier stream become the input to the next transformation. In the standard
form,

```text
h_l = h_{l-1} + f_{l-1}(h_{l-1}),     h_1 = embedding(token).
```

Unrolling gives `h_l = h_1 + sum_{i=1}^{l-1} f_i(h_i)`: the input at depth `l` is the token
embedding plus every previous transformation output with coefficient one. This recurrence
provides direct gradient routes through depth and fixes the forward aggregation rule. The
question is how to form each block's input from what came before it across depth, with the
hidden width held fixed and within the ordinary pre-LN GPT training harness.

## Background

Residual learning separates a hard optimization problem into an identity path plus a learned
correction. For the recurrence above,

```text
dL/dh_l = (dL/dh_L) * prod_{j=l}^{L-1} (I + df_j/dh_j),
```

under the usual row-vector convention for gradients. Expanding the product exposes a term in
which every Jacobian factor is replaced by `I`, so gradients can reach lower layers without
being forced through a long chain of learned transformations.

In pre-LN Transformers the normalization sits before each sublayer, so sublayer outputs are
added to an unnormalized running stream. Diagnostics of pre-LN models report hidden-state
magnitudes increasing with depth, and layer-pruning studies report that many deep layers can be
removed with little immediate damage.

Several tools for learned mixing exist. Self-attention, expert routing, and residual gates show
different ways to make information flow depend on learned scores or gates. Linear-attention
analyses show how some score forms can be rewritten as recurrent state updates. Normalization
methods such as RMSNorm compare vectors by scale-normalized direction without recentring them.
Structured-matrix language, especially semiseparable rank, gives a way to compare depth
aggregation rules by the effective matrix `M` in `h_l = sum_i M_{i->l} v_i`.

## Baselines

**Standard additive residual.** The update `h_l = h_{l-1} + f_{l-1}(h_{l-1})` makes the
effective valid lower-triangular entries `M_{i->l}` all equal to one. In the semiseparable
view this is a rank-1 all-ones kernel over the valid causal entries. Each layer reads the
single accumulated predecessor state.

**ReZero and LayerScale.** These multiply sublayer outputs by learned scalar or diagonal
coefficients before addition. The coefficients are set during training and fixed afterward, and
each layer is presented with one accumulated predecessor state.

**Highway Networks.** Highway layers use input-dependent gates:

```text
h_l = (1 - g_l) * h_{l-1} + g_l * f_{l-1}(h_{l-1}).
```

With scalar gates, unrolling gives carry products through depth, so the effective kernel is
semiseparable rank 1. The gates change with the input and operate on the compressed predecessor.

**DenseNet.** DenseNet gives each layer direct cross-layer access by concatenating preceding
feature maps and projecting them back to a usable width. Concatenation grows the channel
dimension with depth, and the projection differs from the fixed-width residual-stream operation
used by a decoder-only GPT.

**DenseFormer / depth-weighted averaging.** DenseFormer inserts learned scalar averages over
current and past block representations, supplying cheap cross-layer access with learned
coefficients. The coefficients are static for a trained model: the same source weights are used
for every token and context.

**Hyper-Connections / mHC.** Hyper-Connections widen the residual stream into `m` parallel
streams and update them with learned transitions:

```text
H_l = H_{l-1} A_l + f_{l-1}(H_{l-1} alpha_{l-1}) beta_{l-1}^T.
```

Unrolling yields `M_{i->l} = beta_i^T A^x_{i+1->l} alpha_l`, an `m`-semiseparable depth
kernel. This raises the state rank beyond a single stream and is a recurrence through an
`m`-wide predecessor state.

**Dynamic dense connections / MUDDFormer.** Dynamic dense connections generate cross-layer
weights from the current hidden state, with separate streams for query, key, value, and
residual inputs. This gives content-conditioned depth access using extra MLP-generated weights
and multiple decoupled streams.

**Layer attention / MRLA.** Layer-attention variants gate previous-layer information with
projected query-key products and sigmoid-style gates. They provide input-conditioned
cross-layer paths; the separable score form can be represented as a recurrent state update, and
each source is gated separately.

## Evaluation settings

The natural testbed is a GPT-2-class pre-LN decoder trained on a large web-text corpus with a
standard BPE tokenizer. The model scale of interest is around GPT-2 Medium: 24 blocks, 16
heads, `d_model = 1024`, tied token embedding and output projection, AdamW with decoupled
weight decay, cosine learning-rate decay with warmup, gradient clipping, mixed precision, and
distributed data parallel training.

The primary metric is validation cross-entropy on held-out text. Secondary language-modeling
and downstream metrics include perplexity-style evaluations and common zero-/few-shot
benchmarks. Diagnostics for any depth-flow change include hidden-state norms across depth,
gradient norms across depth, and the effective depth mixing matrix induced by the trained
model. Comparisons keep data, model size, optimizer, and schedule fixed so that the depth-flow
rule is the changed component.

## Code framework

The substrate is a standard pre-LN GPT in PyTorch. Token embeddings, position embeddings,
causal self-attention, the MLP, LayerNorm, the language-modeling loss, AdamW grouping, the
cosine schedule, and the DDP training loop are already present. What is not settled is how
information moves up the residual stream across depth — that rule is what is to be designed.
So the scaffold leaves the per-block residual behavior and the block loop as empty slots, and
exposes hooks where any new depth-flow parameters and their optimizer treatment would attach.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# Fixed, pre-existing components (unchanged):
#   LayerNorm, CausalSelfAttention (-> attn), MLP, GPTConfig, AdamW, cosine LR, DDP loop.


class Block(nn.Module):
    """One transformer block: pre-LN attention sublayer + pre-LN MLP sublayer."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # Standard additive residual within a block (fixed unit weights).
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
        # TODO: any parameters the depth-flow rule we design will need.
        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        x = self.transformer.drop(self.transformer.wte(idx))
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = x + self.transformer.wpe(pos)
        # TODO: the depth-wise information-flow rule we will design.
        #       Run the L blocks and decide how each block's input is formed
        #       from what came before it.  (Default: feed each block straight
        #       into the next.)
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() <  2]
        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
            # TODO: a group for any new depth-flow parameters, if needed.
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
```
