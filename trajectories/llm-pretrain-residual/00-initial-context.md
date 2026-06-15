## Research question

A GPT-style language model is a stack of identical transformer blocks, each one reading a normalized
copy of a shared residual stream and writing its attention or MLP output back onto it. The single
thing being designed here is **how information flows through that residual stream across layers** — the
rule that turns one block's output into the next block's input. Everything else (the attention, the
MLP, the normalization, the data, the optimizer schedule) is fixed. The default rule is the plain
Pre-LN additive residual, `x = x + sublayer(LN(x))`, repeated twice per block. The goal is to redesign
that depth-flow rule to lower validation loss on FineWeb — to improve gradient flow, feature reuse, and
training stability across the 24-layer stack without touching anything but the residual logic.

## Prior art before the first rung (the residual lineage the first baseline reacts to)

The first rung is the standard Pre-LN residual itself. It is the resolution of a line of
residual-connection designs; these are the ancestors the ladder reacts to, each with the gap it left.

- **Plain stacked layers (no skip).** Compose `L` width-preserving maps directly. Expressive in
  principle — a deep stack can represent anything a shallow one can by idling the extra layers — but a
  per-layer signal/gradient gain `r` compounds to `r^L`, so unless `r ≈ 1` at every layer the forward
  signal and backward gradient vanish or explode exponentially, and a deeper plain net reaches *higher*
  training error than a shallow one. Gap: depth is not trainable.
- **Residual connection (He et al. 2015), `x_{l+1} = σ(x_l + F(x_l))`.** Add an identity skip so the
  block only has to learn a nudge to identity; the additive `1` in the backward product gives gradients
  a clean route through depth. Made hundreds of layers trainable. But the branch fires at full strength
  at init (the block is *not* the identity at step zero), so the stream variance still compounds with
  depth. Gap: tames the worst of `r^L` but does not pin `r = 1`.
- **Post-LN Transformer, `x ← LN(x + sublayer(x))`.** Normalize *after* the addition. This puts a
  LayerNorm Jacobian on the residual highway, so the backward path is a product of normalization
  Jacobians — the multiplicative structure the skip was meant to avoid — giving large, depth-imbalanced
  gradients near the output at init. Gap: needs learning-rate warm-up to survive the first steps.
- **Pre-LN Transformer, `x ← x + sublayer(LN(x))`.** Move the norm *inside* the branch so the highway
  is a clean identity-plus-addition again; the leading `1` is restored, the last-layer gradient shrinks
  like `1/√L`, and one final LayerNorm before the head handles the linearly-growing stream scale. This
  is the modern default and the first rung. Its residual stream is still a *fixed unit-weight
  accumulator*: every layer adds its branch with coefficient exactly one, the same for every token, and
  in deep Pre-LN stacks the stream variance climbs with depth, shrinking the deep blocks' Jacobians
  toward identity so the deepest layers go half-dead. That last gap — a rigid, unweighted depth-flow
  rule — is the seam every later rung pulls on.

## The fixed substrate

A nanoGPT-style GPT-2 Medium training loop is frozen and must not be touched: 24 layers, 16 heads,
`d = 1024` (~355M params), GPT-2 tokenizer, weight tying between `wte` and `lm_head`, learned absolute
position embeddings added once at the bottom, and a single final LayerNorm before the output head. The
loop runs 13,535 iterations on FineWeb (sample-10BT, ~7.1B training tokens) with micro-batch 32,
gradient accumulation 16, 2-GPU DDP, AdamW (`β = (0.9, 0.95)`, weight decay 0.1, grad-clip 1.0), a
cosine LR schedule with linear warm-up over 4% of steps, and bf16 autocast + `torch.compile`. The
fixed components are `CausalSelfAttention`, `MLP`, `LayerNorm`, and `GPTConfig`; the `_init_weights`
scheme (Linear `std = 0.02`, embeddings `std = 0.02`) and the residual-projection rescaling
(`c_proj.weight` init `std = 0.02 / √(2·n_layer)`, one factor per branch write) are also fixed.

## The editable interface

Exactly one module is editable — `nanoGPT/custom_pretrain.py`, and within it only the residual logic:
the `Block` class (per-block residual behavior), `GPT.__init__` (extra residual parameters), the
**block loop in `GPT.forward`** (how blocks are called and how their outputs are accumulated),
`configure_optimizers` (param groups / LR / weight decay for any new parameters), and the
`CONFIG_OVERRIDES` dict (LR / weight-decay overrides). The contract: `Block.forward` must accept `x`
and return a tensor of the same shape; `GPT.forward` must accept `(idx, targets=None)` and return
`(logits, loss)`. Every rung on the ladder is a fill of this same surface; the methods differ only in
what they put in `GPT.__init__`, the forward block loop, and (when they add parameters) the optimizer
groups.

The starting point is the scaffold default: the **vanilla additive Pre-LN residual**, expressed as a
plain loop over the blocks. Each later method replaces the marked regions and nothing else.

```python
# EDITABLE regions of custom_pretrain.py — default fill (vanilla Pre-LN residual)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))      # attention sublayer
        x = x + self.mlp(self.ln_2(x))       # MLP sublayer
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
        # ── Residual stream parameters ──
        # (default: none — vanilla residual x + sublayer(x) is in Block.forward)
        # Add custom residual parameters here if needed.
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        use_pos = getattr(self.transformer.h[0].attn, 'use_pos_emb', True)
        if use_pos:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            x = x + self.transformer.wpe(pos)
        # ── Residual stream: iterate through transformer blocks ──
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        ...  # fused-AdamW construction (fixed)
        return optimizer
```

## Evaluation settings

A single seed, 42. The primary metric is **validation loss** — cross-entropy on held-out FineWeb,
lower is better. Secondary metrics, all measured on the final checkpoint: **WikiText-2** and
**LAMBADA** perplexity (lower is better) for language-modeling quality, and zero-shot downstream
accuracy on **ARC-Easy** and **HellaSwag** (higher is better) via the lm-evaluation-harness. PIQA and
WinoGrande are also run but held out. Every rung trains under the identical fixed loop above and is
judged first on validation loss, then on whether the perplexity and downstream numbers move with it.
