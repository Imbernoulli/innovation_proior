## Research question

GPT-style language-model pretraining at the 355M scale. Everything about the run is frozen — the
24-layer / 16-head / d=1024 GPT-2 Medium model, the FineWeb-10B data (~7.1B tokens, Chinchilla-optimal
D = 20N), the GPT-2 tokenizer, the AdamW optimizer, the cosine schedule, the 13,535-iteration budget,
the two-GPU DDP loop, and every evaluation script — *except one module*. The single thing being designed
is the **self-attention layer**: the `CausalSelfAttention` class. The default is the standard GPT-2
attention — scaled-dot-product causal multi-head softmax over Q/K/V, with order supplied by a separate
table of **learned absolute position embeddings** (`wpe`) added to the token embeddings before the first
block. The question is whether a better attention layer — a better way of injecting order, or of
scaling/normalizing the score, or of shaping the attention distribution itself — lowers validation loss
and perplexity and carries through to downstream accuracy, with **no change anywhere outside this one
module**.

## Prior art before the first rung

The attention layer the first rung edits is the resolution of a line of architectural choices the
baselines react to. These precede the ladder; the fixed substrate below is what the field converged to
before this task starts.

- **Scaled dot-product attention (Vaswani et al., 2017).** The logit is `q_m^T k_n / sqrt(d_k)`; the
  `1/sqrt(d_k)` is there because for q, k with unit-variance entries the raw dot product has variance
  `d_k`, which at `d_k=64` pushes the softmax into saturated, low-gradient regions. It is the substrate
  every rung keeps. Gap: it fixes the *scale* of the logit but says nothing about *position* or about
  the *magnitude drift* of q and k as training proceeds.
- **Learned absolute position embeddings (BERT/GPT).** A trainable vector per index `0..L-1`, added to
  the token embedding before the stack: `x_i <- x_i + wpe[i]`. Simple and the default here. Expanding
  the logit with `q_m = W_q(x_m + p_m)`, `k_n = W_k(x_n + p_n)` gives four terms, three of which carry
  *absolute* `p_m`/`p_n` rather than the offset `m - n` — so the relative structure language actually
  cares about (a verb three words after its subject is the same relation wherever the sentence sits in
  the buffer) is left to be discovered indirectly. Gap: absolute, additive, and capped at the trained
  length `L`; the offset dependence is implicit, never built in.
- **Sinusoidal absolute encoding (Vaswani et al., 2017).** The fixed-form ancestor of the learned
  table: `p_{i,2t}=sin(i/10000^{2t/d})`, `p_{i,2t+1}=cos(...)`, a geometric sweep of wavelengths.
  Carries no length cap, but is still *additive and absolute* — the same cross-term problem. There is a
  clue in it, though: shifting position by `k` rotates each `(sin,cos)` pair by a fixed angle, so
  sinusoids already encode relative shift as a rotation; the additive scheme just never uses it as the
  mechanism. Gap: relative-ness is latent in the construction but not exploited.
- **Pre-LayerNorm and the residual stream (Xiong et al., 2020).** The block here is pre-norm:
  `x <- x + Attn(LN(x))`, `x <- x + MLP(LN(x))`. Pre-norm makes deep stacks trainable, but it does
  **not** normalize the per-head q/k *inside* attention — the entries of q and k can grow in magnitude
  over training, inflating the logit scale past what the fixed `1/sqrt(d_k)` was tuned for, and pushing
  the softmax toward saturation late in training. Gap: the layer norm sits outside attention; the
  q·k product itself is unnormalized and can drift.

## The fixed substrate

A nanoGPT training loop is frozen and must not be touched. GPT-2 Medium (`n_layer=24`, `n_head=16`,
`n_embd=1024`, no bias, dropout 0); weight-tied token embedding and LM head; pre-LayerNorm blocks
`x + Attn(LN(x))` then `x + MLP(LN(x))` with a 4× GELU MLP; AdamW (`lr=6e-4`, `betas=(0.9,0.95)`,
`weight_decay=0.1`, `grad_clip=1.0`), cosine decay to `lr/10` with 4% linear warmup; bfloat16 autocast,
`torch.compile`, micro-batch 64 × grad-accum 8 over 2 GPUs, 13,535 iterations on FineWeb-10B with the
GPT-2 tokenizer. Cross-entropy is the training loss; checkpoints are saved for downstream evaluation.

One detail of the loop is load-bearing for everything that follows: the model adds learned position
embeddings **only if** the attention module asks for them. In `GPT.forward`, position embeddings are
gated by `getattr(self.transformer.h[0].attn, 'use_pos_emb', True)` — if an attention module sets
`self.use_pos_emb = False`, the `wpe` table is skipped entirely and the module is expected to supply its
own position encoding. So the loop already accommodates an attention layer that injects order itself,
not through the additive table.

## The editable interface

Exactly one region is editable: the `CausalSelfAttention` class in `nanoGPT/custom_pretrain.py`
(`__init__` and `forward`), plus a small `CONFIG_OVERRIDES` dict (allowed keys: `learning_rate`,
`weight_decay`, `warmup_iters`, `min_lr`, `grad_clip`). The contract is fixed: the class is constructed
from a `config` (`n_embd`, `n_head`, `block_size`, `bias`, `dropout`), and `forward(self, x)` takes
`x: (B, T, C)` and returns `(B, T, C)`. Whatever happens between — how Q/K/V are projected, how position
enters, how the score is scaled and masked, how the attention distribution is formed — is open, as long
as the output shape and the causal masking are preserved. If the module supplies its own position
encoding, it sets `self.use_pos_emb = False` so the loop skips the learned `wpe`.

The starting point is the scaffold default: standard GPT-2 causal multi-head softmax attention with the
learned `wpe` left on (`use_pos_emb = True`). Each rung replaces exactly this class and nothing else.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py — default fill (standard GPT-2 attention)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        # Set to False if using custom position encoding (e.g. RoPE)
        self.use_pos_emb = True

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
```

## Evaluation settings

One seed (42). The score is a geometric mean of two settings. **Language modeling (`gpt-345m`)**, lower
is better: validation cross-entropy on a held-out FineWeb shard (`val_loss`, weight 2), and word-level
perplexity on WikiText-2 (`wikitext2_ppl`, weight 1) and LAMBADA (`lambada_ppl`, weight 1). **Downstream
(`lm-eval-345m`)**, higher is better, via the LM Evaluation Harness: ARC-Easy, HellaSwag, PIQA, and
WinoGrande accuracy (equal weight). A strong rung lowers `val_loss`/perplexity *and* lifts downstream
accuracy, using only the attention module.
