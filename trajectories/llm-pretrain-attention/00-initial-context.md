## Research question

GPT-2 Medium language-model pretraining at ~355M parameters. The run is fully specified: 24 layers, 16 heads, d=1024, FineWeb-10B data (~7.1B tokens), the GPT-2 tokenizer, AdamW with cosine schedule, 13,535 iterations, and a two-GPU DDP loop. The only open module is the `CausalSelfAttention` class. The default is standard GPT-2 causal scaled-dot-product softmax attention over Q/K/V, with learned absolute position embeddings added to the token embeddings before the first block. The question is whether a different attention layer — a different way to inject order, scale scores, or shape the attention distribution — improves validation loss, perplexity, and downstream accuracy without any change outside this module.

## Prior art / Background / Baselines

- **Scaled dot-product attention.** The attention score is `q_m^T k_n / sqrt(d_k)`. The `1/sqrt(d_k)` rescaling keeps the softmax from saturating when q and k have unit-variance entries. Gap: it fixes logit scale but encodes no positional information and does not control q/k magnitude drift as training proceeds.
- **Learned absolute position embeddings.** A trainable vector per position is added to the token embedding before the stack: `x_i <- x_i + wpe[i]`. Gap: it is absolute, additive, and capped at the trained length; the offset between positions is implicit, not built into the attention score.
- **Sinusoidal absolute encoding.** Fixed sinusoidal functions of position with geometric wavelengths replace the learned table. Gap: it removes the length cap but is still additive and absolute, so relative offsets are not represented directly in the attention score.
- **Pre-LayerNorm residual block.** Layer normalization is placed before each sublayer: `x <- x + Attn(LN(x))`, `x <- x + MLP(LN(x))`. Gap: layer norm sits outside attention; the q/k product itself is unnormalized and can drift in scale during training.

## Fixed substrate / Code framework

A nanoGPT training loop is frozen and must not be touched. GPT-2 Medium (`n_layer=24`, `n_head=16`, `n_embd=1024`, no bias, dropout 0); weight-tied token embedding and LM head; pre-LayerNorm blocks with a 4× GELU MLP; AdamW (`lr=6e-4`, `betas=(0.9,0.95)`, `weight_decay=0.1`, `grad_clip=1.0`); cosine decay to `lr/10` with 4% linear warmup; bfloat16 autocast; `torch.compile`; micro-batch 64 × grad-accum 8 over 2 GPUs; 13,535 iterations on FineWeb-10B with the GPT-2 tokenizer. Cross-entropy is the training loss; checkpoints are saved for downstream evaluation.

One load-bearing detail: position embeddings are added only if the attention module asks for them. In `GPT.forward`, the gate is `getattr(self.transformer.h[0].attn, 'use_pos_emb', True)`. If an attention module sets `self.use_pos_emb = False`, the learned `wpe` table is skipped and the module is expected to supply its own position encoding.

## Editable interface

Exactly one region is editable: the `CausalSelfAttention` class in `nanoGPT/custom_pretrain.py` (`__init__` and `forward`), plus a small `CONFIG_OVERRIDES` dict (allowed keys: `learning_rate`, `weight_decay`, `warmup_iters`, `min_lr`, `grad_clip`). The contract is fixed: the class is constructed from a `config` (`n_embd`, `n_head`, `block_size`, `bias`, `dropout`), and `forward(self, x)` takes `x: (B, T, C)` and returns `(B, T, C)`. Whatever happens between — how Q/K/V are projected, how position enters, how the score is scaled and masked, how the attention distribution is formed — is open, as long as the output shape and the causal masking are preserved. If the module supplies its own position encoding, it sets `self.use_pos_emb = False` so the loop skips the learned `wpe`.

The starting point is the scaffold default: standard GPT-2 causal multi-head softmax attention with the learned `wpe` left on (`use_pos_emb = True`).

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
        # Set to False if the module supplies its own position encoding
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

One seed (42). The score is a geometric mean of two settings. **Language modeling (`gpt-345m`)**, lower is better: validation cross-entropy on a held-out FineWeb shard (`val_loss`, weight 2), and word-level perplexity on WikiText-2 (`wikitext2_ppl`, weight 1) and LAMBADA (`lambada_ppl`, weight 1). **Downstream (`lm-eval-345m`)**, higher is better, via the LM Evaluation Harness: ARC-Easy, HellaSwag, PIQA, and WinoGrande accuracy (equal weight). A strong rung lowers `val_loss`/perplexity and lifts downstream accuracy, using only the attention module.
