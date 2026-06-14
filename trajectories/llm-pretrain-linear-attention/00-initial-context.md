## Research question

Pretrain a GPT-style language model whose sequence mixer is **linear or otherwise subquadratic** in
sequence length, yet stays competitive in language-model quality with standard quadratic softmax
attention. Softmax attention forms an `L×L` score matrix, so both compute and memory grow like `L²`,
and at inference the key/value cache grows with the context already generated. The single thing being
designed is the **attention sublayer** — the `CausalSelfAttention` class, and the `Block` structure
that wraps it — replaced by a mechanism with a fixed-size state and a subquadratic training path.
Everything else about the model and the training pipeline is fixed. Cheapness is easy; cheapness that
does not lose quality against a strong softmax Transformer on the same data is the whole problem.

## Prior art before the first rung (the subquadratic-mixer lineage)

The first rung reacts to a line of efficient-attention work, each member of which buys one property and
gives up another. These are the ancestors the ladder climbs out of.

- **Linear attention (Katharopoulos et al. 2020).** Replace the softmax kernel `exp(qₜ·kᵢ)` with a
  plain feature-map dot product `φ(qₜ)·φ(kᵢ)`, so `φ(qₜ)` factors out of the causal sum and the layer
  becomes a linear RNN with a matrix-valued state `Sₜ = S_{t−1} + kₜᵀvₜ`, read by `oₜ = qₜ Sₜ`. Gives
  `O(1)`-per-step inference with no growing cache. **Gap:** the additive write never forgets — the
  state only accumulates outer products — so old content dilutes the present, and it loses to softmax
  on language modeling, badly on recall.
- **State-space models (S4, Gu et al. 2021).** A linear recurrence `sₜ = A s_{t−1} + B xₜ`, `oₜ = C sₜ`
  whose unrolled form is a convolution computable by FFT — parallel to train, strong at long range.
  **Gap:** `A,B,C` are input-independent, so it does not do the content-based comparison attention
  does; it cannot decide from the token what to keep.
- **Fixed-decay linear attention / "ALiBi-in-the-recurrence."** Add a single global scalar decay,
  `Sₜ = γ S_{t−1} + kₜᵀvₜ` — a recency bias that clearly helps over no decay and keeps the parallel
  matmul form intact (a scalar pulls out of the cumulative product). **Gap:** one fixed forgetting
  rate for every token, channel, and context — a data-independent gate, exactly the thing 1-D RNN
  experience says a forget gate must *not* be.
- **Gated RNNs / the forget gate (LSTM; Gers et al. 2000).** The lesson the whole ladder leans on: the
  single most important component of a gated cell is the multiplicative forget gate `fₜ ⊙ c_{t−1}`, and
  it must be **data-dependent** to do its job. Plain linear attention is precisely a gateless RNN.
  **Gap:** a generic RNN forget gate depends on the previous *state*, which serializes training and
  kills the parallel form.

## The fixed substrate

A nanoGPT pretraining loop is frozen and must not be touched. **Model:** GPT-2 Medium — 24 layers, 16
heads, `n_embd = 1024`, ~355M parameters, tied input/output embeddings, pre-norm blocks, a 4·d GELU
MLP, weight-tied LM head. **Data:** FineWeb `sample-10BT`, GPT-2 tokenizer, ~7.1B training tokens,
block size 1024. **Optimization:** AdamW (β = 0.9/0.95, weight decay 0.1), cosine LR schedule with 4%
warmup, peak LR 6e-4, gradient clip 1.0, bf16, 13,535 iterations, micro-batch 32, gradient
accumulation 16, 2-GPU DDP. The data loader, tokenizer, schedule, evaluation code, and checkpointing
are all out of scope.

Two facts about the substrate are load-bearing for every rung. First, the model **conditionally** adds
learned absolute position embeddings: in `GPT.forward` it reads `self.transformer.h[0].attn.use_pos_emb`
and only adds `wpe` if that flag is `True`. Any mixer whose own decay/rotation already carries relative
position sets `self.use_pos_emb = False` in `__init__` and the loop skips `wpe`. Second, **`torch.compile`
is disabled** for this task because the FLA Triton kernels are not compatible with it. The
`flash-linear-attention` (FLA) library is pre-installed and exposes 27+ optimized linear-attention
layers with hardware-efficient chunkwise Triton kernels (`fla.layers.GatedLinearAttention`, `DeltaNet`,
`MultiScaleRetention`, `Mamba2`, `RWKV6Attention`, `GatedDeltaNet`, …); a rung may import one of these
or implement its own mechanism from scratch.

## The editable interface

Exactly two regions of `nanoGPT/custom_pretrain.py` are editable: the **`CausalSelfAttention` class**
(lines 33–70 — the mixer itself: Q/K/V projections, feature maps, gating, decay, recurrent state) and
the **`Block` class** (lines 88–100 — how the attention and MLP sublayers compose, norm placement,
residuals). Every rung on the ladder is a fill of these two regions and nothing else. The contract is
fixed by the rest of the loop: `CausalSelfAttention.forward(x)` takes `x: (B, T, n_embd)` and returns a
tensor of the same shape; `__init__(config)` receives a `GPTConfig` with `n_embd`, `n_head`, `dropout`,
`bias`, `block_size`. A mixer that does not use `wpe` sets `self.use_pos_emb = False`.

The starting point is the scaffold default: **standard multi-head softmax attention** (FlashAttention
via `scaled_dot_product_attention`, learned position embeddings on). Each rung replaces exactly these
two definitions.

```python
# EDITABLE region 1 of nanoGPT/custom_pretrain.py (lines 33-70) — default: softmax attention
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
        self.use_pos_emb = True                      # softmax attention uses learned absolute positions

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


# EDITABLE region 2 of nanoGPT/custom_pretrain.py (lines 88-100) — default: pre-norm block
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
```

## Evaluation settings

One seed (42), GPT-2 Medium trained on ~7.1B FineWeb tokens. Metrics: **validation loss** —
cross-entropy on held-out FineWeb (lower is better, the primary signal); **perplexity** on WikiText-2
and LAMBADA (lower is better); and zero-shot **downstream accuracy** on ARC-Easy, HellaSwag, PIQA, and
WinoGrande (higher is better) via the LM evaluation harness. Wall-clock (`elapsed`) is recorded but is
not a ranked metric. The bar is parity with — ideally improvement over — what a softmax Transformer of
the same size and token budget would reach; the comparison is between mixers, at matched parameter
count and matched training.
