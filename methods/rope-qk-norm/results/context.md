# Context: self-attention for GPT-style pretraining (circa 2021-2024)

## Research question

The single layer where a Transformer language model decides *which* token attends to *which*
is the self-attention block, and two persistent defects of the standard GPT-2 form of that
block are on the table at once.

First, the standard block is **position-blind by construction** and patches that in with a
*learned absolute* position embedding. Self-attention is permutation-equivariant: with no
position signal, reordering the tokens just reorders the outputs, so order has to be injected
by hand. GPT-2 does it by adding a trainable vector `wpe[i]` to the token embedding at
position `i`. This is absolute (it encodes *where in the buffer* a token sits, not *how far
apart* two tokens are), additive (it rides into the attention logit through cross terms that
mix absolute positions), and hard-capped at the trained context length (nothing is learned for
positions beyond it). A language model mostly cares about *relative* offset — a verb three
words after its subject is the same relation whether the clause starts at token 5 or token
500 — so an absolute, additive, length-capped scheme is working against the grain.

Second, the dot-product score that feeds the softmax is **unbounded, and it drifts upward over
training**. The score `q·k = ||q|| ||k|| cos(angle)` has no ceiling, the softmax sees only the
*differences* between scores, and at large magnitudes a small relative lead becomes a
near-one-hot row. The `1/sqrt(d_k)` factor calibrates the *expected* score only at
initialization; it is not a bound, and as the projection matrices move during training the
query/key norms grow and the scores climb back into the saturated regime.

The goal at this layer: inject position so that the attention logit depends on the *relative*
offset (and generalizes past the trained length), and at the same time keep the logit
magnitude under control for the *whole* run, so the attention distribution can stay expressive
and training stays stable — all without touching anything outside the attention module
(optimizer, schedule, data, loss, and the rest of the stack are fixed).

## Background

The field state. Scaled-dot-product multi-head attention (Vaswani et al. 2017) is the
substrate: `softmax(Q K^T / sqrt(d_k)) V`, with `Q,K,V` linear projections of the input, split
into heads, the score contracted over the per-head dimension `d_k = d / n_head`. The
`1/sqrt(d_k)` factor has a precise rationale: if the components of `q` and `k` are independent,
mean zero, variance one, then `q·k = sum_i q_i k_i` has mean 0 and variance `d_k`, so the
typical score scales like `sqrt(d_k)`; dividing by `sqrt(d_k)` renormalizes the score variance
to ~1 at initialization. This is a *one-time calibration under init-time independence
assumptions*, not a property maintained through training.

Two background concepts the method rests on.

*Normalization that controls magnitude.* LayerNorm (Ba et al. 2016) maps a feature vector `a`
to `(a − mu)/sigma · g + b`: it re-centers (subtract the mean) and re-scales (divide by the
standard deviation), with a learned gain `g` and bias `b`. RMSNorm (Zhang & Sennrich 2019)
keeps only the re-scaling: `RMSNorm(a) = a / RMS(a) · g`, with `RMS(a) = sqrt((1/n) sum_i
a_i^2)` and a learned gain `g`. Their hypothesis, borne out empirically, is that LayerNorm's
*re-centering* invariance is dispensable — mean-subtraction does not reduce the variance of
the hidden states or gradients — while the *re-scaling* invariance is what stabilizes
magnitudes; dropping the mean makes RMSNorm cheaper (reported ~7-64% faster) and a strict drop
in. A vector passed through RMSNorm has each component at unit root-mean-square (times the
gain), so its Euclidean norm is pinned to `sqrt(n) · (gain RMS)` regardless of the input scale.

*Relative position as rotation, latent in sinusoids.* The sinusoidal absolute encoding
`p_{i,2t} = sin(i / 10000^{2t/d})`, `p_{i,2t+1} = cos(i / 10000^{2t/d})` stacks sinusoids whose
wavelengths sweep geometrically from `2*pi` to `~10000·2*pi`. A latent fact: shifting position
`i -> i + k` maps each `(sin(wi), cos(wi))` pair to `(sin(w(i+k)), cos(w(i+k)))`, which is a
*rotation by a fixed angle `wk`* of that pair, independent of `i`. So sinusoids already carry
relative shift as a rotation — but the additive scheme never uses it as the mechanism.

The motivating diagnostic findings (about the *existing* attention block, knowable before any
fix). When Transformers are scaled up, training loss is observed to *diverge after a few
thousand steps* — documented at ~8B parameters for a vision Transformer (Dehghani et al. 2023)
and connected to earlier reports — and the cause is traced to *extremely large values in the
attention logits*, which drive the attention weights to be almost one-hot with near-zero
entropy (Zhai et al. 2023 named this "attention entropy collapse"). The same instability is
reproducible at *small* scale by training at high learning rate (Wortsman et al. 2023): the
"max attention logit", typically largest in the first layer, grows without bound, the
learning-rate-vs-loss curve shows an explosion past a critical LR, and the run diverges. These
are facts about the baseline mechanism, not about any proposed fix.

## Baselines

**Scaled-dot-product attention with `1/sqrt(d_k)` (Vaswani et al. 2017).** The substrate above.
*Gap:* the `1/sqrt(d_k)` factor sets the score scale only at initialization; it provides no
bound on the score during training, so once the projection norms grow the logits can re-enter
the saturated, near-one-hot regime, and at scale this manifests as divergent training loss.

**Learned absolute position embeddings (GPT-2 `wpe`; BERT).** A trainable vector per position
up to a maximum length `L`, added to the token embedding before projection. *Gap:* expanding
`(x_m + p_m)^T W_q^T W_k (x_n + p_n)` produces cross terms `x_m^T W_q^T W_k p_n`, `p_m^T W_q^T
W_k x_n`, `p_m^T W_q^T W_k p_n` that depend on the *absolute* positions `m`, `n` separately,
not on the offset `m − n`; and the table caps the context at `L` with nothing learned past it.

**Sinusoidal absolute encoding (Vaswani et al. 2017).** The fixed closed form above; not
length-capped, but *gap:* still absolute and additive, with the same cross-term contamination —
the relative structure is latent (as rotation) but unused.

**Relative position families (Shaw et al. 2018; Transformer-XL, Dai et al. 2019; T5, Raffel et
al. 2020; DeBERTa, He et al. 2020).** These make attention explicitly relative by editing the
expanded additive dot product: Shaw injects a learned, *clipped* relative embedding into the
key (and value); Transformer-XL replaces the absolute term with a sinusoidal relative one plus
trainable global vectors and a split key projection; T5 collapses everything to a single
learned scalar bias `b_{m,n}` bucketed by distance; DeBERTa keeps the two content×position
cross terms. *Gaps:* they parameterize the relative signal with learned tables or biases (extra
parameters, clipping discards long-range distinctions), and the position signal lives *inside*
the `N×N` logit matrix as an additive term rather than as a per-token transform of `q` and `k`.

**Cosine-similarity attention / query-key normalization (Henry et al. 2020).** A *different*
way to bound the score: L2-normalize `q` and `k` to unit length so each entry is a cosine in
`[−1, 1]`, then *replace* `1/sqrt(d_k)` with a single learnable scalar temperature `g`
(initialized from the typical sequence length, `g_0 = log2(L^2 − L)`) to stretch the bounded
cosines back out before the softmax. Motivated by softmax saturation hurting diffuse attention
on low-resource translation. *Gap (for the present setting):* it discards the magnitude
entirely and hands the scale to a single learned temperature replacing `1/sqrt(d_k)` — a heavier
change to the scoring than is needed if the only requirement is to *keep the magnitude from
running away* while leaving the well-understood `1/sqrt(d_k)` calibration in place.

**Magnitude-stabilizing alternatives at scale (σReparam, Zhai et al. 2023; z-loss; gradient
clipping).** σReparam reparameterizes weights by their spectral norm to fight attention entropy
collapse; z-loss penalizes the output-logit log-partition; gradient clipping caps update size.
*Gaps:* these act on the weights, the output head, or the optimizer step — not directly on the
per-head query/key magnitudes that set the attention logit scale, and (as reported) clipping
and optimizer changes can leave the attention-logit-growth loss spikes in place.

## Evaluation settings

The fixed pretraining yardstick for this attention layer:

- **Model:** GPT-2 Medium — 24 layers, 16 heads, `d = 1024`, `d_k = 64`, ~355M parameters.
- **Data:** FineWeb 10B sample (Penedo et al. 2024), GPT-2 byte-pair tokenizer, ~7.1B training
  tokens (`D = 20 N`, Chinchilla-optimal for 355M).
- **Training:** 13,535 iterations, micro-batch 64, gradient accumulation 8, 2-GPU DDP. The
  optimizer, learning-rate schedule, dataset, tokenizer, training loop, and evaluation scripts
  are fixed; only the attention module is editable.
- **Metrics:** held-out FineWeb cross-entropy (validation loss) and perplexity; WikiText-2 and
  LAMBADA perplexity; downstream accuracy on ARC-Easy, HellaSwag, PIQA, WinoGrande via the LM
  Evaluation Harness.
- **Stability diagnostics** (for the motivating findings): the *max attention logit* (per
  layer, largest in early layers), attention entropy, and the learning-rate-vs-loss curve / LR
  sensitivity summary statistic; divergence-after-a-few-thousand-steps as the failure to avoid.

## Code framework

The default `CausalSelfAttention` already exists in the nanoGPT training harness, and the
surrounding stack — token + (currently learned absolute) position embeddings, the residual
Transformer blocks, the optimizer, the loss, the training loop — is fixed. The only editable
region is how the per-head queries and keys are formed and scored inside this one module, and
whether the module supplies its own position signal. The substrate provides: the packed QKV
projection, the head split/merge, the causal-masked softmax (with a flash
`scaled_dot_product_attention` fast path), and a flag the module can flip to tell the model to
*stop* adding the learned position embedding when the attention block carries position itself.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention. The QKV/output projections, the head
    split/merge, and the causal-masked softmax already exist. Two slots are open:
    (a) whether/how this module injects position, and (b) what transform (if any)
    is applied to the per-head queries and keys before the dot product."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size),
            )
        # If this module supplies its own position signal, flip this to False and
        # the model will skip adding the learned absolute position embedding (wpe).
        self.use_pos_emb = True
        # TODO: any per-position / per-head state the scoring design we choose will need.

    def _prepare_qk(self, q, k, seq_len):
        # q, k: [B, n_head, T, head_dim]
        # TODO: the transform applied to the per-head queries/keys before scoring,
        #       and the position signal (if this module carries one).
        return q, k

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q, k = self._prepare_qk(q, k, T)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
```

The two open slots are `use_pos_emb` (does this module carry position) and `_prepare_qk` (what,
if anything, transforms the per-head `q`/`k` before the dot product). Everything else — the
projections, the split/merge, the masked softmax, the `1/sqrt(d_k)` scaling on the manual path —
is the existing substrate the design fills into.
