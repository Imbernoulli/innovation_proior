**Problem.** The LSTM memorized a length-specific counter: 0.994 in-distribution `abc` but chance OOD
(0.524) and `exact` OOD exactly 0.0, because its continuous accumulator drifts and its `tanh` read-out
saturates over twice the training length. Carrying the count step by step is the failure; the next rung
should compare blocks directly instead of threading a tally through time.

**Key idea.** Replace the recurrent encoder with a self-attention encoder. Any-to-any attention routes
the block-size comparison (is `#a == #b == #c`) in one hop, with no step-by-step carry to compound. CLS
pooling gives a global summary; multi-head attention lets different heads track different block
boundaries.

**Why this and the residual limitation.** Pure attention is permutation-equivariant, so order must be
injected — here by sinusoidal positional encodings added to the embedding (the vanilla-Transformer
choice). The `1/sqrt(d_k)` scaling keeps the softmax responsive. But the sinusoid is a function of
*absolute* index: at training the encoder only sees indices up to ~65, so at test length 256 the phase
patterns (especially low-frequency dimensions) are out-of-distribution. Expect `exact` OOD to stay 0.0
and `abc` OOD near chance, but `length-ood` retention to *beat* the LSTM's 0.530 because attention
degrades more gracefully than a compounding accumulator. Closing the OOD gap requires changing the
positional representation — the next rung.

**Why not the full paper recipe.** Only the encoder is editable: scaled embedding → sinusoidal PE →
2 pre-norm encoder layers (4 heads, FFN 4×, GELU) → CLS pool → LayerNorm. No decoder, cross-attention,
causal mask, label smoothing, tied embeddings, or warmup schedule — those are irrelevant to a pooled
encoder classifier or fixed by the harness. Pooling flips from the LSTM: read CLS (`h[:, 0]`), since the
bidirectional encoder makes CLS a legitimate global summary.

**Hyperparameters.** `nhead=4` (`d_k=32`), `num_layers=2`, `dim_feedforward=4*hidden_dim=512`,
`dropout=0.0`, `activation="gelu"`, `norm_first=True`; embedding scaled by `sqrt(hidden_dim)`;
`src_key_padding_mask = tokens.eq(pad_id)`; sinusoidal table extended with zeros past `max_len`.

```python
# EDITABLE region of pytorch-examples/synth_counting/custom_strategy.py — step 2: Transformer (sinusoidal PE)
def build_model(config: TaskConfig) -> nn.Module:
    """Vanilla Transformer encoder with sinusoidal positional encodings."""

    class SinusoidalPE(nn.Module):
        def __init__(self, dim: int, max_len: int = 4096):
            super().__init__()
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            T = x.size(1)
            if T > self.pe.size(1):
                # Extend on the fly so OOD lengths do not crash; values past
                # max_len fall back to zero (still part of the standard recipe).
                extra = torch.zeros(1, T - self.pe.size(1), x.size(-1), device=x.device, dtype=x.dtype)
                pe = torch.cat([self.pe.to(x.dtype), extra], dim=1)
            else:
                pe = self.pe[:, :T].to(x.dtype)
            return x + pe

    class TransformerEncoder(nn.Module):
        def __init__(self, cfg: TaskConfig):
            super().__init__()
            self.cfg = cfg
            self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim, padding_idx=cfg.pad_id)
            self.pe = SinusoidalPE(cfg.hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim,
                nhead=4,
                dim_feedforward=4 * cfg.hidden_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.norm = nn.LayerNorm(cfg.hidden_dim)

        def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            x = self.embed(tokens) * math.sqrt(self.cfg.hidden_dim)
            x = self.pe(x)
            key_padding = tokens.eq(self.cfg.pad_id)
            h = self.encoder(x, src_key_padding_mask=key_padding)
            cls = h[:, 0]
            return self.norm(cls)

    return TransformerEncoder(config)
```
