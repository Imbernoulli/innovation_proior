The LSTM gave me the split I was watching for, sharply. In-distribution the recurrent counter is excellent — `abc` membership 0.994, `exact` count 0.998 — but every out-of-distribution number is a collapse: `exact` OOD on $T \in [128, 256]$ is *exactly* 0.0, and `abc`/`length-ood` OOD is 0.524, dead chance for a balanced classifier, giving retention of only 0.530. The cell did not implement an exact integer recurrence that transfers; it memorized a counter calibrated to lengths up to 64, and a per-step increment of $0.99$ or a $\tanh(c_t)$ read-out that was responsive in-range compounds and saturates over twice as many steps. The diagnosis is that the information about a far-away block has to be *carried* step by step, and over 256 steps even a unit-gain cell integrates a long chain of small imperfections. So the fix cannot be a better recurrence: I want to stop carrying the count through time at all and instead let every position look *directly* at every other and compare blocks in one hop.

I propose a **self-attention encoder with sinusoidal positional encodings**. The membership decision — is the `a`-block the same size as the `c`-block — is fundamentally a comparison between two regions of the sequence; a recurrence threads it through a long chain, but attention routes it as a single any-to-any lookup with no step-by-step carry to compound. The primitive is scaled dot-product attention,

$$\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V,$$

where $Q, K, V$ are linear projections of the previous layer, the scores are all pairwise dot products, the softmax is over positions, and the values are mixed accordingly. The $1/\sqrt{d_k}$ is load-bearing, not decoration: a raw dot product of two $d_k$-dimensional unit-scale vectors has variance $d_k$, so for $d_k = \text{hidden\_dim}/4 = 32$ the logits sit around $\pm\sqrt{32}$, the softmax saturates toward one-hot, its Jacobian collapses, and the attention weights stop receiving gradient — dividing by $\sqrt{d_k}$ puts the logit variance back at one and keeps the softmax responsive. I use multiple heads so the encoder can attend to several relations at once — one head can track the $a\!\to\!b$ boundary, another the $b\!\to\!c$ boundary — each a lower-dimensional projection with its own softmax, gathered and mixed by an output projection so the per-head findings compose rather than sit siloed. With $\text{hidden\_dim} = 128$ and 4 heads, $d_k = 32$.

The wall that decides this whole rung is that pure attention is *permutation-equivariant*: $\text{softmax}(QK^{\top})V$ is all dot products and weighted sums over the *set* of positions, with no term that knows which position is which. Shuffle the input rows and the outputs shuffle identically — so the encoder cannot tell $a^n b^n c^n$ from a shuffled bag of the same symbols, fatal for a task whose entire content is order and block length. Order must be injected, and *how* I inject it is the lever the prior-art lineage flags as governing length extrapolation. The lineage-faithful choice for a vanilla Transformer (Vaswani et al. 2017) is the sinusoidal code: for position `pos` and dimension `i`,

$$PE(\text{pos}, 2i) = \sin\!\big(\text{pos}/10000^{2i/d}\big), \qquad PE(\text{pos}, 2i+1) = \cos\!\big(\text{pos}/10000^{2i/d}\big),$$

added to the token embedding at the bottom of the stack. Each dimension pair is a sinusoid at its own frequency, the wavelengths sweeping a geometric range, and the appeal is that a shift by $k$ is a fixed position-independent rotation of each $(\sin, \cos)$ pair, so a head can in principle learn "compare to the position $k$ away" as one operator that works everywhere. Crucially the sinusoid is defined for *any* real position, so unlike a learned per-index table it at least *has* a value at OOD lengths rather than feeding the encoder an untrained noise row.

I should be honest about what sinusoidal-in-*absolute*-index buys at the OOD lengths, because that is precisely where I expect the rung to be limited. The code is a function of the absolute index; in training the model only ever sees indices up to ~65 (64 content tokens plus CLS), so the high-frequency dimensions cycle many times within that range and the encoder learns to read the *combinations* of phase values that occur for $\text{pos} \le 65$. At test length 256, positions 66…256 produce phase combinations — especially in the low-frequency dimensions whose wavelength is comparable to the sequence — that the encoder simply never saw. The rotation-by-$k$ property says relative offsets are *representable*; it does not say the encoder *learned* to read them at absolute indices four times larger than training. So I expect the `exact` OOD count to stay at 0.0 and `abc` OOD near chance — no genuine OOD generalization — but retention to *beat* the LSTM's 0.530, for a specific reason: the LSTM's error is a compounding drift that grows monotonically with length, whereas attention's block comparison does not compound step by step, so even when it cannot read the long absolute positions it degrades more gracefully than a saturating accumulator. That is the testable distinction.

The encoder fits the contract as follows. Embedding `nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)`, scaled by $\sqrt{\text{hidden\_dim}}$ so its amplitude matches the $O(1)$ sinusoids rather than being whispered under them; add the sinusoidal `PE`; then 2 `nn.TransformerEncoderLayer`s with `nhead=4`, feed-forward width $4\times\text{hidden\_dim}$, `dropout=0.0`, `activation="gelu"`, `norm_first=True` (pre-norm residual blocks, the stable ordering for a small stack). Two layers, mirroring the LSTM's depth so the comparison is architecture-vs-architecture, not depth-vs-depth. The padding mask is `src_key_padding_mask = tokens.eq(pad_id)` so attention never mixes padded positions — important because the three environments produce very different padded widths within a batch. Here the pooling rule *flips* from rung one: the encoder is bidirectional, every position including CLS attends to the whole sequence, so the CLS at index 0 is now a legitimate global summary — I pool `h[:, 0]`, not the last position. A final `LayerNorm(hidden_dim)` on the pooled CLS stabilizes the scale into the fixed head. One detail the harness forces: the OOD lengths can exceed the precomputed sinusoidal table, so I precompute `PE` up to a generous `max_len` and, if a batch is longer, concatenate zero rows for the overflow — those positions fall back to no positional signal, part of the standard recipe and, notably, itself a small part of why the very longest positions carry weaker order information. I note it not as a fix but as an honest limitation that feeds the next rung's motivation. As at rung one, only the encoder is editable — no decoder, cross-attention, causal mask, label smoothing, tied embeddings, or warmup schedule, all of which are either irrelevant to a pooled encoder classifier or fixed by the harness.

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
