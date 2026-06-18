The LSTM numbers tell me the split I was watching for, and they tell me it sharply. In-distribution the
recurrent counter is excellent: `abc` membership lands at 0.994 and the `exact` in-distribution count at
0.998 — the cell really did learn to thread a tally through the sequence and compare blocks, exactly the
finite-precision counter recipe predicts. But every out-of-distribution number is a collapse. On `exact`
the OOD accuracy on `T in [128, 256]` is *exactly* 0.0 — not degraded, *zero* — so the `exact` score is
0.0. On `abc`/`length-ood` the OOD accuracy is 0.524, which is chance for a balanced binary classifier:
the model that was right 99.4% of the time in-range is *guessing* the moment the length doubles, and the
extrapolation gap is 0.470, giving a retention of only 0.530. So the LSTM did not implement an exact
integer recurrence that transfers; it memorized a counter calibrated to lengths up to 64. The diagnosis
I sketched at rung one is confirmed by the 0.0 on `exact`: a per-step increment that is 0.99 instead of
1.00, or a `tanh(c_t)` read-out that was responsive at training-range counts, compounds and saturates
over twice as many steps, and the rounded count is then wrong on essentially every long string. The
counter is real but its *scale* is length-specific.

That collapse points at something the recurrent architecture cannot fix by being a better recurrence:
the information about a far-away block has to be *carried* step by step through the cell, and over 256
steps even a unit-gain cell is integrating a long chain of small imperfections. What if I do not carry
the count through time at all, but instead let every position look *directly* at every other position
and compare blocks in one hop? That is the move to self-attention. The membership decision "is the
`a`-block the same size as the `c`-block" is fundamentally a comparison between two regions of the
sequence; a recurrence threads that comparison through a long chain, but attention can route it as a
single any-to-any lookup. The CLS token can attend to the whole string at once and the encoder can, in
principle, compute the block-boundary comparison without the lag that compounded the LSTM's error. So
rung two replaces the recurrent encoder with a self-attention encoder, and the question becomes whether
removing the carry removes the OOD collapse.

Let me build it as the contract allows and watch where it breaks, because attention has its own
length-fragility and I want to know whether I am trading one OOD failure for another. The primitive is
scaled dot-product attention `softmax(QK^T / sqrt(d_k)) V`: queries, keys, values are linear
projections of the previous layer, scores are all pairwise dot products, softmax over positions, mix
the values. The `1/sqrt(d_k)` is not decoration — a raw dot product of two `d_k`-dimensional unit-scale
vectors has variance `d_k`, so for `d_k = hidden_dim/4 = 32` the logits sit around `±sqrt(32)`, the
softmax saturates toward one-hot, its Jacobian collapses, and the attention weights stop receiving
gradient; dividing by `sqrt(d_k)` puts the logit variance back at one and keeps the softmax in its
responsive region. I use multiple heads so the encoder can attend to several relations at once — one
head can track the `a→b` boundary, another the `b→c` boundary — each head a lower-dimensional
projection with its own softmax, gathered and mixed by an output projection so the per-head findings get
composed rather than siloed. With `hidden_dim = 128` and 4 heads, `d_k = 32`.

The wall that decides this whole rung is that pure attention is permutation-equivariant: `softmax(QK^T)V`
is all dot products and weighted sums over the *set* of positions, with no term that knows which
position is which. Shuffle the input rows and the output rows shuffle identically — the encoder cannot
tell `a^n b^n c^n` from a shuffled bag of the same symbols, which is fatal for a task whose entire
content is *order and block length*. So order has to be injected, and *how* I inject it is exactly the
lever the prior-art lineage flagged as governing length extrapolation. The standard, lineage-faithful
choice for a vanilla Transformer is the sinusoidal positional encoding: for position `pos` and dimension
`i`, `PE(pos, 2i) = sin(pos / 10000^{2i/d})`, `PE(pos, 2i+1) = cos(pos / 10000^{2i/d})`, added to the
token embedding at the bottom of the stack. Each dimension pair is a sinusoid at its own frequency, the
wavelengths sweeping a geometric range, and the appeal is that a shift by `k` is a fixed
position-independent rotation of each `(sin, cos)` pair, so a head can in principle learn "compare to the
position `k` away" as one operator that works everywhere. Crucially the sinusoid is defined for *any*
real position, so unlike a learned per-index table it at least *has* a value at the OOD lengths rather
than feeding the encoder an untrained noise row.

But I have to be honest with myself about what sinusoidal-in-absolute-index actually buys at the OOD
lengths, because this is precisely where I expect the rung to be limited, and saying so now makes the
prediction falsifiable against the numbers. The positional code is a function of the *absolute* index.
At training the model only ever sees indices up to ~65 (64 content tokens plus CLS); the
high-frequency dimensions cycle many times within that range, so the *combinations* of phase values the
encoder learns to read are the combinations that occur for `pos ≤ 65`. At test length 256, the absolute
positions 66…256 produce phase combinations — especially in the low-frequency dimensions whose
wavelength is comparable to the sequence — that the encoder simply never saw during training. The
rotation-by-`k` property says relative offsets are *representable*, but it does not say the encoder
*learned* to read them in a way that holds at absolute indices four times larger than training; in
practice the network keys off the absolute phase pattern it was trained on, and that pattern is
out-of-distribution at the long lengths. So I expect the sinusoidal transformer to still fail to
*generalize* the count OOD — the absolute count of `a`'s in `exact` should still be unreachable at
length 256 — but to be *less catastrophic on retention* than the LSTM, for a specific reason: the
LSTM's error is a compounding drift that grows monotonically with length, whereas attention's
block-comparison does not compound step by step, so even if it cannot read the long absolute positions
it degrades more gracefully than a saturating accumulator. That is the testable distinction.

Now the encoder, fit to the contract. Embedding `nn.Embedding(vocab_size, hidden_dim,
padding_idx=pad_id)`, scaled by `sqrt(hidden_dim)` so its amplitude matches the `O(1)` sinusoids rather
than being whispered under them. Add the sinusoidal `PE`. Then 2 `nn.TransformerEncoderLayer`s with
`nhead=4`, feed-forward width `4*hidden_dim` (the position-wise nonlinear capacity), `dropout=0.0`,
`activation="gelu"`, `norm_first=True` (pre-norm residual blocks, the stable ordering for a small
stack). Two layers, mirroring the LSTM's depth so the comparison is architecture-vs-architecture, not
depth-vs-depth. The padding mask is passed as `src_key_padding_mask = tokens.eq(pad_id)` so attention
never mixes in padded positions — important because the three environments produce very different
padded widths within a batch. And here the pooling rule *flips* from rung one: the encoder is
bidirectional, every position including CLS attends to the whole sequence, so the CLS position at index
0 is now a legitimate global summary — I pool `h[:, 0]`, the CLS, not the last position. This is the
attention reading of the same contract, and it is the natural one: CLS pooling is exactly why the
scaffold seeds a CLS token. A final `LayerNorm(hidden_dim)` on the pooled CLS stabilizes the scale into
the fixed head.

One sinusoid-buffer detail the harness forces me to handle: the OOD lengths can exceed the precomputed
positional table, so the table must extend on the fly rather than crash. I precompute `PE` up to a
generous `max_len` and, if a batch is longer, concatenate zero rows for the overflow — those positions
fall back to no positional signal, which is part of the standard recipe and, notably, is itself a small
part of *why* the very longest positions carry weaker order information. I note this not as a fix but as
an honest limitation that feeds the next rung's motivation.

I should again flag what this rung is *not*, per the same-named-baseline warning. This is not the full
encoder-decoder translation Transformer with cross-attention, causal masking, label smoothing, tied
embeddings, and the warmup-then-inverse-sqrt schedule — all of that is either irrelevant to a pooled
*encoder* classifier or lives in the fixed harness (AdamW at a flat 3e-4, BCE/smooth-L1). The editable
slot is the encoder only: scaled embedding, sinusoidal PE, 2 pre-norm encoder layers, CLS pooling,
LayerNorm. So I reason about the encoder's length behavior, not about an optimizer or a decoder I do not
have.

Falsifiable expectations against the LSTM's numbers. In-distribution `abc` I expect to be *at or below*
the LSTM's 0.994 — the recurrent counter is genuinely good in-range and a 2-layer encoder may not match
it on the easy in-distribution regime, so a slightly lower in-distribution `abc` would not surprise me
and would itself be informative. On `exact` OOD I expect the same 0.0 the LSTM hit: nothing in the
sinusoidal absolute code lets the encoder read an exact count at length 256, so the `exact` score should
stay pinned at zero — if it moves off zero I am wrong about the absolute-index limitation. The
prediction that decides whether this rung is an *improvement* is `length-ood` retention: I expect it to
beat the LSTM's 0.530, because attention's block comparison degrades more gracefully than the LSTM's
compounding drift, even though the OOD accuracy itself should stay near chance. If retention rises while
`abc`-in-distribution holds, the rung is a real step up on the metric the task is built to measure, and
the residual failure — no genuine OOD generalization because the positional code is absolute and
out-of-distribution at test length — is exactly the gap the strongest rung must close by changing the
positional representation itself. The full scaffold module is in the answer.
