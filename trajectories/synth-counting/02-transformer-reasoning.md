The LSTM numbers give the split I was watching for, sharply. In-distribution the recurrent counter is
excellent — `abc` membership 0.994, `exact` in-distribution count 0.998 — so the cell really did learn to
thread a tally and compare blocks. But every out-of-distribution number collapses. On `exact` the OOD
accuracy on `T ∈ [128,256]` is *exactly* 0.0, so the `exact` score is 0.0. On `abc`/`length-ood` the OOD
accuracy is 0.524, and I read that precisely: the split is balanced, so chance is 0.5, and with 1,024
eval examples the sampling std of an accuracy estimate is `sqrt(0.25/1024) ≈ 0.0156`; 0.524 is ~1.5 std
above 0.5, inside the noise band of a genuinely chance-level classifier. No retained signal. The
retention number is not independent — `1 − max(0, in_dist − ood) = 1 − (0.994 − 0.524) = 0.530`, matching
the reported value. That identity matters: with OOD pinned at chance, retention is driven *entirely* by
in-distribution accuracy, so a method could "improve retention" merely by getting *worse* in-range. I
note that trap now. The 0.0 on `exact` confirms the diagnosis I sketched: the counter is real but its
scale is length-specific — a `tanh(c_t)` read-out responsive at in-range counts saturates by
`tanh(8.5) ≈ 1` at OOD counts near 85, and a slightly-off per-step increment compounds over twice as many
steps.

That collapse is not something a better recurrence fixes: the far-away block's information must be
*carried* step by step, and over 256 steps even a unit-gain cell integrates a long chain of small
imperfections. So instead of carrying the count through time, let every position look *directly* at every
other and compare blocks in one hop — self-attention. The membership decision "is the `a`-block the same
size as the `c`-block" is fundamentally a comparison between two regions; a recurrence threads it through
a long chain, attention routes it as a single any-to-any lookup, and the CLS token can attend to the
whole string at once. The question this step asks is whether removing the carry removes the OOD collapse.

The attention primitive is `softmax(QK^T/√d_k)V`. The `1/√d_k` is not a choice I get to make but it is
what keeps the softmax responsive: raw dot products of `d_k`-dim vectors have std `√d_k`, and logits
spread that wide make the softmax one-hot with a near-zero Jacobian, so gradient stops flowing to *which*
position to attend. With `hidden_dim=128` and 4 heads, `d_k=32`, and multiple heads let the encoder track
several relations at once — one head the `a→b` boundary, another `b→c`.

The wall that decides this step is that pure attention is permutation-equivariant. Replace input `X` by
`PX` for a permutation `P`: `Q,K,V → PQ,PK,PV`, the scores become `P(QK^T)P^T`, row-wise softmax commutes
with the row-permutation, and the output is `P·softmax(QK^T)V` — the outputs permute identically. Shuffle
the rows, the result shuffles the same way, so the encoder cannot tell `a^n b^n c^n` from a shuffled bag
of the same symbols — fatal for a task whose entire content is order and block length. Order has to be
injected, and *how* it is injected is exactly the lever the counting-expressivity lineage flags as
governing length extrapolation, so it is the real design decision here.

I could inject a *learned* absolute embedding, one trainable row per index, but the training strings reach
index ~65, so rows 66…256 never receive gradient and sit at initialization noise at test length — no
defined value to trust at OOD indices. A relative-bias scheme (ALiBi family) never indexes an untrained
row and is tempting, but it encodes *absolute pairwise distance*, and I have no evidence yet that the
counting decision is about distance rather than within-run index; it also changes the *inside* of the
attention layer, confounding a clean encoder-vs-encoder comparison against the LSTM. So I run the honest
lineage baseline first — sinusoidal encodings added to the embedding — and let it show me *how* an
absolute scheme fails before I spend a positional-representation change. `PE(pos, 2i) = sin(pos/10000^{2i/d})`,
`PE(pos, 2i+1) = cos(pos/10000^{2i/d})`: each dimension pair a sinusoid at its own frequency, and a shift
by `k` is a fixed rotation `R(ωk) = [[cos ωk, −sin ωk],[sin ωk, cos ωk]]` of each `(sin,cos)` pair —
dependent only on `k`, not on `pos` — so a head could in principle learn "compare to the position `k`
away" as one operator valid everywhere. And unlike the learned table, the sinusoid at least *has* a value
at OOD lengths.

What do absolute-index sinusoids actually buy at OOD length? Take the lowest-frequency
dimension: `div_term = 10000^{-126/128} ≈ 1.16e-4`, wavelength `≈ 5.4e4`, far larger than any sequence.
Across training its phase moves over `[0, 65·1.16e-4] = [0, 0.0075]` radians; at test it reaches
`256·1.16e-4 ≈ 0.030` — four times outside the entire trained interval for that channel. The
low-frequency dimensions that carry coarse absolute position take values at test they never took in
training; they are strictly OOD inputs to the first attention layer. The rotation-by-`k` property says
relative offsets are *representable*, but not that a small stack trained with no pressure toward the
relative form *learned* to read them rather than keying off the absolute phase pattern it saw for
`pos ≤ 65` — and the generic outcome for such a stack is the shortcut. So I expect the sinusoidal
transformer to still fail to generalize the count OOD, but to be *less catastrophic on retention* than
the LSTM, for a concrete reason: the LSTM's error is a drift that compounds monotonically with length,
whereas attention's block comparison does not compound step by step, so it degrades more gracefully. That
is the testable distinction.

There is a sharper, architecture-independent reason `exact` OOD stays 0.0 regardless of positional
scheme. The `exact` score is *rounded count equals target* on `T ∈ [128,256]`. In training the count of
`a`'s the regression head ever sees is at most ~63; the fixed `nn.Linear(128,1)`, under smooth-L1, is fit
to emit values in ~`[0,63]`. At test the target can be 200, and a head calibrated to ~63 will not emit
200 for an unseen regime — the *target magnitude itself* is out of distribution, independent of the
encoder. So `exact` OOD is over-determined and the *least* diagnostic of the three environments for
separating positional schemes; the live signal is on `abc`/`length-ood`, a bounded comparison.

I also expect in-distribution `abc` to come out *at or below* the LSTM's 0.994, and that would be
diagnostic rather than a disappointment: the LSTM has a dedicated scalar cell that increments, so
equal-block detection compares two integers it maintained directly; the transformer has no accumulator
and must *reconstruct* the two block lengths from attention patterns and compare them within two layers —
strictly harder to represent. And by the retention identity, a lower in-distribution `abc` mechanically
*raises* retention at fixed chance-level OOD, which is precisely the trap, so I will watch the OOD
accuracy itself, not the retention headline.

Now the encoder on the contract. Scale the token embedding by `√hidden_dim ≈ 11.3` so its amplitude
matches the `O(1)` sinusoids rather than being whispered under them, then add the sinusoidal `PE`. Two
`nn.TransformerEncoderLayer`s, `nhead=4`, feed-forward `4·hidden_dim=512`, `dropout=0.0`,
`activation="gelu"`, `norm_first=True` (pre-norm keeps the residual path clean at initialization for a
small stack) — two layers mirroring the LSTM's depth so the comparison is architecture-vs-architecture,
not depth-vs-depth. `src_key_padding_mask = tokens.eq(pad_id)` so attention never mixes padded positions,
which would otherwise leak a length-dependent constant into every query's softmax. On budget each layer
is ~198k (attention ~65k + FFN ~131k), so the whole encoder ~0.40M, again well under the ceiling —
capacity is not the lever, so I keep width matched to the LSTM.

The pooling *flips* from the LSTM. The encoder is bidirectional — every position, including CLS, attends
to the whole sequence — so the index-0 CLS is now a legitimate global summary, and I pool `h[:,0]`, not
the last position. The same CLS placement forces opposite pooling rules in the two families: the causal
LSTM's index-0 state was blind, this one is the most-informed token. A final `LayerNorm` on the pooled
CLS stabilizes the scale into the head. One buffer detail the OOD lengths force: the precomputed
sinusoidal table can be exceeded, so I extend it on the fly with zero rows for the overflow — those
positions fall back to no positional signal, itself a small part of why the longest positions carry
weaker order information, which I note as an honest limitation feeding the next step.

As with the LSTM, only the encoder is mine: scaled embedding, sinusoidal PE, 2 pre-norm layers, CLS
pooling, LayerNorm — decoder, cross-attention, causal mask, and warmup schedule are irrelevant to a
pooled classifier or fixed by the harness.

The prediction that decides whether this is a real improvement is `length-ood` retention above the
LSTM's 0.530, because attention degrades more gracefully than a compounding accumulator. But I hold the
retention trap: with OOD likely still near chance, a retention rise could come from real OOD signal *or*
merely from a softer in-range fit, so I will only call it genuine generalization if the OOD accuracy
itself lifts off chance. If it does not, the residual failure — no OOD generalization because the
positional code is absolute and OOD at test length — is exactly the gap the next step must close by
changing the positional representation itself.