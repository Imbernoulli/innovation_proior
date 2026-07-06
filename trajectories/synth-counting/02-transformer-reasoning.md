The LSTM numbers tell me the split I was watching for, and they tell me it sharply. In-distribution the
recurrent counter is excellent: `abc` membership lands at 0.994 and the `exact` in-distribution count at
0.998 — the cell really did learn to thread a tally through the sequence and compare blocks, exactly the
finite-precision counter recipe predicts. But every out-of-distribution number is a collapse. On `exact`
the OOD accuracy on `T in [128, 256]` is *exactly* 0.0 — not degraded, *zero* — so the `exact` score is
0.0. On `abc`/`length-ood` the OOD accuracy is 0.524, which is chance for a balanced binary classifier:
the model that was right 99.4% of the time in-range is *guessing* the moment the length doubles.

Two of these numbers are worth reading precisely rather than waving at, because they carry mechanism.
First, is 0.524 really "chance," or a faint signal? The `abc` OOD split is balanced positives against
negatives, so a coin-flip classifier sits at 0.5, and since the frozen harness scores every split on 1,024 freshly
drawn examples, the sampling standard deviation of an accuracy estimate is `sqrt(0.25/1024) ≈ 0.0156`.
The measured 0.524 is `(0.524 − 0.5)/0.0156 ≈ 1.5` standard deviations above 0.5 — well within the noise band of a genuinely chance-level
classifier, so I read it as *no retained signal*, not a weak one. Second, the retention number is not an
independent measurement but an arithmetic consequence I should verify: the harness defines retention as
`1 − max(0, in_dist − ood)`, so for the LSTM that is `1 − (0.994141 − 0.524414) = 1 − 0.469727 =
0.530273`, which matches the reported 0.530 exactly. That identity matters for how I read progress: with
OOD pinned at chance, retention is driven *entirely* by the in-distribution accuracy, and a method could
"improve retention" merely by getting *worse* in-distribution. I note that trap now so I do not mistake
it for generalization later. So the LSTM did not implement an exact integer recurrence that transfers;
it memorized a counter calibrated to lengths up to 64. The diagnosis I sketched at rung one is confirmed
by the 0.0 on `exact`: a per-step increment that is 0.99 instead of 1.00, or a `tanh(c_t)` read-out that
was responsive at training-range counts and saturates by `tanh(8.5) ≈ 1` at OOD counts near 85,
compounds over twice as many steps, and the rounded count is then wrong on essentially every long
string. The counter is real but its *scale* is length-specific.

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
the values. The `1/sqrt(d_k)` is not decoration, and it is worth the arithmetic. If the entries of a
query and a key are unit-scale and independent, a raw dot product of two `d_k`-dimensional vectors has
variance `d_k`, so its standard deviation is `sqrt(d_k)`. For `d_k = hidden_dim / 4 = 32` the logits sit
around `±sqrt(32) ≈ ±5.7`, and the *difference* between two competing logits has standard deviation
`sqrt(2·32) ≈ 8`. A softmax over logits spread by 8 is effectively one-hot: two keys whose logits differ
by 5.7 already differ in softmax weight by a factor `e^{5.7} ≈ 300`. Once the softmax is that peaked its
Jacobian collapses — the derivative of a saturated softmax is near zero — and the attention weights stop
receiving gradient, so the encoder cannot learn *which* position to attend to. Dividing by `sqrt(d_k) =
sqrt(32)` rescales the logits to `O(±1)`, putting the softmax back in its responsive region where the
gradient flows. I use multiple heads so the encoder can attend to several relations at once — one head
can track the `a→b` boundary, another the `b→c` boundary — each head a lower-dimensional projection with
its own softmax, gathered and mixed by an output projection so the per-head findings get composed rather
than siloed. With `hidden_dim = 128` and 4 heads, `d_k = 32`.

The wall that decides this whole rung is that pure attention is permutation-equivariant: `softmax(QK^T)V`
is all dot products and weighted sums over the *set* of positions, with no term that knows which
position is which. I can check it in one line: if `P` is a permutation matrix and I replace the input
`X` by `PX`, then `Q, K, V` become `PQ, PK, PV`, the score matrix becomes `PQ(PK)^T = P(QK^T)P^T`, the
softmax is applied row-wise so it commutes with the row-permutation, and the output is `P·(softmax(QK^T)
V)` — the outputs permute identically. Shuffle the input rows and the output rows shuffle the same way,
which means the encoder cannot tell `a^n b^n c^n` from a shuffled bag of the same symbols, and that is
fatal for a task whose entire content is *order and block length*. So order has to be injected, and
*how* I inject it is exactly the lever the prior-art lineage flagged as governing length extrapolation.

That makes the positional scheme the real design decision of this rung, so I walk the options rather than
reach. A *learned* absolute embedding — one trainable row per index — I reject on the same
untrained-row logic that will bite any absolute scheme: the training strings reach index ~65 (64 content
tokens plus CLS), so only rows 0…65 of the table ever receive a gradient; at test length 256 the rows
66…256 are still at their initialization, pure noise, and a table of `256 × 128` has `~33k` parameters of
which the model trained maybe a quarter. It literally has *no defined value* it can trust at the OOD
indices. A relative-bias scheme in the ALiBi family is more tempting — it adds a bias to the attention
logits proportional to the distance between query and key, so it never indexes an untrained row — but it
encodes *absolute pairwise distance*, and I have no evidence yet that the counting decision is about
distance rather than about within-run index; it also changes the *inside* of the encoder layer (the
attention scores), which would confound an architecture comparison against the LSTM that I want to keep
to the encoder's positional input. I would rather first run the honest lineage baseline and let it show
me *exactly* how an absolute scheme fails, then spend a positional-representation change only if the data
says I must. That baseline is the sinusoidal code: for position `pos` and dimension `i`, `PE(pos, 2i) =
sin(pos / 10000^{2i/d})`, `PE(pos, 2i+1) = cos(pos / 10000^{2i/d})`, added to the token embedding at the
bottom of the stack. Each dimension pair is a sinusoid at its own frequency, the wavelengths sweeping a
geometric range, and the appeal is that a shift by `k` is a fixed position-independent rotation of each
`(sin, cos)` pair, so a head can in principle learn "compare to the position `k` away" as one operator
that works everywhere. Crucially the sinusoid is defined for *any* real position, so unlike the learned
table it at least *has* a value at the OOD lengths rather than feeding the encoder an untrained noise row.

But I have to be honest with myself about what sinusoidal-in-absolute-index actually buys at the OOD
lengths, because this is precisely where I expect the rung to be limited, and saying so now makes the
prediction falsifiable against the numbers. The positional code is a function of the *absolute* index,
and I can make the OOD problem quantitative by looking at the lowest-frequency dimension. The slowest
sinusoid uses `div_term = 10000^{-126/128} = 10^{-3.9375} ≈ 1.16e-4`, so its wavelength is `2π/1.16e-4 ≈
5.4e4` — enormously larger than any sequence. Across the training range the phase of that dimension moves
from `0` to `65 · 1.16e-4 ≈ 0.0075` radians; across the test range it reaches `256 · 1.16e-4 ≈ 0.030`
radians. So on that dimension the encoder trains on values `sin(θ)` for `θ ∈ [0, 0.0075]` and is then
asked, at test, to read `sin(0.030) ≈ 0.030` — a value four times outside the entire trained interval
for that channel. The low-frequency dimensions, the ones that carry coarse *absolute* position, take
values at test that they never took in training; they are strictly out-of-distribution inputs to the
first attention layer. The rotation-by-`k` property says relative offsets are *representable*, but it
does not say the encoder *learned* to read them in a way that holds when the absolute phase pattern
itself is off the trained manifold; in practice the network keys off the absolute phase pattern it was
trained on, and that pattern is OOD at the long lengths. So I expect the sinusoidal transformer to still
fail to *generalize* the count OOD, but to be *less catastrophic on retention* than the LSTM, for a
specific reason: the LSTM's error is a compounding drift that grows monotonically with length, whereas
attention's block-comparison does not compound step by step, so even if it cannot read the long absolute
positions it degrades more gracefully than a saturating accumulator. That is the testable distinction.

It is worth spending one more line on *why* the rotation-by-`k` property is real yet insufficient,
because it is the crux of the whole absolute-vs-relative story and I want the mechanism, not a slogan.
For a single frequency `ω`, the pair `(sin(ω·pos), cos(ω·pos))` under a shift `pos → pos + k` transforms
by the rotation matrix `R(ωk) = [[cos ωk, −sin ωk], [sin ωk, cos ωk]]`, which depends only on `k`, not on
`pos`. So a linear map inside an attention head *could* implement "attend to the token `k` positions
back" as the single fixed operator `R(ωk)` applied to the query's positional part, and that operator is
the same at `pos = 5` and at `pos = 205`. This is the honest case for sinusoids extrapolating. The catch
is that the encoder does not receive `k` as an input and does not get to apply `R` in isolation; it sees
the *sum* embedding-plus-`PE` at each absolute position and must learn, from gradient on training
positions only, weights that read the relation. Whether it converges to the clean relative operator or to
some shortcut that merely fits the absolute phase patterns seen for `pos ≤ 65` is an empirical matter of
optimization, and the generic outcome for a small stack trained with no pressure toward the relative form
is the shortcut. The low-frequency dimensions make this worse, as the wavelength arithmetic showed: their
phase is nearly linear in `pos` over the whole sequence, so they behave like a raw absolute coordinate,
and a raw coordinate at `pos = 200` is simply a value the network never saw. So I expect *some* in-range
benefit from the relative-representability (the transformer should be a competent in-distribution
classifier) and little OOD benefit (the learned reader is keyed to absolute phase), which is exactly the
shape of prediction I want the numbers to adjudicate.

I should also reason about why in-distribution `abc` might come out *below* the LSTM's 0.994 rather than
matching it, because that would be diagnostic rather than a disappointment. The LSTM has a dedicated
mechanism for a count — a scalar cell that literally increments — so equal-block detection is a
comparison of two integers it maintained directly. The transformer has no accumulator; to know that the
`a`-block and the `c`-block are the same size it must, through softmax attention and a position-wise
feed-forward, *reconstruct* the two block lengths from the pattern of which positions attend to which and
then compare them, all within two layers. That is a strictly harder thing to represent than a running
tally, so even in-distribution I would not be shocked if the transformer trades a little membership
sharpness for its graceful-degradation property. And by the retention identity, a lower in-distribution
`abc` mechanically *raises* retention at fixed chance-level OOD — which is precisely the trap I flagged,
and the reason I will insist on watching the OOD accuracy itself, not the retention headline.

There is a sharper, architecture-independent reason to expect the `exact` OOD score to stay pinned at 0.0
no matter how good the positional scheme is, and it is worth separating from the position argument. The
`exact` score is *rounded count equals target* on `T ∈ [128, 256]`. In training the strings reach length
64, so the target count of `a`'s the regression head ever sees is at most ~63; the fixed
`nn.Linear(128, 1)` head is fit, under smooth-L1, to reproduce values in roughly `[0, 63]`. At test the
target can be 200. A linear head calibrated to emit numbers up to ~63 will not emit 200 for an unseen
input regime — the *target magnitude itself* is out of distribution, independent of whether the encoder
represents position perfectly. So the `exact` OOD failure is over-determined: even a hypothetical perfect
positional signal runs into a head that was never asked to produce a number that large. I flag this
because it means `exact` OOD is the *least* diagnostic of the three environments for separating
positional schemes — the live signal is going to be on `abc`/`length-ood`, where the decision is a
bounded comparison rather than an unbounded magnitude.

Now the encoder, fit to the contract, with the shapes checked. Embedding `nn.Embedding(vocab_size,
hidden_dim, padding_idx=pad_id)` turns `[B, T]` into `[B, T, 128]`, scaled by `sqrt(hidden_dim) ≈ 11.3`
so its amplitude matches the `O(1)` sinusoids rather than being whispered under them — without the scale
the learned token identity would be a small perturbation on a unit-amplitude positional wave and the
first layer would attend mostly to position. Add the sinusoidal `PE`, same `[B, T, 128]`. Then 2
`nn.TransformerEncoderLayer`s with `nhead=4`, feed-forward width `4·hidden_dim = 512` (the position-wise
nonlinear capacity), `dropout=0.0`, `activation="gelu"`, `norm_first=True` (pre-norm residual blocks, the
stable ordering for a small stack that keeps the residual path clean at initialization). Two layers,
mirroring the LSTM's depth so the comparison is architecture-vs-architecture, not depth-vs-depth. On the
budget this is comfortably inside the 5M ceiling: each encoder layer is attention `in_proj 3·128·128 +
out_proj 128·128 ≈ 65k` plus feed-forward `128·512 + 512·128 ≈ 131k`, about `~198k` per layer, so two
layers plus embeddings and norms land near `0.40M` — again well under 10% of the cap, so, as at rung one,
capacity is not the binding constraint and I keep width matched to the LSTM rather than spending
parameters that would not touch the length behavior. The padding mask is passed as `src_key_padding_mask
= tokens.eq(pad_id)` so attention never mixes in padded positions — important because the three
environments produce very different padded widths within a batch, and an unmasked pad column would leak a
constant, length-dependent bias into every query's softmax.

And here the pooling rule *flips* from rung one: the encoder is bidirectional, every position including
CLS attends to the whole sequence, so the CLS position at index 0 is now a legitimate global summary — I
pool `h[:, 0]` to get `[B, 128]`, the CLS, not the last position. This is the attention reading of the
same contract, and it is the natural one: CLS pooling is exactly why the scaffold seeds a CLS token, and
because attention is bidirectional the index-0 token has already gathered information from every content
position, so unlike the causal LSTM it is not summarising an empty prefix. A final `LayerNorm(hidden_dim)`
on the pooled CLS stabilizes the scale into the fixed head. I pause on one asymmetry the pooling choice
exposes against rung one: the LSTM had to read `lengths - 1` because its index-0 state was blind, whereas
here index-0 is the most-informed token, so the *same* CLS placement forces opposite pooling rules in the
two families. That is not an incidental code detail — it is the scaffold quietly encoding that a causal
summariser and a bidirectional one see the sequence from opposite ends, and reading it wrong would, as at
rung one, silently cap accuracy at chance.

One sinusoid-buffer detail the harness forces me to handle: the OOD lengths can exceed the precomputed
positional table, so the table must extend on the fly rather than crash. I precompute `PE` up to a
generous `max_len` and, if a batch is longer, concatenate zero rows for the overflow — those positions
fall back to no positional signal, which is part of the standard recipe and, notably, is itself a small
part of *why* the very longest positions carry weaker order information. I note this not as a fix but as
an honest limitation that feeds the next rung's motivation.

I should again flag what this rung is *not*, per the same-named-baseline note. This is not the full
encoder-decoder translation Transformer with cross-attention, causal masking, label smoothing, tied
embeddings, and the warmup-then-inverse-sqrt schedule — all of that is either irrelevant to a pooled
*encoder* classifier or lives in the fixed harness (AdamW at a flat 3e-4, BCE/smooth-L1). The editable
slot is the encoder only: scaled embedding, sinusoidal PE, 2 pre-norm encoder layers, CLS pooling,
LayerNorm. So I reason about the encoder's length behavior, not about an optimizer or a decoder I do not
have.

Falsifiable expectations against the LSTM's numbers. In-distribution `abc` I expect to be *at or below*
the LSTM's 0.994 — the recurrent counter is genuinely good in-range and a 2-layer encoder may not match
it on the easy in-distribution regime, so a slightly lower in-distribution `abc` would not surprise me
and would itself be informative. On `exact` OOD I expect the same 0.0 the LSTM hit, now for the *doubled*
reason above — absolute-index positions are OOD *and* the target magnitude is OOD — so the `exact` score
should stay pinned at zero; if it moves off zero I am wrong about at least one of those limitations. The
prediction that decides whether this rung is an *improvement* is `length-ood` retention: I expect it to
beat the LSTM's 0.530, because attention's block comparison degrades more gracefully than the LSTM's
compounding drift. But I hold onto the retention-arithmetic trap I flagged: since OOD is likely to stay
near chance (0.524-ish), retention is `1 − (in_dist − 0.524)`, so a retention rise could come *either*
from real OOD signal *or* merely from a lower in-distribution `abc`. I will only call it genuine
generalization if the OOD accuracy itself lifts off chance; otherwise a higher retention is just the
transformer being softer in-range. If retention rises while `abc`-in-distribution holds up, the rung is a
real step up on the metric the task is built to measure, and the residual failure — no genuine OOD
generalization because the positional code is absolute and out-of-distribution at test length — is
exactly the gap the strongest rung must close by changing the positional representation itself. The full
scaffold module is in the answer.
