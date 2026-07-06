I start at the bottom of the ladder, before I commit to any Transformer at all, because I want a
reference point that encodes order in the most classical way available — through recurrence rather than
through an explicit positional code. The default scaffold fill is a learned absolute positional
embedding bolted onto the decoder-only backbone, and I already distrust it for the OOD split: a learned
table has one trainable vector per slot and literally no entry past the training range, so at content
length 30 there is nothing to look up. I want the first rung to be a method whose handling of position
is *structural*, so that whatever it does at unseen lengths is a property of the architecture and not of
a table that runs out. The recurrent attention seq2seq of Bahdanau, Cho, and Bengio (2015) is exactly
that reference: it carries order purely through the step-by-step recurrence and adds no positional
encoding anywhere. It is also the natural non-Transformer baseline the task names, the thing the rest of
the ladder will be measured against, so it belongs at rung one.

Before I settle on it, let me actually walk the design space I have at this rung, because there are three
recurrent-flavoured options in front of me and I want to eliminate the tempting ones by argument, not by
reflex. Option one is the plain encoder-decoder that compresses the whole source into a single
fixed-width state `c = h_T` and decodes from that. It is the cheapest thing to build, and at the training
lengths it would probably work — so it is genuinely tempting. But I can price out why it must fail the
only question this task asks. The content alphabet is 16 symbols, so a source of length `L` carries
`L·log2(16) = 4L` bits of information that the decoder must reproduce exactly. At the OOD ceiling `L = 40`
that is 160 bits the summary vector has to hold; at the training ceiling `L = 20` it is only 80. A
`d_model = 128` float vector is nominally wide enough for either, but the pressure is not raw capacity —
it is that the *same* fixed width must serve every length, so a network tuned to pack 80 bits over `[1,20]`
has no reason to have learned the packing for 160 bits it never saw, and the decoder must unspool all of
it through a recurrence with no way to re-read the source. That is precisely a length-dependent
bottleneck, and length dependence is the failure mode I am here to avoid. Option two is a Transformer with
the scaffold's learned absolute table — but that is the default fill I already distrust, and it is not a
recurrent reference at all; it belongs to the rungs above, not to the structural floor I want here. Option
three, the one I take, keeps *every* encoder state and lets the decoder read a content-weighted blend, so
the read is length-invariant in width regardless of how long the source is. The additive-attention
seq2seq is option three, and the bit-counting argument against option one is exactly why it exists.

Let me reconstruct what it actually computes, because the way it has to be fit into this harness is not
the textbook translation version. The disease the additive-attention seq2seq cures is the
fixed-length-context bottleneck I just priced: an RNN reads the source left to right,
`h_t = f(x_t, h_{t-1})`, and the older design hands the decoder only the final state `c = h_{T}` as a
single fixed-width summary of the whole input. Everything the decoder will ever need about the source has
to be squeezed into that one vector before the first output symbol comes out, and a constant-size vector
cannot hold an unbounded amount of content — so the basic model degrades on long inputs, worst beyond the
lengths it trained on. The cure is to keep *all* the encoder's per-position states `h_1, …, h_{T}` and let
the decoder, at each output step, search the source for the position it should be reading right now and
pull a content-weighted blend from there. If the weights `α_{i1}, …, α_{iT}` are nonnegative and sum to
one, the decoder reads `c_i = Σ_j α_{ij} h_j`, a per-step context recomputed every output step instead of
one summary frozen at the start. That removes the bottleneck: a long source is no harder than a short one
because the decoder only ever reads one position's worth of blended content at a time, and the width of
that read is `d_model` no matter whether `T` is 5 or 40.

The weights themselves come from a content match. The decoder, before emitting `y_i`, knows its
recurrent state `s_{i-1}` — what it has produced and what it intends next — and each source position is
represented by its annotation `h_j`. So the score `e_{ij} = a(s_{i-1}, h_j)` measures how well position
`j` matches what the decoder is about to do, and the natural cheap scorer for a (query, key) pair whose
vectors need not live in the same space is a one-hidden-layer additive MLP,
`e_{ij} = v_a^⊤ tanh(W_a s_{i-1} + U_a h_j)`. A softmax over positions turns those scores into a
distribution, `α_{ij} = softmax_j(e_{ij})`, which is what keeps the read length-invariant — `c_i` is a
convex combination of the annotations, so its scale does not grow with the number of source positions.
I use `s_{i-1}` and not `s_i` as the query because `s_i` depends on `c_i`, which would be circular; the
pre-step state is also the right semantics for "where should I look now." This is the load-bearing
machinery: a content-based, non-monotonic, differentiable soft alignment, with order supplied entirely
by the recurrence that builds the `h_j` and the `s_i`. The non-monotonic part matters concretely for this
task: `reverse` needs the alignment for output position `k` to point at source position `L−k+1`, which
runs *backwards* as `k` grows, and a content-plus-state scorer has no built-in preference for increasing
`j`, so it can place its mass wherever the match is — forward for `delim`, backward for `reverse`, wrapped
for `repeat` — with the same mechanism.

Now the part that matters most for this trajectory: how that method has to be expressed in *this* task's
edit surface, because it is not the generic translation network and I must derive the version the
harness actually runs. The scaffold gives me a single token stream
`[BOS] x_1 … x_T [SEP] y_1 … y_M [EOS]` and a fixed `forward(tokens) -> [B, T, V]` API; there is no
separate source and target tensor. So I have to manufacture the encoder-decoder split inside `forward`.
The `[SEP]` token is the seam: I find `sep_pos` per row as the position of `SEP_ID`, treat the content
between `BOS` and `SEP` as the source, and treat everything from `SEP` onward as the decoding region. Let
me get the shapes exactly right, because an off-by-one here would silently corrupt the source. With the
layout above, `[BOS]` sits at index 0 and `SEP` at index `sep_pos`, so the source content occupies indices
`1 … sep_pos-1`, of length `src_lens = sep_pos - 1`. I slice `tokens[:, 1 : 1+max_src]` where
`max_src = max(src_lens)` across the batch, build a per-row mask `src_idx < src_lens` so shorter rows in
the batch zero out their padding tail, and overwrite the masked slots with `PAD_ID`. For a concrete row —
`delim` with `x = a b c`, stream `[BOS a b c SEP a b c EOS]`, `T = 9` — `SEP_ID` lands at index 4, so
`sep_pos = 4`, `src_lens = 3`, and the slice `tokens[:,1:4] = a b c` is exactly the source, no seam, no
BOS. That check is worth doing by hand precisely because the split is invisible in the API.

I embed the source slice, run a *bidirectional* LSTM encoder over it so each annotation `h_j` sees both
sides of position `j` (the forward direction summarises `x_1..x_j`, the backward direction `x_j..x_T`,
concatenated to width `d_model`), and mask out the padding positions so attention never reads them.
Bidirectionality is not cosmetic for a copy task: for `reverse`, the annotation at source position `j`
that the decoder will want when it is near the *end* of its output must already encode where `j` sits
relative to the source end, which the backward pass supplies; a purely left-to-right encoder would make
`h_j` ignorant of everything to its right. Choosing hidden width `d_model // 2 = 64` per direction so the
concatenation lands back at `d_model = 128` keeps the annotation width matched to the decoder without a
projection. The decoder is an `LSTMCell` initialised from a `tanh` projection of the masked-mean encoder
summary, and it steps over *every* position of the stream, but I only let it advance — and only let it
write logits — once the position has reached or passed `sep_pos`. Concretely a per-row boolean
`active = pos >= sep_pos` gates the state update (`h, c` are held frozen on inactive positions) and the
logit write, so the model produces predictions only on the target region, which is exactly where the loss
mask lives. Freezing the state on inactive positions rather than skipping them is what lets me run one
uniform `for pos in range(T)` loop over a rectangular `[B, T]` batch whose rows have different `sep_pos`:
each row simply starts writing when its own gate opens.

The attention inside the decoder is the Bahdanau additive form, derived above, but typed to this
harness. I project the encoder annotations once into keys `keys = attn_key(enc_out)`; at each position I
project the current decoder state into a query `attn_query(h)`, add it to the keys, push through `tanh`,
read off a scalar per source position with `attn_v`, mask the padded positions to `-inf`, softmax to get
the alignment weights, and form `context = Σ_j α_j h_j` by a batched matrix-vector product against the
encoder outputs. The decoder cell input is the concatenation of the embedding of the current token and
that context; the output logits come from a linear layer on the concatenation of the new state and the
context. Two details are forced by the single-stream layout. First, the decoder reads `tokens[:, pos]`
as its input embedding at every step — there is no teacher-forced separate target tensor, so the prefix
(content and `SEP`) flows through the cell too, but its logits are discarded by the `active` gate.
Second, because the whole thing is one `forward` over a `[B, T]` tensor, the attention is recomputed at
every one of the `T` positions, which makes this the slowest baseline on the clock by a wide margin —
the recurrence is sequential and the per-step alignment is an extra matmul each step.

I want to price that cost, because it will show up on the elapsed column and I should predict it rather
than be surprised. The forward pass is a Python-level loop of `T` sequential steps — `T` runs up to
`2·40 + 3 ≈ 83` on the OOD split — and each step does an attention over up to `max_src ≈ 40` source
positions: a `[B, max_src, d]` add, a `tanh`, a reduction to `[B, max_src]` scores, a softmax, and a
`bmm`. Nothing here parallelises across positions the way a Transformer's single batched attention does,
because step `pos` needs `h` from step `pos-1`. So the wall-clock is `O(T)` sequential kernel launches per
batch times 6000 steps, and the per-step attention is an extra `O(max_src · d)` matmul. That is the
structural reason to expect this rung to be several times slower than any Transformer rung — I would guess
in the hundreds of seconds per variant, several times what a single parallel Transformer forward would cost — and it is the
second, quieter argument (after length generalization) for moving off recurrence once I have the
reference number.

To keep the budget honest let me count parameters, because "small model" should be a number. The token
embedding is `20 × 128 = 2560`. The bidirectional encoder LSTM at input 128, hidden 64, is dominated by
its `4·(64·128 + 64·64)` gate weights per direction, roughly `50k` per direction, `~99k` for the pair.
The two `init_h`/`init_c` projections are `128·128 + 128` each, `~33k` together. The additive attention
adds `attn_key` and `attn_query` at `128·128` each and a `128·1` `attn_v`, `~33k`. The decoder `LSTMCell`
at input `2·128 = 256`, hidden 128 is the single heaviest block, `4·(128·256 + 128·128) ≈ 198k`. The
readout `Linear(256, 20)` is `~5k`. That totals to roughly `0.37M` parameters — a genuinely small model,
with the decoder cell and the encoder carrying the mass, and no positional table anywhere. The absence of
that table is the whole point: there is no slot to run out of at length 40.

This is the place to be explicit about what the harness does *not* expose, because it changes the method
from its translation form. There is no bidirectional GRU with reset/update gates and a maxout deep
output here — I use a plain bidirectional `LSTM` for the encoder and a single `LSTMCell` for the
decoder, which is the lighter gated unit with the same long-range-gradient property, and a bare linear
readout instead of maxout. There is no beam search — evaluation is greedy autoregressive decoding driven
by the fixed loop. There is no separate `[NULL]`/fertility machinery. The `build_positional_scheme` hook
is returned as an empty placeholder (`name="lstm_none"`, all three callables `None`, no extra modules),
because the recurrence *is* the positional mechanism: the encoder's left-to-right and right-to-left
passes stamp order into the annotations, and the decoder's sequential stepping stamps order into the
output, so there is nothing for the Transformer's positional hooks to do. `build_model` returns the
recurrent module directly instead of a `SeqModel`. The single design lever the task actually cares about
— "how does position enter" — is answered here by "through recurrence, with no explicit code at all."

Let me verify the whole gated loop reproduces the intended computation on the small example, because if
the `active` gate is wrong the loss would train on garbage and I would never know from the shapes. Take
`delim`, `x = a b c`, stream `[BOS a b c SEP a b c EOS]`, indices `0..8`, `sep_pos = 4`. For positions
`0,1,2,3` (`BOS a b c`) the gate `pos >= 4` is false, so `h, c` stay at their initial `tanh`-projected
summary and no logits are written — the prefix flows through the embedding lookup but changes nothing
downstream. At `pos = 4` the token is `SEP`, the gate opens, the decoder takes one step reading the `SEP`
embedding plus the attention context, and writes the logit that must predict the first target symbol `a`;
the loss mask expects exactly a prediction at position 4 for target position 1, so the seam aligns. At
`pos = 5,6,7` the decoder reads the already-emitted `a, b, c` and predicts `b, c, EOS`. That is the
correct teacher-forced target sequence, produced only on the active region, with the alignment free to
point `α` at source index 1 then 2 then 3 for `delim` — or at 3, 2, 1 for `reverse`, which the content
scorer permits. The trace confirms the gate writes predictions on precisely the loss-masked positions.

One initialisation choice deserves its own justification, because it is the seam between encoder and
decoder and I picked it against an obvious alternative. The decoder cell needs a starting `(h, c)`, and
the two candidates are the encoder's final state or a summary of all annotations. Seeding from the final
forward/backward state is the textbook default, but the final state of a bidirectional pass is
*positional* — it emphasises the source ends — whereas a copy task wants the decoder to begin with an
even, position-agnostic prior over the source so the attention, not the seed, decides where to look
first. So I initialise from the masked *mean* of the annotations, `summary = (Σ_j h_j) / src_lens`, pushed
through `tanh(init_h)` and `tanh(init_c)`. Dividing by the true `src_lens` rather than by `max_src` is
load-bearing: if I divided by the padded width, a length-3 source in a batch whose `max_src` is 40 would
have its summary shrunk by `3/40`, an artefact of batching that the model would then have to learn to
undo differently at every length — exactly the kind of length-coupling I am trying to keep out of the
seed. The mean also keeps the seed's scale `O(1)` in the annotations regardless of `T`, matching the
length-invariance the convex-combination context already has downstream.

It is worth checking numerically that the context read really is scale-invariant in length, since that
is the property the whole bottleneck argument rests on. The context is `c_i = Σ_j α_j h_j` with the `α_j`
a softmax, so `Σ_j α_j = 1` exactly, and each `h_j` is a `tanh`-bounded LSTM output living in roughly
`[-1, 1]` per coordinate. A convex combination of vectors in a bounded box is itself in that box: whether
the sum runs over 5 source positions or 40, `c_i` cannot grow past the per-coordinate bound, so a
length-40 read arrives at the decoder cell at the same scale as a length-5 read. Contrast the discarded
plain-encoder path, where the summary is a *single* `h_T` that must encode all 160 bits at once — there
the length shows up as saturation pressure inside one vector, whereas here length only changes *which*
annotations get weight, never the magnitude of what the decoder receives. That is the concrete sense in
which soft attention removes the length dependence, and it is why I trust the mechanism to at least state
the right rule at OOD even if it cannot hold a long output together.

The three variants stress that alignment differently, and naming the target index for each makes the
`reverse`-is-hardest intuition precise. For `delim`, output position `k` must align to source position
`k` — the identity map, a diagonal alignment the softmax can track by a constant offset from the previous
step. For `repeat` (`y = x` twice), output position `k` aligns to source position `((k−1) mod L) + 1` — a
sawtooth that resets at the seam of the two copies, so the alignment must jump back to source index 1
exactly once, at `k = L+1`, which is a length-dependent event the model has to detect. For `reverse`,
output position `k` aligns to source position `L − k + 1` — a strictly *descending* diagonal, the one
case where the alignment must move opposite to the output index. All three are expressible by the same
content-plus-state scorer because it is non-monotonic, but the descending and the wrap patterns lean
harder on the annotation encoding the source *length* (so the model knows where `L−k+1` and the wrap point
land), and length is exactly what a fixed-width encoder represents least reliably out of range. So even
before any numbers I expect the OOD token accuracy to order `delim ≥ repeat ≥ reverse`, tracking how much
each variant's alignment target depends on knowing `L` at an unseen length.

There is a compounding-error arithmetic worth doing on paper now, because it predicts the *shape* of the
gap between the two metrics before I ever see a number. Greedy autoregressive exact match over an output
of length `M` requires every one of the `M` next-token predictions to be individually correct, so if the
per-token accuracy in a regime is roughly `p` and the errors were independent, the sequence-level match
would be about `p^M`. Feed the OOD ceiling `M = 40` through that: even a strong per-token `p = 0.95` gives
`0.95^40 ≈ 0.13`, and `p = 0.90` gives `0.90^40 ≈ 0.015` — already essentially zero exact match while the
token column still reads 0.90. The errors are not truly independent and a copy model can self-correct, so
this is a bound-flavoured estimate rather than a prediction of the exact value, but its lesson is robust:
exact match is a product over the whole output and collapses fast, while token accuracy is an average and
degrades gently. That is precisely why I expect the two columns to tell different stories on OOD — a
near-zero exact match sitting above a clearly-non-zero token accuracy is the fingerprint of a model whose
local rule survives but whose long output cannot hold together, and it is the quantity the rungs above
must move.

What I expect, and what I will watch on the two splits. At the training lengths every variant should be
easy: the source is short, the bottleneck never bites, and content-based soft alignment plus recurrence
should give near-perfect in-distribution exact match across copy, repeat, and reverse — the alignment
just has to learn to point at the right source position (the same index for `delim`, the mirrored index
for `reverse`, the modular index for `repeat`). The OOD split is the real test, and here I am genuinely
uncertain. In principle recurrence and content-based attention should generalise the *rule* of copying
to longer inputs, since neither mechanism has a hard length cap the way the learned table does. But two
structural pressures push the other way at length 30–40. The encoder's hidden state is still a
fixed-width channel, so a longer source crowds more content through the same annotations; and the
decoder state, initialised from a single masked-mean summary, accumulates drift over a longer output
that no positional anchor corrects. So I expect the sequence-level exact match to be the brittle metric:
one wrong symbol anywhere fails the whole sequence, and over 30–40 output steps a single drift is likely.
The token-level accuracy should degrade more gracefully than exact match, since most symbols can still
be aligned even when the sequence is not perfect end to end. If OOD exact match collapses to zero while
OOD token accuracy stays well above chance, that is the signature of a model that has the right local
alignment rule but cannot hold a long output together without an explicit position signal — which is
exactly the gap the Transformer rungs above will try to close by putting order back in as a code rather
than as a recurrence. That falsifiable shape — perfect ID, zero OOD exact match, non-trivial OOD token
accuracy — is what rung one should establish as the floor, and it is what the next rung will have to
diagnose and beat. (The full scaffold module is in the answer.)
