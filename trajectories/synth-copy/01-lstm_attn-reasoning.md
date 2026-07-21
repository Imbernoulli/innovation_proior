I start at the bottom of the ladder, before I commit to any Transformer at all, because I want a
reference point that encodes order in the most classical way available — through recurrence rather than
through an explicit positional code. The default scaffold fill is a learned absolute positional
embedding bolted onto the decoder-only backbone, and I already distrust it for the OOD split: a learned
table has one trainable vector per slot and literally no entry past the training range, so at content
length 30 there is nothing to look up. I want the first rung to be a method whose handling of position
is *structural*, so that whatever it does at unseen lengths is a property of the architecture and not of
a table that runs out. The recurrent attention seq2seq of Bahdanau, Cho, and Bengio (2015) is exactly
that reference: it carries order purely through the step-by-step recurrence and adds no positional
encoding anywhere. It is also the natural non-Transformer baseline the task names, so it belongs at rung
one.

There are three recurrent-flavoured options in front of me, and only one survives an argument. The plain
encoder-decoder compresses the whole source into a single fixed-width state `c = h_T` and decodes from
that; it is the cheapest thing to build and at training lengths it would probably work. But I can price
out why it must fail the only question this task asks. The content alphabet is 16 symbols, so a source of
length `L` carries `L·log2(16) = 4L` bits the decoder must reproduce exactly — 80 bits at the training
ceiling `L = 20`, 160 bits at the OOD ceiling `L = 40`. A `d_model = 128` float vector is nominally wide
enough for either, but the pressure is not raw capacity: the *same* fixed width must serve every length,
so a network tuned to pack 80 bits over `[1,20]` has no reason to have learned the packing for 160 bits
it never saw, and the decoder must unspool all of it through a recurrence with no way to re-read the
source. That is a length-dependent bottleneck, exactly the failure mode I am here to avoid. The scaffold's
learned absolute table is not a recurrent reference at all and belongs to the rungs above. The option I
take keeps *every* encoder state and lets the decoder read a content-weighted blend `c_i = Σ_j α_{ij} h_j`,
so the read is length-invariant in width regardless of how long the source is: the additive-attention
seq2seq. The bit-counting argument is exactly why it exists — an RNN reads the source left to right and,
in the older design, hands the decoder only `c = h_T`, and a constant-size vector cannot hold an unbounded
amount of content. Keeping all annotations `h_1, …, h_T` and pulling a fresh convex blend at each output
step removes that: a long source is no harder than a short one, and the width of the read is `d_model`
whether `T` is 5 or 40.

The weights come from a content match. Before emitting `y_i` the decoder knows its recurrent state
`s_{i-1}` — what it has produced and intends next — and each source position is represented by its
annotation `h_j`. The score `e_{ij} = a(s_{i-1}, h_j)` measures how well position `j` matches what the
decoder is about to do, and the natural cheap scorer for a (query, key) pair whose vectors need not share
a space is a one-hidden-layer additive MLP, `e_{ij} = v_a^⊤ tanh(W_a s_{i-1} + U_a h_j)`. A softmax over
positions turns those scores into a distribution `α_{ij}`, which keeps the read length-invariant — `c_i`
is a convex combination of annotations, so its scale does not grow with the number of source positions. I
use `s_{i-1}` and not `s_i` as the query because `s_i` depends on `c_i`, which would be circular; the
pre-step state is also the right semantics for "where should I look now." The non-monotonic part matters
concretely for this task: `reverse` needs the alignment for output position `k` to point at source
position `L−k+1`, which runs *backwards* as `k` grows, and a content-plus-state scorer has no built-in
preference for increasing `j`, so it can place its mass wherever the match is — forward for `delim`,
backward for `reverse`, wrapped for `repeat` — with the same mechanism.

Now the part that matters most: how that method has to be expressed in *this* task's edit surface, because
it is not the generic translation network. The scaffold gives me a single token stream
`[BOS] x_1 … x_T [SEP] y_1 … y_M [EOS]` and a fixed `forward(tokens) -> [B, T, V]` API; there is no
separate source and target tensor. So I manufacture the encoder-decoder split inside `forward`. The `[SEP]`
token is the seam: I find `sep_pos` per row as the position of `SEP_ID`, treat the content between `BOS`
and `SEP` as the source, and everything from `SEP` onward as the decoding region. With `[BOS]` at index 0
and `SEP` at `sep_pos`, the source occupies indices `1 … sep_pos-1`, of length `src_lens = sep_pos - 1`. I
slice `tokens[:, 1 : 1+max_src]` where `max_src = max(src_lens)`, build a per-row mask `src_idx < src_lens`
so shorter rows zero their padding tail, and overwrite the masked slots with `PAD_ID`. For `delim` with
`x = a b c`, stream `[BOS a b c SEP a b c EOS]`, `SEP_ID` lands at index 4, so `sep_pos = 4`,
`src_lens = 3`, and `tokens[:,1:4] = a b c` is exactly the source — no seam, no BOS. The split is invisible
in the API, which is precisely why the off-by-one has to be pinned down here.

I embed the source slice, run a *bidirectional* LSTM encoder over it so each annotation `h_j` sees both
sides of position `j` (forward summarises `x_1..x_j`, backward `x_j..x_T`, concatenated to width
`d_model`), and mask out padding so attention never reads it. Bidirectionality is not cosmetic for a copy
task: for `reverse`, the annotation at source position `j` that the decoder wants when it is near the *end*
of its output must already encode where `j` sits relative to the source end, which the backward pass
supplies; a purely left-to-right encoder would make `h_j` ignorant of everything to its right. Hidden
width `d_model // 2 = 64` per direction so the concatenation lands back at `d_model = 128` without a
projection. The decoder is an `LSTMCell` initialised from a `tanh` projection of the masked-mean encoder
summary, and it steps over *every* position of the stream, but I only let it advance — and only let it
write logits — once the position has reached or passed `sep_pos`. A per-row boolean `active = pos >=
sep_pos` gates the state update (`h, c` held frozen on inactive positions) and the logit write, so the
model produces predictions only on the target region, exactly where the loss mask lives. Freezing the
state on inactive positions rather than skipping them is what lets me run one uniform `for pos in
range(T)` loop over a rectangular `[B, T]` batch whose rows have different `sep_pos`: each row starts
writing when its own gate opens. On the `delim` example, positions `0..3` (`BOS a b c`) keep the initial
summary and write nothing; at `pos = 4` the gate opens on `SEP` and the cell writes the logit that must
predict the first target symbol `a`, which is exactly the position the loss mask expects — the seam
aligns, and the alignment is then free to point `α` at source index 1, 2, 3 for `delim` or 3, 2, 1 for
`reverse`.

The attention inside the decoder is the Bahdanau additive form, typed to this harness. I project the
encoder annotations once into keys `keys = attn_key(enc_out)`; at each position I project the current
decoder state into a query, add it to the keys, push through `tanh`, read off a scalar per source position
with `attn_v`, mask the padded positions to `-inf`, softmax to get the alignment weights, and form
`context = Σ_j α_j h_j` by a batched matrix-vector product against the encoder outputs. The decoder cell
input is the concatenation of the current token's embedding and that context; the logits come from a
linear layer on the concatenation of the new state and the context. Two details are forced by the
single-stream layout. First, the decoder reads `tokens[:, pos]` as its input embedding at every step —
there is no teacher-forced separate target tensor, so the prefix flows through the cell too, but its logits
are discarded by the `active` gate. Second, because the whole thing is one `forward` over a `[B, T]`
tensor, the attention is recomputed at every one of the `T` positions.

That last point makes this the slowest baseline on the clock by a wide margin, and I should predict its
size rather than be surprised. The forward pass is a Python-level loop of `T` sequential steps — `T` runs
up to `2·40 + 3 ≈ 83` on OOD — and each step does an attention over up to `max_src ≈ 40` positions: a
`[B, max_src, d]` add, a `tanh`, a reduction, a softmax, a `bmm`. Nothing here parallelises across
positions the way a Transformer's single batched attention does, because step `pos` needs `h` from
step `pos-1`. So the wall-clock is `O(T)` sequential kernel launches per batch times 6000 steps — I would
guess in the hundreds of seconds per variant, several times what a single parallel Transformer forward
would cost. That is the second, quieter argument (after length generalization) for moving off recurrence
once I have the reference number.

The design deliberately drops the translation-era trimmings the harness does not need. There is no
bidirectional GRU with maxout deep output — I use a plain bidirectional `LSTM` encoder and a single
`LSTMCell` decoder, the lighter gated units with the same long-range-gradient property, and a bare linear
readout. No beam search — evaluation is greedy autoregressive decoding driven by the fixed loop. No
`[NULL]`/fertility machinery. The token embedding, encoder pair, and decoder cell come to roughly `0.37M`
parameters, a genuinely small model — and, load-bearingly, with no positional table anywhere: there is no
slot to run out of at length 40. The `build_positional_scheme` hook is returned as an empty placeholder
(`name="lstm_none"`, all three callables `None`), because the recurrence *is* the positional mechanism,
and `build_model` returns the recurrent module directly instead of a `SeqModel`. The one design lever the
task cares about — how does position enter — is answered here by "through recurrence, with no explicit
code at all."

One initialisation choice deserves its own justification, because it is the seam between encoder and
decoder and I pick it against an obvious alternative. The decoder cell needs a starting `(h, c)`, and the
two candidates are the encoder's final state or a summary of all annotations. Seeding from the final
forward/backward state is the textbook default, but that state is *positional* — it emphasises the source
ends — whereas a copy task wants the decoder to begin with an even, position-agnostic prior over the
source so the attention, not the seed, decides where to look first. So I initialise from the masked *mean*
of the annotations, `summary = (Σ_j h_j) / src_lens`, through `tanh(init_h)` and `tanh(init_c)`. Dividing
by the true `src_lens` rather than by `max_src` is load-bearing: if I divided by the padded width, a
length-3 source in a batch whose `max_src` is 40 would have its summary shrunk by `3/40`, a batching
artefact the model would then have to learn to undo differently at every length — exactly the
length-coupling I am keeping out of the seed. The mean also keeps the seed's scale `O(1)` regardless of
`T`, matching the length-invariance the convex-combination context already has downstream: `Σ_j α_j = 1`
and each `h_j` is `tanh`-bounded, so `c_i` cannot grow with the number of source positions — where the
discarded plain-encoder path forces all 160 bits through one saturating `h_T`, here length only changes
*which* annotations get weight, never the magnitude of what the decoder receives.

The three variants stress that alignment differently, and naming the target index for each makes the
`reverse`-is-hardest intuition precise. For `delim`, output position `k` aligns to source position `k` —
the identity map, a diagonal the softmax tracks by a constant offset. For `repeat` (`y = x` twice), `k`
aligns to `((k−1) mod L) + 1` — a sawtooth that resets at the seam of the two copies, so the alignment
must jump back to source index 1 exactly once, at `k = L+1`, a length-dependent event the model has to
detect. For `reverse`, `k` aligns to `L − k + 1` — a strictly *descending* diagonal, the one case where
the alignment moves opposite to the output index. All three are expressible by the same non-monotonic
scorer, but the descending and wrap patterns lean harder on the annotation encoding the source *length*,
and length is exactly what a fixed-width encoder represents least reliably out of range. So even before any
numbers I expect the OOD token accuracy to order `delim ≥ repeat ≥ reverse`.

There is a compounding-error arithmetic worth doing now, because it predicts the *shape* of the gap
between the two metrics. Greedy autoregressive exact match over an output of length `M` requires every one
of the `M` next-token predictions to be individually correct, so if per-token accuracy is roughly `p` and
errors were independent, sequence match would be about `p^M`. At the OOD ceiling `M = 40`, even a strong
`p = 0.95` gives `0.95^40 ≈ 0.13` and `p = 0.90` gives `0.90^40 ≈ 0.015` — essentially zero exact match
while the token column still reads 0.90. Errors are not truly independent and a copy model can
self-correct, so this is a bound-flavoured estimate, but its lesson is robust: exact match is a product
over the whole output and collapses fast, token accuracy is an average and degrades gently. So the two
columns will tell different stories on OOD.

At the training lengths every variant should be easy — the source is short, the bottleneck never bites,
and content-based soft alignment plus recurrence should give near-perfect in-distribution exact match. The
OOD split is the real test, and here I am genuinely uncertain. Recurrence and content-based attention have
no hard length cap the way the learned table does, so in principle they should generalise the *rule* of
copying. But the encoder's hidden state is still a fixed-width channel, so a longer source crowds more
content through the same annotations, and the decoder state, seeded from a single masked-mean summary,
accumulates drift over a longer output that no positional anchor corrects. So the falsifiable floor I
expect this rung to establish is perfect ID, zero OOD exact match, and non-trivial OOD token accuracy —
the signature of a model that has the right local alignment rule but cannot hold a long output together
without an explicit position signal, which is exactly the gap the Transformer rungs above will try to
close. (The full scaffold module is in the answer.)
