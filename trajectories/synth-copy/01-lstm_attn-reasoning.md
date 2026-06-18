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

Let me reconstruct why this method exists and what it actually computes, because the way it has to be
fit into this harness is not the textbook translation version. The disease the additive-attention
seq2seq cures is the fixed-length-context bottleneck of the plain encoder-decoder: an RNN reads the
source left to right, `h_t = f(x_t, h_{t-1})`, and the older design hands the decoder only the final
state `c = h_{T}` as a single fixed-width summary of the whole input. Everything the decoder will ever
need about the source has to be squeezed into that one vector before the first output symbol comes out,
and a constant-size vector cannot hold an unbounded amount of content — so the basic model degrades on
long inputs, worst beyond the lengths it trained on. The cure is to keep *all* the encoder's
per-position states `h_1, …, h_{T}` and let the decoder, at each output step, search the source for the
position it should be reading right now and pull a content-weighted blend from there. If the weights
`α_{i1}, …, α_{iT}` are nonnegative and sum to one, the decoder reads `c_i = Σ_j α_{ij} h_j`, a per-step
context recomputed every output step instead of one summary frozen at the start. That removes the
bottleneck: a long source is no harder than a short one because the decoder only ever reads one
position's worth of blended content at a time.

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
by the recurrence that builds the `h_j` and the `s_i`.

Now the part that matters most for this trajectory: how that method has to be expressed in *this* task's
edit surface, because it is not the generic translation network and I must derive the version the
harness actually runs. The scaffold gives me a single token stream
`[BOS] x_1 … x_T [SEP] y_1 … y_M [EOS]` and a fixed `forward(tokens) -> [B, T, V]` API; there is no
separate source and target tensor. So I have to manufacture the encoder-decoder split inside `forward`.
The `[SEP]` token is the seam: I find `sep_pos` per row as the position of `SEP_ID`, treat the content
between `BOS` and `SEP` as the source, and treat everything from `SEP` onward as the decoding region. I
embed the source slice, run a *bidirectional* LSTM encoder over it so each annotation `h_j` sees both
sides of position `j` (the forward direction summarises `x_1..x_j`, the backward direction `x_j..x_T`,
concatenated to width `d_model`), and mask out the padding positions so attention never reads them. The
decoder is an `LSTMCell` initialised from a `tanh` projection of the masked-mean encoder summary, and it
steps over *every* position of the stream, but I only let it advance — and only let it write logits —
once the position has reached or passed `sep_pos`. Concretely a per-row boolean `active = pos >= sep_pos`
gates the state update (`h, c` are held frozen on inactive positions) and the logit write, so the model
produces predictions only on the target region, which is exactly where the loss mask lives.

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
