I start at the architecture with the cleanest claim to *actually counting*. The question is about
counters, so the first thing I want to know is how far a recurrent net gets on its own: if a recurrence
can hold an integer count across the sequence, that is the honest floor everything later has to beat; if
it cannot, I learn precisely *where* recurrence breaks, and that diagnosis seeds the next model I try.

The recurrent options are a vanilla `nn.RNN`, the scaffold's default 2-layer GRU, an LSTM, or a
hand-wired non-parametric accumulator. The accumulator is out on the contract: the editable slot returns
a learnable `nn.Module` that the fixed AdamW loop trains from initialization, so a frozen counter is not
in the design space — and it would teach me nothing about what a *learned* recurrence can do, which is
the whole point of the floor. The live choice is GRU versus LSTM, because the GRU is the default and I
have to justify overriding it. A GRU carries state as a convex interpolation,
`h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t`, with the candidate `h̃_t` squashed through `tanh`; there is no
separate, *unsquashed* memory line. A running tally needs to live somewhere that (a) is not re-compressed
through a saturating nonlinearity every step and (b) passes gradient at unit gain, and the GRU's held
state is the same state the squash touches — the moment the update gate opens to write an increment, the
new content re-enters through `tanh`. The LSTM separates the two: a linear cell `c_t` written additively,
and a *separate* output `h_t = o_t ⊙ tanh(c_t)` that squashes only the read-out. So it can hold an
undamped linear tally in `c_t` while exposing a bounded view — the exact separation a counter wants, and
the reason to override the default GRU rather than keep it.

The reason a plain recurrence is the wrong place to *stop* is the gradient. Backpropagating an error
through `h_t = f(h_{t-1}, x_t)` over `q` steps multiplies it by a product of `q` Jacobian factors, each
(derivative of the squash)×(recurrent weight); for sigmoid/tanh units each factor is below one, so the
product decays geometrically — the vanishing gradient. At a generous 0.8 per factor, 30 steps gives
`0.8^30 ≈ 1e-3` and 60 steps `≈ 1e-6`; the credit is gone. On `abc` the lag is exactly what matters: to
decide `a^n b^n c^n` the model compares the `a`-block at the front against the `c`-block at the back,
dozens of steps apart for the longer training strings. A vanilla RNN never threads a count that far; it
learns local texture. That is the disease the gated cell was built to cure — its linear cell state rides
a self-loop at unit gain, so the backpropagated state error is multiplied by the forget gate per step
(`ε_s^t = … + f_{t+1} ⊙ ε_s^{t+1}`, unit-gain when the forget gate is open), and the three gates decide
from context when to write, hold, and read. The forward dynamics `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`,
`h_t = o_t ⊙ tanh(c_t)` are a programmable accumulator: input gate opens on `a`, candidate contributes
`+δ`, forget gate near one holds the tally undamped, and a later contribution checks it against the `b`
and `c` blocks. The counting recipe is a fixed point of the gate settings, not a length-specific table —
which is why in principle the cell can implement an integer counter that even extrapolates.

That is the optimistic reading, and the read-out is where I expect it to break. The head never sees
`c_t`; it sees `h_t = o_t ⊙ tanh(c_t)`, a squashed view. Suppose the clean recipe: increment `c` by `δ`
on each `a`. To keep successive counts resolvable the gap `tanh(δ(k+1)) − tanh(δk)` must stay above
noise, and `tanh` flattens hard — `tanh(2)=0.96`, `tanh(3)=0.995`, `tanh(4)=0.9993` — so once `δk` passes
~3 the read-out pins near 1 and consecutive counts are indistinguishable. In-range the largest count is
~21 (the `a`-block of the longest training `abc` string) or ~63 (all-`a` `exact` strings), pushing the
network toward a small step, `δ ≈ 0.1`, putting count 21 at `tanh(2.1)=0.97` — compressing but workable.
At OOD length 256 the counts reach ~85, and `tanh(8.5) ≈ 0.9999997`: every large count collapses onto the
same saturated value, so the read-out cannot tell 40 from 85, arithmetically, *independent* of whether
the cell integrated perfectly. And a learned increment of `0.99δ` instead of `δ` compounds linearly: a 1%
drift over 21 steps is 4% over 85. Both mechanisms grow with length. So my strongest a-priori prediction
is not "the LSTM fails" but a *split*: excellent in-range, sharp degradation — plausibly to the floor —
once counts leave the trained band. A finer corollary falls out of the same picture: the read-out
fragility is a *magnitude* problem, and an *equality* test survives saturation where a magnitude read
does not — two blocks of equal count both saturate to the same `tanh` value together, but `tanh(2δ)`
versus `tanh(3δ)` for two different counts do not stay apart. So if anything survives OOD it should be
the equality-structured `abc` membership decision rather than the absolute `exact` count.

Capacity is not the binding constraint. Embedding is
`5 × 128 = 640`; one `nn.LSTM` layer at width 128 holds four gate matrices input-to-hidden and again
hidden-to-hidden, `2 × 4 × 128 × 128 ≈ 131k` weights, so two layers ≈ 264k, plus a `LayerNorm(128)` —
about 0.27M, ~5% of the 5M ceiling. I could go to width 256 or four layers and stay inside budget, but the
failure I expect is *representational* — a squashed tally that does not survive length doubling — and
neither more width nor more depth changes that: a wider `tanh` still saturates, a deeper stack still
carries the count through time. Two layers is the minimum that buys the one thing depth helps here,
division of labor: the lower layer carries the tallies, the upper combines them into the `a==b==c`
comparison, rather than one state doing both. So I spend the minimum on capacity and keep the comparison
against later models honest at matched width and depth — whatever the OOD numbers show is attributable to
the encoder, not to spent parameters.

Now the encoder, fit to the contract. Embedding `nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)`
maps the five symbols into 128-dim with the padding row pinned to zero, turning `[B,T]` into `[B,T,128]`;
then `nn.LSTM(128, 128, num_layers=2, batch_first=True)` gives the top-layer hidden at every step,
`[B,T,128]`, width matched to `hidden_dim` so no projection is needed. Pooling must collapse time to
`[B,128]`, and the one subtlety is *where* to read. The CLS token sits at position 0, *before* any
content, and an LSTM is causal: its state at position 0 has seen only CLS, so reading CLS would summarise
an empty sequence. The state that has seen the whole string is the one at the last valid position, so I
gather at `lengths - 1` — `(lengths-1).clamp(min=0).view(-1,1,1).expand(-1,1,128)` then
`gather(1,·).squeeze(1)`, the clamp guarding the degenerate CLS-only row. Getting this wrong would feed
the head a hidden that never saw the `c`-block, capping `abc` at chance no matter how good the cell. A
final `nn.LayerNorm(hidden_dim)` stabilizes the pooled scale into the fixed head, which matters because a
length-8 `exact` string and a length-256 OOD string would otherwise hand the single shared head summaries
on scales that grow with length.

One more reason recurrence is the right floor, from the *structure of the negatives*: the `abc` generator
draws not just wrong-block-count negatives but swapped order and interleaved middles. A pure bag-of-counts
feature cannot separate `a^n c^n b^n` from `a^n b^n c^n` — same counts, different order — so the encoder
must be order-sensitive, and a left-to-right recurrence is order-sensitive by construction: the cell that
opened its input gate on `a` then expected `b` then `c` lands in a different trajectory the moment the
order is wrong. That favors *starting* recurrent over jumping straight to a permutation-agnostic model
that would have to be told about order separately.

This LSTM fills only the *encoder* slot: embedding, 2-layer LSTM, last-position read, LayerNorm. The
optimizer, loss, head, and data are the fixed harness, so the usual sequence-to-one training knobs —
early stopping, masked-MSE, custom clipping — are not mine to set; I reason only about the encoder.

Per environment, then: high in-distribution accuracy on both `abc` and `exact`, but the `exact` OOD
score — which demands the rounded count be exactly right at length 256 — plausibly collapses to the
floor, and `length-ood` retention suffers because the in-range counter does not transfer to twice the
length. That split would say the recurrent counter is *memorized in-range* rather than *implemented as
an exact integer recurrence*, so the next model has to change the representation of length itself rather
than the way the count is carried.