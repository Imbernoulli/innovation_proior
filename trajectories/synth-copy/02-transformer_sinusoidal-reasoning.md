The recurrent baseline came back with the exact shape I predicted, and the numbers tell me precisely
what to fix. Across all three variants the in-distribution split is perfect — `exact_match_id = 1.0`,
`token_acc_id = 1.0` — so the soft-alignment recurrence learns the copying rule flawlessly at the
lengths it trained on. But the OOD split is a wall: `exact_match_ood = 0.0` on every variant, and the
OOD token accuracy is low and decaying with task difficulty — `0.371` on `delim`, `0.207` on `repeat`,
`0.180` on `reverse`. Read that gradient: `delim` (the source index maps straight through) holds the
most tokens, `repeat` (the alignment must wrap modulo the source length) holds fewer, and `reverse` (the
alignment must run backwards) holds the fewest. The score is pinned at `0.5` on all three because it is
exactly half ID and half OOD, and OOD exact match is zero. This is the signature I flagged at rung one: a
model with the right *local* alignment rule but no way to hold a long output together. The decoder state
is seeded from a single masked-mean summary and accumulates drift over 30–40 output steps with no
positional anchor to correct it, so one symbol slips somewhere and the sequence-level match fails even
while most tokens are still right. And it is slow — 400–500 seconds per variant, because the recurrence
is sequential and the per-step alignment is an extra matmul at every one of the `T` positions. So the
diagnosis is sharp: recurrence carries order *implicitly*, and implicit order does not survive being
stretched past the training length. The cure has to be an *explicit* position signal, and the natural
substrate for it is the decoder-only Transformer the rest of the ladder shares — which also kills the
sequential-recurrence cost, since attention parallelises over positions.

But the Transformer has its own problem before it can even start, and I have to face it head-on because
it is the whole reason a positional scheme exists. Self-attention is a function of a *set* of key/value
vectors. Take the attention output at one position: it is `softmax(q·k_j/√d)` weights times the values
`v_j`, summed over `j`. The query is a linear function of the token at my position; each key and value
is a linear function of the token at position `j`. Nowhere does the index `j` appear — only the contents
of the two token vectors. Permute the input tokens and every `k_j, v_j` is just relabelled, the set is
identical, the softmax is over the same scores, the weighted sum is the same number. The feed-forward
sublayer acts per position and cannot see neighbours either. So a stack of these reads its input as a
*bag* of token vectors: `a b` and `b a` are the same object. (The causal mask does break some of this —
I will come back to that at a later rung — but the default move, and the one I am taking now, is to put
order back in by hand.) For the recurrent baseline I never had to, because the recurrence supplied order;
moving to attention, I have to manufacture a position signal explicitly. That is exactly the lever the
task hands me, and the cleanest first explicit scheme is the one that started the Transformer line:
sinusoidal absolute encoding.

So what should the per-position signal be? Operationally, the number "this token is at position `t`" has
to enter the computation in a form the existing machinery — linear layers, dot products, softmax — can
consume, and all of that eats `d_model`-dimensional vectors. So I manufacture, for each position `t`, a
vector `p_t ∈ ℝ^{d_model}` that *is* the position, and fuse it with the token embedding at the bottom of
the stack. Add, do not concatenate: the first linear layer on a concatenation `[embed; p]` is
`W_a·embed + W_b·p`, which is exactly applying one matrix to the embedding and another to the position
and summing — so concatenation just buys the position a private projection at the cost of extra width
everywhere above, whereas plain addition `embed + p` still lets the first projection read a linear
function `W·p` of the position and widens nothing. The model carves out a subspace of the `d_model`
dimensions to hold position and reads it off. The scaffold's `token_embedding_extra(positions)` hook is
precisely this: it returns an additive `[B, T, D]` term, and `SeqModel.forward` adds it to the token
embedding before layer one. So my whole job is to define what vector that hook returns.

Now, what is `p_t`? The dumbest try is to stuff the raw integer `t` into a channel. But `t` is unbounded:
train on lengths to 20, run at 40, and the model is fed position values it never saw, large-magnitude
inputs that push the downstream linears into uncalibrated regimes — the activations blow up. Normalising
to `t/L` bounds it but breaks the meaning of a step: in a length-10 sequence one token back is a delta of
`0.1`, in a length-100 sequence it is `0.01`, so "look one position back" is no longer a fixed quantity
the model can learn a single rule for. I want something bounded *and* shift-consistent — where a fixed
positional step always advances the code by the same amount, regardless of absolute position or sequence
length. A periodic function gives both: `sin(ω t)` lives in `[-1, 1]` for all `t` and a step of `Δt`
always advances the phase by `ω Δt`. But one sinusoid aliases — it repeats every period, and a single
scalar cannot separate hundreds of positions. The fix is many frequencies at once, the continuous
analogue of a binary counter: the low bits flip fast (fine local position), the high bits flip slowly
(coarse global position), and the *combination* separates an enormous range while each digit stays
bounded. So I use a whole vector of sinusoids at geometrically spaced frequencies
`ω_i = 10000^{-2i/d}`, `i = 0 … d/2−1` — the fastest with wavelength `2π`, the slowest near
`10000·2π` — geometric so the wavelengths tile the scale axis evenly in log-space, giving roughly equal
resolution from local to global.

There is one more thing the encoding has to give me, and it is the property that makes it more than a
bounded counter: a fixed relative shift should be a fixed linear map. If I stored only `sin(ω t)` per
dimension, I could not recover `sin(ω(t+k))` from it — the angle-addition formula needs `cos(ω t)`,
which I threw away, and at a given value of `sin` I do not know whether the phase is rising or falling.
The cure is to keep both `sin(ω t)` and `cos(ω t)` for each frequency — store the full phase as a point
on the unit circle. Then the shift `t → t+k` is, for each frequency, the orthogonal rotation by `ω k`,
and stacking those rotations block-diagonally gives `p_{t+k} = M_k p_t`, a single linear transformation
that depends only on the offset `k` and is identical at every absolute `t`. That is why the layout pairs
sine with cosine per frequency and interleaves them: even dimensions `PE(t,2i)=sin(t·ω_i)`, odd
dimensions `PE(t,2i+1)=cos(t·ω_i)`. And it is why I can hope for extrapolation at all — `sin` and `cos`
are defined for every real `t`, so position 30 is not a missing table slot the way the scaffold default's
learned table is; it is just the next point along the same curves, and the rotation relation still holds
there with the same `M_k`. That last point is the direct answer to rung one's failure mode: where the
recurrence had no anchor and the learned table had no entry, the sinusoid has a closed-form value
everywhere.

Now I have to be careful and honest, because the methods derivation of this scheme already names the
catch, and the catch is exactly what I will be testing. "Defined everywhere" is *not* the same as "the
model can use it everywhere." During training the attention learns to interpret the particular phase
combinations that occur over `[0, 20)`. Past 20, the joint pattern of all those sinusoids across the
dimensions is a configuration the model never encountered — it is out of distribution. The encoding
function does not die, but the model's *learned interpretation* of it only covers the training range. So
I am genuinely unsure this beats rung one on OOD exact match; what I am confident of is that it parallels
the loop (no sequential recurrence) and that it gives the model a real positional code instead of an
implicit one. The honest expectation is that this is a *diagnostic* rung: if even a closed-form,
extrapolation-friendly absolute code fails OOD, that is strong evidence the problem is *absoluteness
itself*, not the particular code — which is the thread the next rungs pull.

Let me make this concrete in the edit surface, and note exactly how the harness version differs from the
canonical translation-era scheme. I fill `build_positional_scheme` to build the `[max_total_len, d_model]`
sinusoidal table in closed form — `pe[:, 0::2] = sin(positions·div_term)`,
`pe[:, 1::2] = cos(positions·div_term)`, with `div_term = exp(arange(0,d,2)·(−ln 10000 / d))` computed
in log space rather than by repeated `pow` — wrap it in a *frozen* `nn.Embedding.from_pretrained(pe,
freeze=True)` (no gradient, no learnable parameter), register it through `scheme.extra_modules` so it
moves to the GPU, and return only the `token_embedding_extra` hook; `attn_bias` and `rotary` stay `None`.
`build_model` returns the plain `SeqModel` with `use_lstm=False`, so the decoder-only backbone is back
and the recurrence is gone. Two harness-specific differences from the textbook sinusoidal recipe matter.
First, there is **no `√d_model` embedding scaling** here — the canonical scheme multiplies the token
lookup by `√d_model` to put content and position on comparable scales before the sum, but this scaffold's
`SeqModel` adds `token_embedding_extra(positions)` straight onto the raw `token_embed` output, so I do not
control that scaling and do not introduce it; the learned token embedding simply has to grow on its own
to balance the `O(1)` sinusoids. Second, there is **no dropout** on the sum (the task fixes
`dropout = 0.0`), unlike the original which applies dropout to the embedding-plus-position. And the
positions fed in are `[0, T)` over the *whole* stream `[BOS] x … [SEP] y … [EOS]`, not a separate
source/target indexing — the table is indexed by absolute stream position with a `clamp` to the table's
last row as a safety bound (the table is sized to `max_total_len = 256`, comfortably past `2·L_train`, so
the clamp never actually fires on this task's lengths). Nothing else in the editable block changes.

So the delta from rung one is precise: where the LSTM carried order through a sequential recurrence with
no explicit code, I now run the parallel decoder-only Transformer and stamp every token with a frozen
sinusoidal absolute position before layer one. My falsifiable expectations against the rung-one numbers:
in-distribution should stay perfect (`exact_match_id ≈ 1.0` on all three variants — the absolute code is
more than enough at training length), and the elapsed time should drop sharply from the LSTM's 400–500 s
to roughly the other Transformer rungs' ~130 s, because attention parallelises. The open question is OOD,
and I have a specific prediction: if absoluteness is the disease, sinusoidal should *not* rescue OOD exact
match — I expect `exact_match_ood` to stay at or near `0.0` on all three variants, and possibly with OOD
token accuracy even *worse* than the LSTM's `0.37/0.21/0.18`, because the out-of-range phase patterns are
actively misleading the attention rather than merely under-anchoring it. If that comes true — perfect ID,
zero OOD, and OOD token accuracy at or below rung one — then the lesson is unambiguous and sets up the
next rung: the failure is not "which absolute code," it is *absolute position at all*, and the move is to
stop prescribing an absolute code and let the model build a relative one. (The full scaffold module is in
the answer.)
