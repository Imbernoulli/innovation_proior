The recurrent baseline came back with the exact shape I predicted. Across all three variants the
in-distribution split is perfect — `exact_match_id = 1.0`, `token_acc_id = 1.0` — so the soft-alignment
recurrence learns the copying rule flawlessly at training length. But OOD is a wall: `exact_match_ood =
0.0` on every variant, with OOD token accuracy low and decaying with difficulty — `0.371` on `delim`,
`0.207` on `repeat`, `0.180` on `reverse`. That gradient is the alignment story I expected: `delim` (source
index maps straight through) holds the most tokens, `repeat` (alignment wraps modulo the source length)
fewer, `reverse` (alignment runs backwards) the fewest. Taken relative to `delim`, `repeat` retains
`0.207/0.371 = 0.56` and `reverse` `0.180/0.371 = 0.49` — a roughly two-to-one falloff from easiest to
hardest alignment, exactly the `delim ≥ repeat ≥ reverse` ordering I argued from how much each variant's
target index depends on knowing the source length at an unseen scale. And all three sit well above the
16-symbol chance level of `1/16 = 0.0625`, so the recurrence did learn a real local alignment rule that
partially transfers — it places most tokens correctly at length 30–40, just not all of them. Meanwhile
every OOD exact match is a hard `0.0`, which the product-over-output arithmetic from rung one explains at a
glance: a per-token `0.37` over a 40-token output predicts `0.37^40`, a number with seventeen leading
zeros. So the wall is not "the model cannot copy"; it is "the model cannot copy *without a single slip*
across an output twice as long as any it trained on." The decoder state is seeded from a single masked-mean
summary and drifts over 30–40 output steps with no positional anchor to correct it. The missing ingredient
is a per-position anchor, and it has to be an *explicit* position signal — implicit, recurrence-carried
order does not survive being stretched. The natural substrate is the decoder-only Transformer the rest of
the ladder shares, which also kills the sequential-recurrence cost (400–500 s/variant), since attention
parallelises over positions.

But the Transformer has its own problem before it can start, and it is the whole reason a positional
scheme exists. Self-attention is a function of a *set* of key/value vectors: the output at one position is
`softmax(q·k_j/√d)` weights times the values `v_j`, summed over `j`. The query is a linear function of my
token; each key and value is a linear function of the token at position `j`. Nowhere does the index `j`
appear. Permute the input and every `k_j, v_j` is relabelled, the set is identical, the weighted sum is
the same number; the feed-forward sublayer acts per position and cannot see neighbours either. So a stack
reads its input as a *bag* of token vectors — `a b` and `b a` are the same object. (A causal mask is not
perfectly symmetric, so it dents this — but the default move is to put order back in by hand, which is what
the recurrence did for me for free at rung one.) The cleanest first explicit scheme is the one that started
the Transformer line: sinusoidal absolute encoding.

Sinusoidal and not the scaffold's learned table, and the choice is not arbitrary. The learned table holds
one trainable vector per slot up to `max_total_len`; while slots for positions `21…40` physically exist in
a table sized to 256, they receive *zero gradient* during training because no training sequence ever
reaches them — every target-region slot past the training envelope is an untrained `N(0, 0.02)` vector,
literally random noise the optimiser never touched. The sinusoid replaces those untrained slots with a
deterministic closed-form value, the same curve one step further along. If I am going to test whether
*absoluteness itself* is the problem, I have to test it with the strongest absolute code available, so a
failure cannot be blamed on "you used the weak absolute scheme." Sinusoidal is that code.

So what should the per-position signal be? Operationally, "this token is at position `t`" has to enter in a
form the linear layers, dot products, and softmax can consume — all of which eat `d_model`-dimensional
vectors. So I manufacture, for each `t`, a vector `p_t ∈ ℝ^{d_model}` and fuse it with the token embedding
at the bottom of the stack. Add, do not concatenate: the first linear layer on `[embed; p]` is
`W_a·embed + W_b·p`, exactly one matrix on the embedding and another on the position, summed — so
concatenation just buys the position a private projection at the cost of extra width everywhere above,
whereas plain addition `embed + p` still lets the first projection read a linear function `W·p` and widens
nothing. The scaffold's `token_embedding_extra(positions)` hook is precisely this additive term, added to
the token embedding before layer one, so my whole job is to define what vector it returns.

Now, what is `p_t`? Stuffing the raw integer `t` into a channel fails — `t` is unbounded, so training to 20
and running at 40 feeds the downstream linears large-magnitude values they never saw and the activations
blow up. Normalising to `t/L` bounds it but breaks the meaning of a step: one token back is `0.1` in a
length-10 sequence and `0.01` in a length-100 one, so "look one position back" is no longer a fixed
quantity. I want something bounded *and* shift-consistent, where a fixed step always advances the code by
the same amount. A periodic function gives both: `sin(ω t)` lives in `[-1, 1]` and a step of `Δt` advances
the phase by `ω Δt`. But one sinusoid aliases, and a single scalar cannot separate hundreds of positions.
The fix is many frequencies at once, the continuous analogue of a binary counter: low bits flip fast (fine
local position), high bits flip slowly (coarse global position), and the combination separates an enormous
range while each digit stays bounded. So I use a vector of sinusoids at geometrically spaced frequencies
`ω_i = 10000^{-2i/d}`, `i = 0 … d/2−1` — geometric so the wavelengths tile the scale axis evenly in
log-space, giving roughly equal resolution from local to global.

The frequency counts turn out to predict exactly where extrapolation will hurt. For `d = 128` there are
`d/2 = 64` frequencies. A frequency whose wavelength exceeds the training range 20 never completes even one
cycle in `[0, 20)`, so its phase there is a single monotone arc, and at position 40 that arc continues into
phase values that never occurred in training. The threshold is `2π/ω_i > 20`, i.e. `ω_i < 2π/20 ≈ 0.314`;
solving `10000^{-i/64} < 0.314` gives `(i/64)(9.21) > 1.158`, so `i > 8.05` — frequencies `i = 9 … 63`, a
full `55 of the 64`, never complete a cycle in the training window. For those 55 dimensions the phase at
length 40 is a value the attention never observed. Only the fastest `~9` frequencies cycle enough to repeat
their phase patterns into the OOD range, and those are the *local* ones. So the coarse, long-range
positional dimensions — the ones a length-40 sequence most needs to distinguish its far positions — are
exactly the ones that go out of distribution at evaluation. Bounded and closed-form does not save them,
because "the model has seen this phase configuration" is the property that matters, and for 55 of 64
channels it has not.

There is one more property the encoding must give, and it is what makes it more than a bounded counter: a
fixed relative shift should be a fixed linear map. If I stored only `sin(ω t)`, I could not recover
`sin(ω(t+k))` — the angle-addition needs `cos(ω t)`, and at a given `sin` I do not know whether the phase
is rising or falling. So I keep both `sin(ω t)` and `cos(ω t)` per frequency, the full phase as a point on
the unit circle. Then the shift `t → t+k` is, for each frequency, rotation by `ω k`, and stacking those
rotations block-diagonally gives `p_{t+k} = M_k p_t`, a single linear map depending only on the offset `k`,
identical at every absolute `t`. This is why the layout pairs sine with cosine and interleaves them —
`PE(t,2i)=sin(t·ω_i)`, `PE(t,2i+1)=cos(t·ω_i)` — and why I can hope for extrapolation at all: `sin` and
`cos` are defined for every real `t`, so position 30 is not a missing table slot, it is the next point on
the same curves, and the same `M_k` still holds there. A single dense `M_k` can carry "shift by `k`," which
is exactly what a copy/reverse alignment needs — attend to the token a fixed offset away — so an attention
head *could* in principle learn a query/key geometry that implements "look `k` back" as one linear relation.

But I have to be honest, because "defined everywhere" is *not* "the model can use it everywhere," and that
gap is exactly what this rung tests. During training the attention learns to interpret the particular phase
combinations that occur over `[0, 20)`. Past 20 the joint pattern across the dimensions is a configuration
never encountered — out of distribution. The encoding function does not die, but the model's *learned
interpretation* only covers the training range, and I just counted 55 of 64 channels whose phase goes
novel there. So I am genuinely unsure this beats rung one on OOD exact match. What I am confident of is that
it parallels the loop and gives the model a real positional code instead of an implicit one. The honest
role of this rung is diagnostic: if even a closed-form, extrapolation-friendly absolute code fails OOD,
that is strong evidence the problem is *absoluteness itself*, not the particular code.

In the edit surface, `build_positional_scheme` builds the `[max_total_len, d_model]` sinusoidal table in
closed form — `pe[:, 0::2] = sin(positions·div_term)`, `pe[:, 1::2] = cos(positions·div_term)` with
`div_term = exp(arange(0,d,2)·(−ln 10000 / d))` in log space rather than repeated `pow` — wraps it in a
*frozen* `nn.Embedding.from_pretrained(pe, freeze=True)` (no gradient, no learnable parameter), registers
it via `scheme.extra_modules` so it moves to the GPU, and returns only the `token_embedding_extra` hook;
`attn_bias` and `rotary` stay `None`. `build_model` returns the plain `SeqModel` with `use_lstm=False`, so
the decoder-only backbone is back and the recurrence is gone. Two harness-specific differences from the
textbook recipe matter. First, there is **no `√d_model` embedding scaling**: the canonical scheme
multiplies the token lookup by `√d_model ≈ 11.3` to keep content dominant over the bounded `[-1,1]`
sinusoids, but this scaffold adds `token_embedding_extra(positions)` straight onto the raw `token_embed`
output, so at initialisation an `N(0, 0.02)`-scale token vector would be swamped by position two orders of
magnitude over — and gradient descent has to grow the token-embedding norm until content and the fixed
position term are comparable, all the adaptation on the embedding side. That is one more reason the scheme
is tuned to the `[0, 20)` regime: the embedding scale the optimiser settles on is calibrated against the
phase statistics it saw. Second, there is **no dropout** on the sum (the task fixes `dropout = 0.0`). And
positions are `[0, T)` over the whole stream, indexed by absolute stream position with a `clamp` to the
table's last row as a safety bound — the table is sized to `max_total_len = 256`, comfortably past
`2·L_train`, so the clamp never fires here. Nothing else in the editable block changes.

On the clock, the Transformer replaces rung one's chain of `~83` dependent kernel launches with a *single*
batched attention: all `T` query positions attend to all `T` keys in one `[T, T]` score matrix, `O(T² d)`
but one fully parallel kernel per layer, four layers, no sequential dependency. `T² = 83² ≈ 6900` is
trivial for the GPU, and the sinusoidal add is a single `[B, T, d]` table lookup once at the bottom. So I
expect elapsed to fall sharply from the LSTM's 400–500 s to a couple of minutes per variant.

My falsifiable expectations against the rung-one numbers: in-distribution stays perfect (`exact_match_id ≈
1.0` on all three), and elapsed drops sharply. The open question is OOD, and my prediction is specific: if
absoluteness is the disease, sinusoidal should *not* rescue OOD exact match — I expect `exact_match_ood` at
or near `0.0` on all three, and possibly OOD token accuracy even *worse* than the LSTM's `0.37/0.21/0.18`,
because the out-of-range phase patterns actively mislead the attention rather than merely under-anchor it.
If that comes true, the lesson is unambiguous: the failure is not "which absolute code," it is *absolute
position at all* — an index keyed over `[0, 20)` has no counterpart over `[21, 40)`, and no sharpening of
the absolute code can conjure one. (The full scaffold module is in the answer.)
