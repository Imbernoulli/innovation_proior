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

Before I leave rung one's numbers I want to read them harder, because the exact ratios carry a mechanism
I can use. The OOD token accuracies are `0.371 / 0.207 / 0.180` on `delim / repeat / reverse`. Take them
relative to `delim`: `repeat` retains `0.207 / 0.371 = 0.56` of the identity-map accuracy, and `reverse`
retains `0.180 / 0.371 = 0.49` — roughly a two-to-one falloff from the easiest to the hardest alignment.
That ordering is exactly the `delim ≥ repeat ≥ reverse` I predicted from how much each variant's target
index depends on knowing the source length at an unseen scale, and the fact that all three sit *well above*
the 16-symbol chance level of `1/16 = 0.0625` tells me something specific: the recurrence did learn a real
local alignment rule that partially transfers: it is placing most tokens correctly even at length 30–40,
just not all of them. Meanwhile every OOD exact match is a hard `0.0`. Put those two facts together with
the product-over-output arithmetic from rung one — sequence match behaves like per-token accuracy raised
to the output length — and a per-token `0.37` over a 40-token output predicts `0.37^{40}`, a number with
seventeen leading zeros: exact match is not merely low, it is annihilated. So the wall is not "the model
cannot copy"; it is "the model cannot copy *without a single slip* across an output twice as long as any
it trained on," and the missing ingredient is a per-position anchor that stops the slow drift. That is the
precise hole an explicit positional code exists to fill.

But the Transformer has its own problem before it can even start, and I have to face it head-on because
it is the whole reason a positional scheme exists. Self-attention is a function of a *set* of key/value
vectors. Take the attention output at one position: it is `softmax(q·k_j/√d)` weights times the values
`v_j`, summed over `j`. The query is a linear function of the token at my position; each key and value
is a linear function of the token at position `j`. Nowhere does the index `j` appear — only the contents
of the two token vectors. Permute the input tokens and every `k_j, v_j` is just relabelled, the set is
identical, the softmax is over the same scores, the weighted sum is the same number. The feed-forward
sublayer acts per position and cannot see neighbours either. So a stack of these reads its input as a
*bag* of token vectors: `a b` and `b a` are the same object. (A causal mask is not perfectly symmetric,
so it dents this a little — but the default move, and the one I am taking now, is to put order back in by
hand.) For the recurrent baseline I never had to, because the recurrence supplied order;
moving to attention, I have to manufacture a position signal explicitly. That is exactly the lever the
task hands me, and the cleanest first explicit scheme is the one that started the Transformer line:
sinusoidal absolute encoding.

I should say why sinusoidal and not the other absolute code on the table, because the choice is not
arbitrary and I want to have argued it. The two absolute options are the scaffold's learned table and the
closed-form sinusoid. The learned table is the default fill, and I can reject it at this rung on the same
inspection ground as before: it holds one trainable vector per slot up to `max_total_len`, and while slots
for positions `21…40` do physically exist in a table sized to 256, they receive *zero gradient* during
training because no training sequence ever reaches those positions — content lengths are drawn from
`[1, 20]`, so the longest stream is `1 + 20 + 1 + 40 + 1` at most for `repeat`, but crucially the source
region a copy head must key on never extends past position ~21, and every target-region slot past the
training envelope is an untrained `N(0, 0.02)` vector. So a learned absolute table does not merely
generalise poorly out of range; the relevant far slots are literally random noise the optimiser never
touched. The sinusoid is the strictly better absolute code precisely because it replaces those untrained
slots with a deterministic closed-form value — the same curve evaluated one step further along. If I am
going to test whether *absoluteness itself* is the problem, I have to test it with the strongest absolute
code available, so that a failure cannot be blamed on "you used the weak absolute scheme." Sinusoidal is
that strongest absolute code, which is why it is the right first explicit rung.

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

Let me put numbers on this frequency ladder for `d = 128`, because the counts turn out to predict exactly
where extrapolation will hurt. There are `d/2 = 64` frequencies. The fastest, `ω_0 = 10000^0 = 1`, has
wavelength `2π ≈ 6.28` tokens, so it completes about `20 / 6.28 ≈ 3.2` full cycles across the training
range `[0, 20)`. The slowest, `ω_{63} = 10000^{-126/128} = 10000^{-0.984} ≈ 1.2·10^{-4}`, has wavelength
`2π / 1.2·10^{-4} ≈ 5.2·10^4` tokens — it barely moves off zero across the whole task. Now ask the
question that actually governs OOD: for how many of these 64 frequencies does the phase at an evaluation
position 40 land in a range the model already saw during training over `[0, 20)`? A frequency whose
wavelength exceeds 20 never completes even one cycle in training, so its phase over `[0, 20)` is a single
monotone arc; at position 40 that arc continues into phase values that never occurred in training. The
threshold is `2π / ω_i > 20`, i.e. `ω_i < 2π/20 ≈ 0.314`. Solving `10000^{-i/64} < 0.314` gives
`(i/64)·ln 10000 > -ln 0.314`, i.e. `(i/64)(9.21) > 1.158`, so `i > 8.05` — frequencies `i = 9 … 63`, a
full `55 of the 64`, never complete a cycle in the training window. For those 55 dimensions the phase at
length 40 is literally a value the attention never observed while learning to read this code. Only the
fastest `~9` frequencies cycle enough to repeat their phase patterns into the OOD range, and those are the
*local* ones. So the arithmetic already whispers the verdict: the coarse, long-range positional dimensions
— the ones a length-40 sequence most needs to distinguish its far positions — are exactly the ones whose
values go out of distribution at evaluation. Bounded and closed-form does not save them, because
"the model has seen this phase configuration" is the property that actually matters, and for 55 of 64
channels it has not.

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

Let me verify the rotation claim explicitly on one frequency block, because it is the load-bearing algebra
and I want to see it, not assert it. For frequency `ω`, the two coordinates hold `[sin ωt, cos ωt]`. The
angle-addition identities give `sin ω(t+k) = sin ωt cos ωk + cos ωt sin ωk` and
`cos ω(t+k) = cos ωt cos ωk − sin ωt sin ωk`. Stack those: `[sin ω(t+k), cos ω(t+k)]` equals the matrix
`[[cos ωk, sin ωk], [−sin ωk, cos ωk]]` times `[sin ωt, cos ωt]`. That `2×2` matrix is exactly the planar
rotation by angle `ωk`; it depends only on the offset `k`, not on `t`, and it is orthogonal so it
preserves norm. Block-diagonally stacking these across the 64 frequencies gives the promised
`p_{t+k} = M_k p_t` with a single `k`-dependent, `t`-independent linear map. This is genuinely more than a
counter: a single dense matrix `M_k` can carry "shift by `k`," which is precisely the operation a
copy/reverse alignment needs — attend to the token a fixed offset away — so an attention head *could*, in
principle, learn a query/key geometry that implements "look `k` back" as one linear relation. The catch,
which the next paragraph makes precise, is that whether the model *learned* to use `M_k` at the offsets
that only appear past length 20 is a different question from whether `M_k` is mathematically available
there. The algebra holds at every real `t`; the training signal does not.

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
to balance the `O(1)` sinusoids. It is worth being concrete about that balance, because it is a real
consequence of the harness rather than a cosmetic omission. The canonical recipe multiplies the token
lookup by `√d_model = √128 ≈ 11.3` before adding position, so a token embedding initialised at unit scale
is lifted to `~11×` the amplitude of the bounded `[-1, 1]` sinusoids, keeping content dominant and
position a modest additive hint. Here there is no such multiply, so at initialisation an `N(0, 0.02)`-ish
token vector has per-coordinate magnitude `~0.02` while each sinusoid coordinate swings in `[-1, 1]` —
position would *swamp* content by roughly two orders of magnitude if training did nothing. What actually
happens is that gradient descent grows the token embedding norm until content and the fixed-scale position
term are comparable; the frozen table cannot move, so all the adaptation is on the embedding side. This is
fine for learning at training length, but it is one more reason the scheme's behaviour is tuned to the
`[0, 20)` regime: the embedding scale the optimiser settles on is calibrated against the phase statistics
it saw, and those statistics shift out of range for the 55 coarse frequencies I counted above. Second, there is **no dropout** on the sum (the task fixes
`dropout = 0.0`), unlike the original which applies dropout to the embedding-plus-position. And the
positions fed in are `[0, T)` over the *whole* stream `[BOS] x … [SEP] y … [EOS]`, not a separate
source/target indexing — the table is indexed by absolute stream position with a `clamp` to the table's
last row as a safety bound (the table is sized to `max_total_len = 256`, comfortably past `2·L_train`, so
the clamp never actually fires on this task's lengths). Nothing else in the editable block changes.

The clock deserves one explicit comparison, since it is the second reason I moved off recurrence and I
should predict its size. Rung one ran a Python-level loop of `T` sequential steps, each launching its own
attention kernels, and `T` reaches `~83` on OOD — so the per-batch cost is a chain of `~83` dependent
launches that no amount of hardware parallelism can collapse, which is why it clocked in the 400–500 s
range. The Transformer replaces that with a *single* batched attention: all `T` query positions attend to
all `T` keys in one `[T, T]` score matrix, an `O(T² d)` operation but one fully parallel kernel per layer,
four layers, no sequential dependency across positions within a forward pass. `T² = 83² ≈ 6900` is a
trivially small matrix for the GPU, so the four-layer forward is a handful of large parallel matmuls
rather than `83` serial ones. I therefore expect the elapsed to fall by roughly the factor the
serial-to-parallel change buys — from the LSTM's ~450 s down to a small fraction of that, likely a couple
of minutes per variant — and the sinusoidal add itself is a single `[B, T, d]` table lookup once at the bottom, costing
essentially nothing on top of the bare attention.

So the delta from rung one is precise: where the LSTM carried order through a sequential recurrence with
no explicit code, I now run the parallel decoder-only Transformer and stamp every token with a frozen
sinusoidal absolute position before layer one. My falsifiable expectations against the rung-one numbers:
in-distribution should stay perfect (`exact_match_id ≈ 1.0` on all three variants — the absolute code is
more than enough at training length), and the elapsed time should drop sharply from the LSTM's 400–500 s
to a couple of minutes per variant, because attention parallelises. The open question is OOD,
and I have a specific prediction: if absoluteness is the disease, sinusoidal should *not* rescue OOD exact
match — I expect `exact_match_ood` to stay at or near `0.0` on all three variants, and possibly with OOD
token accuracy even *worse* than the LSTM's `0.37/0.21/0.18`, because the out-of-range phase patterns are
actively misleading the attention rather than merely under-anchoring it. If that comes true — perfect ID,
zero OOD, and OOD token accuracy at or below rung one — then the lesson is unambiguous: the failure is not
"which absolute code," it is *absolute position at all* — an absolute index keyed on over `[0, 20)` simply
has no counterpart over `[21, 40)`, and no sharpening of the absolute code can conjure one. That verdict is
exactly what this diagnostic rung exists to deliver, and it is the constraint every rung after it has to
answer to. (The full scaffold module is in the answer.)
