The sinusoidal run confirmed my worst-case prediction, and it confirmed it cleanly. In-distribution is
still perfect — `exact_match_id` is `1.0` on `delim` and `repeat`, `0.999` on `reverse`, with token
accuracy essentially `1.0` everywhere — so the absolute code is more than enough at the training length,
and the move to the parallel Transformer paid off on the clock: elapsed dropped from the LSTM's
400–500 s to ~125–130 s per variant. But OOD is *worse than rung one*, exactly as I feared. The
sequence-level OOD match is `0.0` on `delim` and `repeat` and `0.001` on `reverse` — effectively zero —
and the OOD token accuracy collapsed to `0.071 / 0.066 / 0.031`, well below the LSTM's `0.371 / 0.207 /
0.180`. So adding a *closed-form, extrapolation-friendly* absolute code did not merely fail to help OOD,
it made the per-token OOD accuracy actively worse than the recurrence that had no explicit code at all.
That is the decisive datum. The out-of-range phase patterns are not neutral; they are misleading the
attention. And it rules out the hypothesis that the fix is "a better absolute code," because sinusoidal
*is* the better absolute code — defined everywhere, bounded, shift-consistent — and it is the worst
extrapolator on the board. The disease is absolute position itself. Whatever the model learned to do with
positions over `[0, 20)`, it cannot transfer to `[21, 40)` because the absolute index it was keying on
simply does not recur out there.

Let me quantify how bad "worse than rung one" actually is, because one of these numbers is more damning
than a glance suggests. Ratioing sinusoidal's OOD token accuracy against the LSTM's: `delim`
`0.071 / 0.371 = 0.19`, `repeat` `0.066 / 0.207 = 0.32`, `reverse` `0.031 / 0.180 = 0.17` — sinusoidal
retains only a fifth to a third of the recurrence's per-token accuracy out of range. But the sharper test
is against blind chance. With 16 content symbols, a model that had learned nothing usable about position
and simply guessed the right *symbol identity* from content alone would still score around `1/16 = 0.0625`
per token. Sinusoidal's `delim` `0.071` and `repeat` `0.066` sit barely above that floor, and its
`reverse` `0.031` is *below* it. That last one is the decisive datum: an OOD token accuracy beneath the
uniform-guess baseline means the out-of-range positional code is not merely uninformative, it is
*anti*-informative — the phase pattern at positions 21–40 is confidently pointing the attention at the
wrong source token, worse than if the model ignored position entirely. This is exactly the 55-of-64 coarse
frequencies I counted at the previous rung coming due: their phase values at length 40 are configurations
the attention learned to read as *some* in-range position, so it dutifully attends there, and there is
wrong. A code that actively misleads is stronger evidence than a code that merely fails to help, and it is
what turns "absolute position does not extrapolate" from a suspicion into a verdict.

So I should stop *prescribing* a position scheme. Every option I would normally reach for is a
prescription with a fixed shape chosen before I have seen the data: sinusoidal is a periodic absolute
code (just measured to break), a learned table has no entry past training length (the scaffold default,
ruled out by inspection), a relative-bias table fixes a bucketing and a cutoff, a rotation scheme fixes a
frequency schedule, a linear-penalty scheme fixes a recency slope. Each fixed shape mismatches some task
or some length regime, and the in-distribution metric cannot even tell them apart — at training length
they all hit near-perfect accuracy, which is exactly what I just saw. I keep trying to pick the *best*
prescription, and rung two showed that the best absolute prescription is still a wall. That is the moment
to question the premise instead of the choice.

It is worth being concrete about *how* each fixed shape mismatches out of range, because "each mismatches
something" is only convincing if I can name the something. A relative-bias table with, say, `B` buckets
and a distance cutoff assigns every offset beyond the cutoff to one shared far-bucket, and that far-bucket
is trained only on whatever far distances actually occurred in `[0, 20)` — at OOD the offsets `21…80`
either collapse into that one saturated bucket (losing all resolution among far positions) or fall past a
cutoff the training never populated. A rotation scheme fixes a per-dimension frequency schedule exactly
like the sinusoid did, so its slow-frequency dimensions face the same "phase never seen past length 20"
problem I counted at the previous rung, just moved from the embedding into the dot product. A
linear-penalty scheme commits to one monotone recency slope per head, decided before training, which is a
strong shape prior that could be right or could fight a task needing a far peak. Each of these is a
specific out-of-range failure I can point at, and — the damning part — none of them is visible at training
length, where every scheme hits near-perfect accuracy and the in-distribution metric cannot separate them.
So I cannot select among prescriptions by the one signal I can measure cheaply. That is the moment to
question the premise instead of the choice.

Here is the premise I have been carrying since rung two: "self-attention with no position signal cannot
tell `a b` from `b a`, so I must add one." Let me check whether that is actually true *in this harness*,
because if it is not, I have been solving a problem I do not have. My model is a **decoder** — the
scaffold's `CausalSelfAttention` builds a lower-triangular mask and fills the upper triangle with `-inf`
before the softmax. So the query at position `t` attends only to positions `≤ t`. That is not a symmetric
set operation. Stare at it: the token at position 1 sees a window of size 1 (itself); position 2 sees a
window of size 2; position `t` sees a window of size `t`. *The size of the visible set is the position.*
The causal mask is not a neutral implementation detail — it injects order. A bidirectional encoder with
no positional code really would collapse to a bag of words, but a masked decoder does not. So the
premise is false for the architecture I actually have, and the implication is immediate: maybe the right
move for length generalization is not a better prescription but *no prescription at all* — delete the
positional scheme, let the causal mask supply order, and let gradient descent decide for itself how to
use it. The appeal is threefold. It removes the fixed inductive bias that was the source of every failure
mode, including the one I just measured. It is free — the attention score is exactly `q_t^⊤ k_i`, no
table, no bias term, no rotation, so none of the overhead. And it is the minimal change to the editable
block: return a scheme that supplies *nothing*.

Let me make "free" literal, because it is a real point in NoPE's favour and I should count it rather than
gesture at it. Sinusoidal cost one `token_embedding_extra` lookup-and-add at the bottom of the stack: a
gather of `T` rows from the `[256, 128]` frozen table and a `[B, T, 128]` addition, once per forward.
NoPE deletes that hook entirely — `SeqModel.forward` sees `token_embedding_extra is None` and skips the
add — so there is one fewer `[B, T, 128]` tensor materialised and one fewer elementwise add per forward
pass. Inside attention, sinusoidal changed nothing (it acted only at the embedding), but the more general
point is that the two *other* hooks a positional scheme could fill — `attn_bias` (a `[H, T, T]` add before
the mask) and `rotary` (a per-layer rotation of `q, k` inside every attention sublayer) — are also `None`,
so across four layers NoPE adds exactly zero positional arithmetic. The attention score is the bare
`q_t^⊤ k_i / √{head_dim}` with `head_dim = 128 / 4 = 32`, so `√{head_dim} ≈ 5.66`, and the only tensors in
the score are the content projections. NoPE is therefore not just the smallest edit but the *cheapest*
model on the ladder: strictly less work than any explicit scheme, while being the only one whose position
signal has no fixed shape to be wrong out of range. When a candidate is simultaneously the cheapest and
the least prescriptive, and the most prescriptive absolute code just posted a below-chance OOD number, the
burden of proof shifts onto keeping any scheme at all.

But "the mask carries some order" is suggestive, not a guarantee that a no-PE causal model can represent
position *well enough to extrapolate*. I want to convince myself the architecture is expressive enough —
exhibit, by hand, weights that compute position from nothing but the mask, so that I know SGD at least has
a target to find. Start with absolute position in a single layer. At the input the hidden state is just
the word embeddings, no position added. The only position-bearing structure is the causal mask, the
softmax, and the `BOS` token anchored at position 1 (the scaffold's layout always begins with `BOS`).
Reserve three coordinates of the residual stream. Design the embedding so that for every token coordinate
1 is `1`, coordinate 2 is `1` iff the token is `BOS` else `0`, and coordinate 3 is `0`. Make one head's
key projection read coordinate 1 — which is `1` for every token, so every key is identical. Then for a
query at position `t`, every logit over the `t` visible keys is the same value, the softmax over equal
logits is *uniform*, `α̂_i = 1/t` for each `i ≤ t`. The mask did the counting; the softmax turned the
count into a number. Now make the value projection read coordinate 2 — `1` only at `BOS` — so the
attention output is `Σ_{i≤t} α̂_i v_i = (1/t)·v_{BOS}`, and the output projection copies that into
coordinate 3. After layer one, coordinate 3 holds `1/t`, a faithful injective code for absolute position
(strictly decreasing in `t`); the GELU MLP that follows is a universal approximator, so it can re-map
`1/t` to `t` or any monotone re-coding. The `BOS` anchor set the numerator, the causal window size set
the denominator. So the architecture *can* recover absolute position with no positional code at all.

Let me check the counting numerically, because the whole construction hinges on "softmax over `t` equal
logits is uniform," and I want to see it hold at small `t`. At `t = 2` the query sees two keys with equal
logits `ℓ`, so the softmax weights are `e^ℓ / (e^ℓ + e^ℓ) = 1/2` each — and `1/t = 1/2`, matching. At
`t = 4`, four equal logits give `1/4` each, `1/t = 0.25`. The identity is exact for any `t` because equal
logits cancel in the softmax ratio regardless of their common value, so the head does not even need
calibrated logit magnitudes — it just needs identical keys, which reading the all-ones coordinate 1
guarantees. Good: the counting is robust. But the same arithmetic exposes a limitation that matters for
extrapolation, and I should name it now. The recovered code is `1/t`, and its *resolution* — the gap
between adjacent positions — is `1/t − 1/(t+1) = 1/(t(t+1))`. At the short end that is coarse and easy:
positions 1 and 2 differ by `1 − 1/2 = 0.5`. At the long end it is crushed: positions 39 and 40 differ by
`1/(39·40) ≈ 0.00064`. So while `1/t` is injective across `1…40` (strictly decreasing, every position a
distinct value), the far positions are packed into a vanishingly thin band, and any downstream MLP trying
to separate position 39 from 40 must resolve a `6·10^{-4}` difference against activation noise. This is a
concrete, architecture-level reason that even the *recoverable* absolute code degrades with distance — and
it reinforces why the move that matters is not "recover a better absolute index" but "make the score
depend on the relative offset `t − i`," whose resolution does not collapse at large `t`.

Absolute position is exactly what failed to extrapolate, though, so the question that matters is whether
the *later* layers can convert that recovered absolute index into a *relative* signal — make the
attention dot product depend on `t − i`. Construct that too. Assume coordinate 3 of every hidden vector
now holds the position, preserved across the residual stream. In a layer `l ≥ 2`, engineer the query at
position `t` to read out `[1, −t, …]` (row 1 of `W_Q` reads coordinate 1, giving `1`; row 2 reads
coordinate 3 with coefficient `−1`, giving `−t`) and the key at position `i` to read out `[i, 1, …]` (row
1 of `W_K` reads coordinate 3, giving `i`; row 2 reads coordinate 1, giving `1`). Then
`⟨q_t, k_i⟩ = 1·i + (−t)·1 + (content) = (content) − (t − i)`. The logit splits into a content term plus a
pure function of the relative offset `t − i`. Let me put numbers through it to be sure the offset is what
falls out: a query at position `t = 7` reading `[1, −7]` against a key at position `i = 4` reading
`[4, 1]` gives `1·4 + (−7)·1 = 4 − 7 = −3 = −(7 − 4) = −(t − i)`; the same query against a key at `i = 6`
gives `6 − 7 = −1`, less penalised because it is nearer. So the nearer key gets the higher score purely
from the position coordinates, and the absolute indices 7, 4, 6 have vanished from the result — only their
differences survive. Crucially, the resolution problem I just flagged for `1/t` does not recur here: the
gap between offset `3` and offset `4` is a full unit in logit space at *every* absolute position, whether
`t` is 7 or 37, because the construction subtracts the raw indices rather than their reciprocals. That is
the concrete reason a relative score is the right target and an absolute one is not. That is exactly the
form of a relative positional encoding —
the score depends on the distance, not on the absolute indices — and `−(t − i)` is just the simplest
reachable case; because the first layer's MLP can write any function of absolute position into coordinate
3, later layers can realise a far richer dependence on `t − i`. So a no-PE causal Transformer can recover
absolute position in layer one and synthesise a relative encoding in any later layer. The architecture
contains absolute-like and relative-like behaviour as special cases, and — unlike every explicit scheme —
it does not force a prescription; SGD picks whichever the task rewards.

For copy / repeat / reverse, what it should reward is relative-and-bimodal. While emitting `y_k` the model
needs to attend to the corresponding `x` token, which may be far back into the input, *and* to the local
decoding context — a short-and-long attention pattern, not the monotone recency that a linear-penalty
scheme would force. Let me be concrete about that bimodality, because it is the property that separates
NoPE from a prior that only knows "nearer is better." To emit the next symbol correctly the head must
place mass on the *paired source token* — roughly `L` positions back for `delim`, up to `~2L` back for
`reverse` — and it also benefits from watching the immediately preceding output token to keep the emission
counter aligned. That is a distribution with two separated peaks: one far, one near. A score that is
purely a decreasing function of distance can only ever produce a single near-peaked mass; it cannot put a
second peak far away without also lifting everything between. A learned-from-the-mask relative score has
no such constraint — the query/key geometry can carve out a bump at a specific offset while leaving the
in-between distances low — so NoPE can in principle express the exact two-peaked read the task rewards.
This is the freedom that a fixed-shape prescription trades away, and it is precisely why I want to test
the prescription-free model before committing to any particular distance shape. Relative distance is the rule that is the *same* at every length: "the source token
`reverse`-paired with output position `k` is at a fixed offset from the seam" does not change when the
sequence grows. So if the no-PE model learns a relative encoding, it has a real shot at OOD where the
absolute schemes had none — the offset that worked at length 15 is the same offset at length 30, and
nothing in `q_t^⊤ k_i` goes out of distribution because there is no absolute code to fall off the end of.
That is the precise mechanism by which deleting the scheme could beat rung two: it does not add a relative
prior, it *removes* the absolute prior that was the thing breaking, and lets the mask-derived position be
relative if the data wants it.

In the edit surface this is the smallest possible change, and the methods-level derivation already
matches this exact harness, so there is little to re-express. `build_positional_scheme` returns a scheme
named `"nope"` with all three hooks `None` and an empty `extra_modules` — no additive token embedding, no
attention bias, no rotation, no learnable positional parameter. `SeqModel.forward` then skips the
`token_embedding_extra` add (it is `None`), `CausalSelfAttention` adds no bias before the causal mask,
and applies no rotary, so the attention score is exactly `q_t^⊤ k_i / √{head_dim}` under the lower-
triangular mask. `build_model` returns the plain `SeqModel(use_lstm=False)`. The one thing worth naming
that the harness *does* provide and that the construction leans on: the layout always starts with `BOS`
at position 0, which is the anchor the layer-one absolute-position construction needs, and the causal mask
is applied unconditionally — so both ingredients of the existence proof are present in the fixed
substrate. There is no machinery to omit here because there is no machinery at all; the method *is* the
absence of a positional scheme.

My falsifiable expectations against the rung-two numbers. In-distribution should stay perfect on all
three variants (`exact_match_id ≈ 1.0`) — removing the absolute code costs nothing at training length,
since the mask already supplies order there. Elapsed should be about the same as sinusoidal, ~125–135 s,
or marginally faster (one fewer add per layer). The whole bet rides on OOD, and here is the specific
prediction that would falsify the absolute schemes' framing: `exact_match_ood` should move *off zero* on
at least some variants, and OOD token accuracy should jump well above sinusoidal's `0.071 / 0.066 /
0.031` — back above the LSTM and beyond — because the model can now key on relative offsets that recur at
unseen lengths. I expect the gains to be uneven across variants, and it is worth reasoning out the offset each variant
demands to see why. For `delim`, emitting `y_k` requires attending to source position `k`; measured from
the query's own position in the stream (which sits at `sep_pos + k`), the target is a *fixed* offset back
past the seam — the same relative jump at every length, the friendliest case for a relative code. For
`repeat`, the second copy asks for source position `((k−1) mod L)+1`, so the offset resets once at the
copy boundary but is otherwise locally periodic — also relative, with one length-dependent wrap. For
`reverse`, emitting `y_k` requires source position `L − k + 1`: when `k` is near the *end* of the output
the model must reach all the way back to the *start* of the input, an offset of order `2L` that grows with
length and is the single longest reach on the board. So `reverse` is the hardest test of whether the
learned relative code truly extrapolates — it needs a genuinely long-range relative jump that at OOD
length 40 is roughly `~80` positions, far past anything the training range exercised. `delim` and `repeat`
involve offsets that track the seam more locally, so a mask-derived relative rule should transfer to them
more readily. If NoPE lifts OOD exact match on `delim`/`repeat`/`reverse`
above zero while holding ID perfect, that is the confirmation that absoluteness was the disease and
relative-from-the-mask is the cure. What it would leave unsettled is a question of *shape*: a relative code
SGD induces from nothing but the causal mask is only as sharp as the offsets the training data actually
pinned down, and the longest reaches — `reverse`'s roughly `2L ≈ 80`-position jump at OOD length 40 — are
exactly where a learned-from-nothing code is thinnest and most liable to fray. So even a NoPE win would
likely carry a soft edge on the longest-reach variant, and that soft edge is the seam the rest of the
ladder has to work on. (The full scaffold module is in the answer.)
