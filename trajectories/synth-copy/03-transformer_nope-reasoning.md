The sinusoidal run confirmed my worst-case prediction, cleanly. In-distribution is still perfect —
`exact_match_id` is `1.0` on `delim` and `repeat`, `0.999` on `reverse`, token accuracy essentially `1.0`
everywhere — and the move to the parallel Transformer paid off on the clock: elapsed dropped from the
LSTM's 400–500 s to ~125–130 s per variant. But OOD is *worse than rung one*, exactly as I feared:
sequence-level OOD match is `0.0` on `delim`/`repeat` and `0.001` on `reverse`, and OOD token accuracy
collapsed to `0.071 / 0.066 / 0.031`, well below the LSTM's `0.371 / 0.207 / 0.180`. So a closed-form,
extrapolation-friendly absolute code did not merely fail to help OOD — it made per-token OOD accuracy
actively worse than the recurrence that had no explicit code at all.

That damage is sharper than the ratios first suggest. Against blind chance: a model that learned nothing
usable about position and guessed the right symbol identity from content alone would score around
`1/16 = 0.0625` per token. Sinusoidal's `delim` `0.071` and `repeat` `0.066` sit barely above that floor,
and its `reverse` `0.031` is *below* it. An OOD token accuracy beneath the uniform-guess baseline means the
out-of-range positional code is not merely uninformative, it is *anti*-informative — the phase pattern at
positions 21–40 is confidently pointing the attention at the wrong source token, worse than ignoring
position entirely. This is the 55-of-64 coarse frequencies I counted last rung coming due: their phase
values at length 40 are configurations the attention learned to read as *some* in-range position, so it
dutifully attends there, and there is wrong. A code that actively misleads is stronger evidence than one
that merely fails to help. The disease is absolute position itself: whatever the model learned to do over
`[0, 20)` cannot transfer to `[21, 40)` because the absolute index it keys on does not recur out there.

So I should stop *prescribing* a position scheme. Every option I would normally reach for is a fixed shape
chosen before I see the data, and each mismatches some length regime in a way the in-distribution metric
cannot even reveal — at training length they all hit near-perfect accuracy. Sinusoidal is a periodic
absolute code (just measured to break); the learned table has no entry past training length (ruled out by
inspection). A relative-bias table with `B` buckets and a distance cutoff trains its far-bucket only on
the sparse far distances that appear in `[0, 20)`, then at OOD either collapses offsets `21…80` into that
one saturated bucket (losing all far resolution) or drops them past a cutoff training never populated. A
rotation scheme fixes a per-dimension frequency schedule exactly like the sinusoid, so its slow
frequencies face the same "phase never seen past 20" problem, moved from the embedding into the dot
product. A linear-penalty scheme commits to one monotone recency slope, a strong shape prior that could
fight a task needing a far peak. Each is a nameable out-of-range failure, none visible at training length.
I keep trying to pick the *best* prescription, and the best absolute one is still a wall. That is the
moment to question the premise, not the choice.

Here is the premise I have carried since rung two: "self-attention with no position signal cannot tell
`a b` from `b a`, so I must add one." Is that true *in this harness*? My model is a **decoder** — the
scaffold's `CausalSelfAttention` builds a lower-triangular mask and fills the upper triangle with `-inf`
before the softmax, so the query at position `t` attends only to positions `≤ t`. That is not a symmetric
set operation. Position 1 sees a window of size 1 (itself); position 2 sees size 2; position `t` sees size
`t`. *The size of the visible set is the position.* The causal mask is not a neutral implementation detail
— it injects order. A bidirectional encoder with no positional code really would collapse to a bag of
words, but a masked decoder does not. So the premise is false for the architecture I actually have, and the
move for length generalization is not a better prescription but *no prescription at all*: delete the
positional scheme, let the causal mask supply order, and let gradient descent decide how to use it. It
removes the fixed inductive bias that was the source of every failure mode; it is free (the attention score
is exactly `q_t^⊤ k_i`, no table, no bias, no rotation — strictly less work than any explicit scheme); and
it is the minimal change to the editable block: return a scheme that supplies nothing.

But "the mask carries some order" is suggestive, not a guarantee that a no-PE causal model can *represent*
position well enough to extrapolate. I want to exhibit, by hand, weights that compute position from
nothing but the mask, so I know SGD has a target to find. Start with absolute position in a single layer.
At the input the hidden state is just word embeddings; the only position-bearing structure is the mask,
the softmax, and the `BOS` token anchored at the first position (the layout always begins with `BOS`). Reserve
three coordinates: design the embedding so coordinate 1 is `1` for every token, coordinate 2 is `1` iff the
token is `BOS`, coordinate 3 is `0`. Make one head's key projection read coordinate 1 — `1` for every
token, so every key is identical; then for a query at position `t`, every logit over the `t` visible keys
is equal, and the softmax over equal logits is *uniform*, `α̂_i = 1/t`. The mask did the counting; the
softmax turned the count into a number. Make the value projection read coordinate 2 (`1` only at `BOS`), so
the attention output is `Σ_{i≤t} α̂_i v_i = (1/t)·v_{BOS}`, and the output projection copies that into
coordinate 3. After layer one, coordinate 3 holds `1/t`, a faithful injective code for absolute position,
and the GELU MLP that follows can re-map `1/t` to `t` or any monotone recoding. So the architecture *can*
recover absolute position with no positional code.

The same arithmetic exposes a limitation that matters for extrapolation. The recovered code is `1/t`, whose
resolution — the gap between adjacent positions — is `1/t − 1/(t+1) = 1/(t(t+1))`. At the short end that is
coarse and easy: positions 1 and 2 differ by `0.5`. At the long end it is crushed: positions 39 and 40
differ by `1/(39·40) ≈ 6·10^{-4}`, which a downstream MLP must resolve against activation noise. So even
the *recoverable* absolute code degrades with distance — an architecture-level reason the move that matters
is not "recover a better absolute index" but "make the score depend on the relative offset `t − i`," whose
resolution does not collapse at large `t`.

Absolute position is exactly what failed to extrapolate, so the real question is whether *later* layers can
convert the recovered absolute index into a *relative* signal. Construct that too. Assume coordinate 3 now
holds the position across the residual stream. In a layer `l ≥ 2`, engineer the query at position `t` to
read out `[1, −t, …]` and the key at position `i` to read out `[i, 1, …]`; then `⟨q_t, k_i⟩ = 1·i + (−t)·1
+ (content) = (content) − (t − i)`. The logit splits into a content term plus a pure function of the
relative offset `t − i`, with the absolute indices vanished — only their difference survives, and the
nearer key gets the higher score purely from the position coordinates. Crucially the resolution problem
does not recur: the gap between offset 3 and offset 4 is a full unit in logit space at *every* absolute
position, whether `t` is 7 or 37, because the construction subtracts raw indices rather than reciprocals.
And `−(t − i)` is just the simplest reachable case; because layer one's MLP can write any function of
absolute position into coordinate 3, later layers can realise a far richer dependence on `t − i`. So a
no-PE causal Transformer can recover absolute position in layer one and synthesise a relative encoding in
any later layer — it contains absolute-like and relative-like behaviour as special cases, and, unlike every
explicit scheme, forces no prescription; SGD picks whichever the task rewards.

For copy / repeat / reverse, what it should reward is relative-and-bimodal. To emit `y_k` the head must
place mass on the *paired source token* — roughly `L` positions back for `delim`, up to `~2L` for `reverse`
— *and* watch the immediately preceding output token to keep the emission counter aligned. That is a
distribution with two separated peaks, one far and one near. A score that is purely a decreasing function
of distance can only ever produce a single near-peaked mass; it cannot put a second peak far away without
lifting everything between. A learned-from-the-mask relative score has no such constraint — the query/key
geometry can carve out a bump at a specific offset while leaving in-between distances low. And relative
distance is the rule that is the *same* at every length: the offset from the seam to the source token
`reverse`-paired with output position `k` does not change when the sequence grows. So if the no-PE model
learns a relative encoding it has a real shot at OOD where the absolute schemes had none — the offset that
worked at length 15 is the same at length 30, and nothing in `q_t^⊤ k_i` falls off the end of a table. The
mechanism by which deleting the scheme could beat rung two is precise: it does not add a relative prior, it
*removes* the absolute prior that was breaking, and lets the mask-derived position be relative if the data
wants it.

In the edit surface this is the smallest possible change, and the harness already supplies both ingredients
the existence proof leans on. `build_positional_scheme` returns a scheme named `"nope"` with all three
hooks `None` and empty `extra_modules` — no additive embedding, no attention bias, no rotation. Then
`SeqModel.forward` skips the `token_embedding_extra` add, `CausalSelfAttention` adds no bias before the
causal mask and applies no rotary, so the score is exactly `q_t^⊤ k_i / √{head_dim}` under the
lower-triangular mask. `build_model` returns the plain `SeqModel(use_lstm=False)`. The layout always starts
with `BOS` (the anchor for the layer-one construction) and the causal mask is applied
unconditionally, so both ingredients are present; there is no machinery to add, because the method *is* the
absence of a positional scheme.

My falsifiable expectations against the rung-two numbers. In-distribution should stay perfect on all three
(removing the absolute code costs nothing at training length, where the mask already supplies order).
Elapsed about the same as sinusoidal, ~125–135 s. The whole bet rides on OOD: `exact_match_ood` should move
*off zero* on at least some variants and OOD token accuracy should jump well above sinusoidal's `0.071 /
0.066 / 0.031`, because the model can now key on relative offsets that recur at unseen lengths. I expect the
gains uneven, and the per-variant offset says why. For `delim`, emitting `y_k` attends to source position
`k`, a fixed jump back past the seam at every length — the friendliest case. For `repeat`, the offset
resets once at the copy boundary but is otherwise locally periodic. For `reverse`, `y_k` needs source
position `L − k + 1`, so near the end of the output the model reaches all the way back to the start of the
input, an offset of order `2L` that grows with length — `~80` positions at OOD length 40, far past anything
training exercised. So `reverse` is the hardest test of whether the learned relative code truly
extrapolates, and even a NoPE win would likely carry a soft edge on that longest-reach variant. If NoPE
lifts OOD exact match above zero while holding ID perfect, that confirms absoluteness was the disease and
relative-from-the-mask is the cure. (The full scaffold module is in the answer.)
