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

So I should stop *prescribing* a position scheme. Every option I would normally reach for is a
prescription with a fixed shape chosen before I have seen the data: sinusoidal is a periodic absolute
code (just measured to break), a learned table has no entry past training length (the scaffold default,
ruled out by inspection), a relative-bias table fixes a bucketing and a cutoff, a rotation scheme fixes a
frequency schedule, a linear-penalty scheme fixes a recency slope. Each fixed shape mismatches some task
or some length regime, and the in-distribution metric cannot even tell them apart — at training length
they all hit near-perfect accuracy, which is exactly what I just saw. I keep trying to pick the *best*
prescription, and rung two showed that the best absolute prescription is still a wall. That is the moment
to question the premise instead of the choice.

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

Absolute position is exactly what failed to extrapolate, though, so the question that matters is whether
the *later* layers can convert that recovered absolute index into a *relative* signal — make the
attention dot product depend on `t − i`. Construct that too. Assume coordinate 3 of every hidden vector
now holds the position, preserved across the residual stream. In a layer `l ≥ 2`, engineer the query at
position `t` to read out `[1, −t, …]` (row 1 of `W_Q` reads coordinate 1, giving `1`; row 2 reads
coordinate 3 with coefficient `−1`, giving `−t`) and the key at position `i` to read out `[i, 1, …]` (row
1 of `W_K` reads coordinate 3, giving `i`; row 2 reads coordinate 1, giving `1`). Then
`⟨q_t, k_i⟩ = 1·i + (−t)·1 + (content) = (content) − (t − i)`. The logit splits into a content term plus a
pure function of the relative offset `t − i`. That is exactly the form of a relative positional encoding —
the score depends on the distance, not on the absolute indices — and `−(t − i)` is just the simplest
reachable case; because the first layer's MLP can write any function of absolute position into coordinate
3, later layers can realise a far richer dependence on `t − i`. So a no-PE causal Transformer can recover
absolute position in layer one and synthesise a relative encoding in any later layer. The architecture
contains absolute-like and relative-like behaviour as special cases, and — unlike every explicit scheme —
it does not force a prescription; SGD picks whichever the task rewards.

For copy / repeat / reverse, what it should reward is relative-and-bimodal. While emitting `y_k` the model
needs to attend to the corresponding `x` token, which may be far back into the input, *and* to the local
decoding context — a short-and-long attention pattern, not the monotone recency that a linear-penalty
scheme would force. Relative distance is the rule that is the *same* at every length: "the source token
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
unseen lengths. I expect the gains to be uneven across variants: `reverse` needs the model to attend from
the end of the output back to the start of the input, a long, length-dependent offset, so it is the
hardest test of whether the learned relative code truly extrapolates; `delim` and `repeat` involve
offsets that track the seam more locally. If NoPE lifts OOD exact match on `delim`/`repeat`/`reverse`
above zero while holding ID perfect, that is the confirmation that absoluteness was the disease and
relative-from-the-mask is the cure — and it sets up the final rung, where instead of *hoping* SGD finds a
relative code I give it a relative recency bias explicitly and see whether a well-shaped prior beats the
learned-from-nothing one. (The full scaffold module is in the answer.)
