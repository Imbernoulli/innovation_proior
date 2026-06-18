NoPE broke the wall, and the numbers tell me both that the diagnosis was right and where it is still
soft. Deleting the positional code lifted OOD off zero on every variant: `exact_match_ood` went from
sinusoidal's `0.0 / 0.0 / 0.001` to `0.297 / 0.550 / 0.403`, and OOD token accuracy from
`0.071 / 0.066 / 0.031` to `0.433 / 0.757 / 0.434` — past the LSTM, past everything. In-distribution
held perfect (`0.998 / 0.996 / 1.0`), elapsed stayed ~130 s, and the geometric-mean score jumped from
`0.50` to `0.706`. So absoluteness *was* the disease: with no absolute code to fall off the end of, the
mask-derived position let SGD find a relative-ish rule that recurs at unseen lengths. That confirms the
whole arc. But read the per-variant OOD numbers closely, because they expose what NoPE leaves on the
table. `repeat` extrapolates best (`0.550` exact, `0.757` token) — its offsets track the seam in a fairly
local, periodic way. `delim` and `reverse` are the weaker pair (`0.297`/`0.403` exact, `0.433`/`0.434`
token), and the symmetry of their token accuracies around `0.43` is telling: NoPE recovers position by
*construction* — the layer-one `1/t` count anchored at `BOS`, re-coded by the MLP into a relative score —
but that recovered code is something SGD has to discover and then keep stable across 30–40 unseen
positions, with no inductive bias pushing it toward the right shape. It works, but it is *learned from
nothing*, and "learned from nothing" is exactly the kind of thing that holds at length 25 and frays at
length 40. The next move is the obvious one: stop *hoping* the model induces a good relative code and
*give* it one — a relative bias whose value at an unseen distance is the obvious continuation of its value
at seen distances, so there is no out-of-distribution regime to fray in.

So I want a relative scheme, but I have to choose it carefully, because NoPE already taught me the failure
modes of the explicit family. A learned relative-bias table (T5-style) fixes a bucketing and a cutoff,
and the far buckets are trained only on whatever far distances appeared in `[0, 20)`, so even bucketing
only partly defines behaviour out of range — and it adds parameters and a gather, which is real overhead.
A rotation scheme is relative-by-construction but injects per-layer rotations that cost compute, and its
frequency schedule is a fixed prescription too. What I want is the property the relative family has —
position lives in the *score*, depends only on `t − i`, never enters the values, reasserts at every layer
— but with a bias-versus-distance function so simple it needs no parameters, no gather, and is defined,
smoothly, for *every* distance including ones I never trained on. The disease with sinusoidal was that the
signal past the training length was a novel high-dimensional pattern; the cure is a position signal whose
value at distance `d` is the obvious continuation of its values at small `d`, with no learned dial that
could be miscalibrated out of range. The simplest such function of distance is a straight line. So: add to
the query–key score, before the softmax, a bias that is *linear* in the relative distance `i − j`.

Which direction does the line go? I want a recency bias — distant keys matter less than nearby ones, all
else equal — and crucially I want that preference to have the *same shape* at every distance, so distance
40 is just "distance 4 but more so," never a regime the model has not conceptually seen. So make the bias
*more negative* the farther the key is from the query. For a causal query at position `i` over keys
`j ≤ i`, the relative distance is `i − j ≥ 0`, and I subtract a penalty proportional to it: the key right
next to the query (distance 0) gets penalty 0, one step back gets `−m`, two steps back `−2m`, linearly. A
per-head slope `m > 0`. No embedding added anywhere, no learned bias, no bucket gather — and it is defined
at any distance, because at distance 40 the penalty is just `40·m`, the same line rather than a new
absolute vector. This directly attacks the failure I diagnosed at rung two and the softness I diagnosed at
rung three: the thing the model interprets is a scalar that grows linearly and predictably, not a phase
pattern that goes novel (sinusoidal) and not a code SGD had to induce from scratch (NoPE).

Does this keep the structural properties NoPE showed me matter? Position lives in the scores, not the
values — the bias is added to the attention scores, the values are untouched, so the attention output
carries no absolute-position component, exactly like the relative code NoPE synthesised in its later
layers. It injects at every layer — every attention sublayer adds the same bias, so position reasserts
itself throughout, where the sinusoidal add happened only once at the bottom. And it depends only on
`i − j`, the relative offset, which is the quantity that recurs at unseen lengths. So this is the
explicit, well-shaped version of the thing NoPE was groping toward — but instead of a single learned
recency shape I get a *spread* of them, because the slope is per-head.

The slope needs care. One slope for all heads would force every head to the same recency preference,
which is clearly wrong — some heads should look locally, some should keep a long view, and for `reverse`
in particular I need at least one head that can reach far back into the input without being crushed by the
penalty. So `m` is per-head, and the `H` slopes should span a range of recency scales: a head with large
`m` is penalised hard for distance and becomes local; a head with tiny `m` barely notices distance and
keeps a long-range view. The long-range heads are the precious ones, so I want more resolution near zero —
the slopes bunched toward 0, spread toward 1 — which is what a geometric sequence gives (equal ratios =
equal log-spacing, dense near the small end). The clean choice is start = ratio = `2^{−8/n}`, giving
`2^{−8/n}, 2^{−16/n}, …, 2^{−8}`; for the common power-of-two case it reduces to `1/2, 1/4, …, 1/256` at
8 heads. This task has 4 heads, also a power of two, so the formula applies directly:
`2^{−8/4} = 2^{−2} = 1/4`, giving slopes `1/4, 1/16, 1/64, 1/256` — four distinct recency scales from a
fairly local head down to one that barely penalises distance, which is the long-range head `reverse`
needs. For a non-power-of-two head count the scaffold falls back to the closest-power-of-two set plus
interleaved extras, but that path is unused here.

Should the slopes be learned? My instinct is to let gradient descent find them, but for *extrapolation*
specifically that is the wrong instinct, and NoPE is the cautionary tale: learned slopes would be tuned to
the distances seen in `[0, 20)`, with no training signal about how the slope should behave at the far
distances that only appear at evaluation, so a learned slope could overfit the in-range recency statistics
and generalise *worse* out of range — the same "learned from nothing reliable past the training length"
weakness I am trying to fix. A fixed geometric set is the same line everywhere and cannot be miscalibrated
to the training length. So I fix the slopes before training. They are cached as a parameter-free buffer
wrapped in a tiny `nn.Module` so they follow the model to the GPU, registered via `scheme.extra_modules`,
and carry no gradient.

One scaling subtlety, because it is easy to get wrong. Standard attention divides the scores by
`√{head_dim}` to tame the variance of a random dot product. My bias is not a random dot product — it is a
deterministic geometric penalty I am choosing directly in score-space. If I also divided it by
`√{head_dim}` I would just be rescaling slopes I already choose freely, coupling my slope choice to the
head dimension for no reason. So the linear bias is added *after* the `√{head_dim}` scaling, not inside it.
In this harness that is automatic: `CausalSelfAttention` computes `attn_scores = q·k / √head_dim` first,
then adds `scheme.attn_bias(T, …)`, so my bias is correctly outside the scaling.

Now the concrete fill, and here is where the harness version differs from the canonical implementation, so
I derive *this* one. The scaffold's `attn_bias` hook must return a `[H, T, T]` (or `[1, T, T]`) tensor
that is added to the scores *before* the causal mask is applied (the loop applies the lower-triangular
`-inf` mask itself, right after adding the bias). The canonical ALiBi implementation exploits softmax's
per-row shift-invariance to replace the honest staircase `m·[−(i−1), …, −1, 0]` with a single broadcast
row `m·[0, 1, …, T−1]`, since the two differ only by a per-row constant that softmax kills — a cheap
trick that works *because* the bias is linear. This harness does **not** take that shortcut: it builds the
full signed relative-distance matrix `rel = idx[None, :] − idx[:, None]` (so `rel_{ij} = j − i`, negative
below the diagonal where `j < i`) and sets `bias = slope · rel`, i.e. `bias_{ij} = m·(j − i) = −m·(i − j)`
— the honest staircase, more negative for keys farther *back*, exactly as the math says, with no
shift-invariance optimisation. Above the diagonal `rel` is positive (a *positive* bias), but those entries
are immediately overwritten by the loop's `-inf` causal mask before the softmax, so they never matter — I
do not need to mask them myself. The result is shaped `[H, T, T]` (one slope per head), the loop
broadcasts it across the batch and adds the causal mask on top, and the values and token embeddings stay
completely position-free. `build_positional_scheme` returns the scheme with only the `attn_bias` hook set
(`token_embedding_extra` and `rotary` are `None`); `build_model` returns the plain
`SeqModel(use_lstm=False)`. What the harness does *not* expose that the canonical recipe might use: the
softmax-shift cheap-construction (it builds the full matrix instead), and any folding of the bias into the
causal mask in one allocation (here they are two separate adds). Neither changes the algorithm — the
attention weights are identical to the canonical ALiBi — only the arithmetic path differs.

My falsifiable expectations against the NoPE numbers, which are the bar this rung must clear. In-
distribution should stay perfect on all three variants (the bias is negligible at short distances, and
the model had no trouble at training length without it). The whole bet is OOD, and the prediction is
specific: ALiBi should beat NoPE on OOD exact match where NoPE's *learned* relative code was softest — I
expect the biggest lift on `delim` (NoPE only `0.297`) and a solid lift on `repeat`, because the linear
recency bias is precisely the well-shaped relative prior NoPE had to induce from scratch, and it is
defined identically at length 40 as at length 15. The risk I have to name is `reverse`: ALiBi's recency
bias is a *monotone* preference for nearby tokens, but `reverse` requires attending from the end of the
output all the way back to the *start* of the input — the longest possible offset — which is exactly the
direction the recency penalty fights. So `reverse` is where a built-in recency prior could backfire, and
I expect it to be ALiBi's weakest variant, possibly even at or below NoPE there, with the long-range head
(`slope = 1/256`) doing the heavy lifting if it works at all. The honest claim is therefore aggregate, not
uniform: ALiBi should lift the geometric-mean score above NoPE's `0.706` by winning clearly on
`delim`/`repeat` even if it pays a little on `reverse` — a well-shaped, parameter-free relative prior
beating a relative code learned from nothing. If instead `reverse` collapses far enough to drag the
geometric mean below NoPE, that would falsify the "monotone recency is good enough here" assumption and
point the next step toward a non-monotone relative scheme. (The full scaffold module is in the answer.)
