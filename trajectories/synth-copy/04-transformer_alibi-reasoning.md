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

Let me confirm I am reading the summary number correctly, because the whole ranking rides on it and I want
to see the geomean fall out of the per-variant scores by hand. Each variant's `score` is
`0.5·exact_id + 0.5·exact_ood`: `delim` is `0.5·0.998 + 0.5·0.297 = 0.499 + 0.149 = 0.648`, `repeat` is
`0.5·0.996 + 0.5·0.550 = 0.498 + 0.275 = 0.773`, `reverse` is `0.5·1.0 + 0.5·0.403 = 0.500 + 0.202 =
0.702`. The task summary is the geometric mean, `(0.648·0.773·0.702)^{1/3}`: the product is
`0.648·0.773 = 0.501`, times `0.702` is `0.352`, and the cube root of `0.352` is `0.706` — matching the
reported `0.7055`. Doing this by hand is not busywork; it tells me *where the leverage is*. Because the
summary is a geometric mean, it is dominated by the smallest factor — the variant with the weakest score
drags the whole number down more than a strong variant lifts it. NoPE's smallest factor is `delim` at
`0.648`, pulled down by its OOD exact of only `0.297`. So the single most valuable thing I can do at this
rung is raise the *weakest* variant's OOD exact match; a gain there moves the geomean more than an equal
gain on `repeat`, which is already the strong factor at `0.773`. That reframes the target: not "beat NoPE
on average" but "lift NoPE's soft variants, `delim` and `reverse`, without giving back too much elsewhere."

So I want a relative scheme, but I have to choose it carefully, because NoPE already taught me the failure
modes of the explicit family. A learned relative-bias table (T5-style) fixes a bucketing and a cutoff,
and the far buckets are trained only on whatever far distances appeared in `[0, 20)`, so even bucketing
only partly defines behaviour out of range — and it adds parameters and a gather, which is real overhead:
a table of, say, 32 buckets × 4 heads is 128 learned scalars, each a dial that can be miscalibrated, plus
a per-layer gather that maps every `(i, j)` pair to its bucket index. Worse, the miscalibration is exactly
where it hurts — the single far-bucket absorbing all distances past the cutoff is trained on the sparse
tail of in-range distances and then asked to govern the *dense* far distances that only appear at OOD. A
rotation scheme is relative-by-construction but injects per-layer rotations of `q` and `k` inside every one
of the four attention sublayers — an extra elementwise multiply-and-combine on `[B, H, T, head_dim]`
tensors at each layer rather than a single additive bias — and its frequency schedule is a fixed
prescription with the very same slow-frequency-goes-novel problem the sinusoid had, now relocated into the
dot product. So the relative family splits into "cheap and shapeless" (which I already tested as NoPE) and
"parametric or compute-heavy prescriptions" (T5, rotation) — and what I want is the missing corner: a
prescription with the *right* shape but *no* parameters and *no* per-layer compute beyond one add. What I want is the property the relative family has —
position lives in the *score*, depends only on `t − i`, never enters the values, reasserts at every layer
— but with a bias-versus-distance function so simple it needs no parameters, no gather, and is defined,
smoothly, for *every* distance including ones I never trained on. The disease with sinusoidal was that the
signal past the training length was a novel high-dimensional pattern; the cure is a position signal whose
value at distance `d` is the obvious continuation of its values at small `d`, with no learned dial that
could be miscalibrated out of range. The simplest such function of distance is a straight line. So: add to
the query–key score, before the softmax, a bias that is *linear* in the relative distance `i − j`.

Let me justify "line" against the other monotone shapes I could pick, because the functional form is a
real choice and extrapolation is unforgiving of it. A logarithmic penalty `−m·log(1 + d)` grows too
slowly — the difference between distance 20 and distance 40 is `log(41) − log(21) ≈ 0.67`, barely half a
unit, so the far tail is nearly flat and the model gets almost no help distinguishing far positions out of
range. A quadratic or exponential penalty grows too fast — `e^{m d}` or `m d²` would make the OOD
distances, which are exactly the ones twice as large as training, blow the bias into a regime where the
softmax saturates to a one-hot on the nearest key, killing the long-range head `reverse` needs. The line
is the unique shape that is (a) monotone, (b) unbounded so it keeps discriminating arbitrarily far, and
(c) *self-similar under scaling* — the increment of penalty per step is a constant `m`, the same at
distance 3 and at distance 40, so there is no distinguished length scale where behaviour changes. That
self-similarity is the whole extrapolation argument: because the penalty per position is constant, distance
40 genuinely is "distance 20, continued," with no cliff and no flattening. And it needs no parameters —
the line is fixed by one slope per head, which I will set rather than learn. The line is not merely the
*simplest* choice; it is the one whose shape looks identical at every distance, which is exactly the
property rung two's phase patterns and rung three's `1/t` resolution both lacked.

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

Let me turn those four slopes into distances, because a slope is abstract but "how far can this head see
before its attention is halved" is exactly the quantity that decides whether `reverse` is reachable. A key
at distance `d` carries additive bias `−m·d`, which after the softmax's exponential multiplies its weight
by `e^{−m d}` relative to the adjacent key. The distance at which a head's weight is halved solves
`e^{−m d} = 1/2`, i.e. `d = ln 2 / m ≈ 0.693 / m`. For the four slopes: `m = 1/4` gives `d ≈ 2.8`,
`m = 1/16` gives `d ≈ 11`, `m = 1/64` gives `d ≈ 44`, and `m = 1/256` gives `d ≈ 177`. So the heads form a
ladder of reach — one head effectively local (half-weight by ~3 tokens), one mid-range (~11), one that
comfortably spans the training length (~44), and one whose half-weight distance (~177) sits far beyond even
the OOD ceiling. That top head is the one `reverse` lives or dies on: emitting the last output symbol
requires reaching back roughly `2L ≈ 80` positions at OOD length 40, and I can check what the penalty does
to each head at that distance. The local head (`m = 1/4`) applies `−80/4 = −20`, so `e^{−20} ≈ 2·10^{-9}` —
the far source token is annihilated, that head simply cannot reach the input start. The long-range head
(`m = 1/256`) applies `−80/256 = −0.3125`, so `e^{−0.3125} ≈ 0.73` — the far token keeps 73% of the weight
of an adjacent one, easily enough to attend to. So the geometric slope set is not decoration: it is exactly
what buys `reverse` a fighting chance, by keeping one head almost distance-blind while the others
specialise local. This is also the precise reason I expect `reverse` to be the *weakest* variant rather
than a total loss — it has one head that can reach, but only one, so it is running on a thin margin the
recency prior is actively taxing.

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

Let me trace the matrix on a `T = 3` example for one head of slope `m`, because the sign convention is
the one place this can go silently wrong and I want to see the staircase come out the right way round.
`idx = [0, 1, 2]`, and `rel_{ij} = idx[j] − idx[i] = j − i` gives the matrix rows `[0,1,2]`, `[-1,0,1]`,
`[-2,-1,0]` for `i = 0, 1, 2`. Multiply by `m`: `bias = m·rel`. Take the query at the last position
`i = 2`, which under the causal mask legitimately sees all of `j = 0, 1, 2`: its bias row is
`[−2m, −m, 0]`, so the farthest-back key `j = 0` gets `−2m`, the next `j = 1` gets `−m`, and the adjacent
`j = 2` gets `0` — monotonically more negative the farther back, exactly the recency staircase I wanted.
Now the query at `i = 0`: its row is `[0, +m, +2m]`, positive above the diagonal because `rel` is positive
where `j > i`. Those two positive entries are keys at *future* positions the causal mask forbids, and the
loop overwrites them with `−∞` before the softmax, so they contribute exactly zero to the attention weights
and their wrong-signed bias never reaches the output. That is why I do not need to mask the upper triangle
myself — the harness's own causal mask, applied after my bias, disposes of it. The trace confirms the sign
convention: `bias_{ij} = m·(j − i)` reads as `−m·(distance back)` on every row the model is actually
allowed to attend over.

One more numeric check on the scaling decision, since it interacts with the slope magnitudes I just chose.
The head dimension here is `head_dim = d_model / n_heads = 128 / 4 = 32`, so `√head_dim = √32 ≈ 5.66`. If
the bias were folded *inside* the `q·k / √head_dim` scaling, every slope would be effectively divided by
`5.66` — my `1/4` head would behave like a `1/22.6` head, and the whole carefully chosen half-attenuation
ladder (2.8, 11, 44, 177 tokens) would stretch out by a factor of `5.66` to roughly (16, 63, 250, 1000)
tokens, making even the "local" head span the entire sequence and destroying the local/long-range spread
that `reverse` and `delim` need to coexist. Adding the bias *after* the scaling keeps the slopes meaning
exactly what I designed them to mean in raw score-space, decoupled from `head_dim`. The harness does this
for free — `attn_scores = q·k / √head_dim` first, then `+ attn_bias` — so the ladder is preserved.

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
