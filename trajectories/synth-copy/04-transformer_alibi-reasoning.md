NoPE broke the wall, and the numbers say both that the diagnosis was right and where it is still soft.
Deleting the positional code lifted OOD off zero on every variant: `exact_match_ood` went from sinusoidal's
`0.0 / 0.0 / 0.001` to `0.297 / 0.550 / 0.403`, and OOD token accuracy from `0.071 / 0.066 / 0.031` to
`0.433 / 0.757 / 0.434` — past the LSTM, past everything. In-distribution held perfect (`0.998 / 0.996 /
1.0`), elapsed stayed ~130 s, and the geometric-mean score jumped from `0.50` to `0.706`. So absoluteness
*was* the disease: with no absolute code to fall off the end of, the mask-derived position let SGD find a
relative-ish rule that recurs at unseen lengths. But the per-variant OOD numbers expose what NoPE leaves on
the table. `repeat` extrapolates best (`0.550` exact, `0.757` token) — its offsets track the seam in a
local, periodic way. `delim` and `reverse` are the weaker pair (`0.297`/`0.403` exact, both token accuracies
near `0.43`), and that is the tell: NoPE recovers position by *construction* — the layer-one `1/t` count
anchored at `BOS`, recoded into a relative score — but that code is something SGD has to discover and keep
stable across 30–40 unseen positions with no inductive bias pushing it toward the right shape. It works, but
it is *learned from nothing*, exactly the kind of thing that holds at length 25 and frays at length 40. The
next move: stop *hoping* the model induces a good relative code and *give* it one — a relative bias whose
value at an unseen distance is the obvious continuation of its value at seen distances, so there is no
out-of-distribution regime to fray in.

Where to aim it is set by the summary being a *geometric* mean, which is dominated by its smallest factor:
the weakest variant drags the whole number down more than a strong one lifts it. NoPE's smallest factor is
`delim` at score `0.648`, pulled down by its OOD exact of only `0.297`. So the most valuable thing I can do
is raise the *weakest* variant's OOD exact match; a gain there moves the geomean more than an equal gain on
`repeat`, already the strong factor at `0.773`. The target is not "beat NoPE on average" but "lift NoPE's
soft variants, `delim` and `reverse`, without giving back too much elsewhere."

So I want a relative scheme, but rung three already named the failure modes of the explicit family. A
learned relative-bias table (T5-style) fixes a bucketing and cutoff whose far bucket is trained only on the
sparse in-range far distances and then asked to govern the dense far distances that only appear at OOD — and
it adds parameters and a per-layer gather. A rotation scheme is relative-by-construction but injects
per-layer rotations of `q`, `k` inside every attention sublayer and carries the same fixed frequency
schedule the sinusoid had. So the relative family splits into "cheap and shapeless" (NoPE, already tested)
and "parametric or compute-heavy prescriptions" (T5, rotation). What I want is the missing corner: a
prescription with the *right* shape but *no* parameters and *no* per-layer compute beyond one add — position
in the *score*, depending only on `t − i`, reasserting at every layer, with a bias-versus-distance function
so simple it is defined smoothly for every distance including ones never trained on. The simplest such
function is a straight line, so I add to the query–key score, before the softmax, a bias *linear* in the
relative distance.

"Line" against the other monotone shapes is a real choice, and extrapolation is unforgiving of it. A
logarithmic penalty `−m·log(1 + d)` grows too slowly — distances 20 and 40 differ by `log(41) − log(21) ≈
0.67`, barely half a unit, so the far tail is nearly flat and gives almost no help distinguishing far
positions out of range. A quadratic or exponential penalty grows too fast — at the OOD distances, twice as
large as training, `e^{m d}` or `m d²` blows the bias into a regime where the softmax saturates to one-hot on
the nearest key, killing the long-range head `reverse` needs. The line is the unique shape that is monotone,
unbounded (so it keeps discriminating arbitrarily far), and *self-similar under scaling*: the penalty per
step is a constant `m`, the same at distance 3 and 40, so there is no distinguished length scale where
behaviour changes. That self-similarity is the whole extrapolation argument — distance 40 genuinely is
"distance 20, continued," with no cliff and no flattening — and it needs no parameters, just one slope per
head, set rather than learned. It is the property rung two's phase patterns and rung three's `1/t`
resolution both lacked.

I want a recency bias — distant keys matter less, all else equal — with the same shape at every distance, so
distance 40 is "distance 4 but more so," never a regime the model has not conceptually seen. So make the
bias *more negative* the farther back the key. For a causal query at position `i` over keys `j ≤ i`, the
relative distance is `i − j ≥ 0`, and I subtract a penalty proportional to it: the adjacent key (distance 0)
gets 0, one step back `−m`, two steps back `−2m`. This keeps the structural properties NoPE showed matter:
position lives in the scores, not the values (the bias is added to scores, values untouched, so the output
carries no absolute-position component), it injects at every attention sublayer where the sinusoidal add
happened only once at the bottom, and it depends only on `i − j`, the quantity that recurs at unseen
lengths. It is the explicit, well-shaped version of what NoPE groped toward — but instead of one learned
recency shape I get a *spread* of them, because the slope is per-head.

The slope needs care. One slope for all heads would force every head to the same recency preference, which
is wrong — some heads should look locally, some keep a long view, and for `reverse` I need at least one head
that reaches far back without being crushed. So `m` is per-head, and the `H` slopes should span a range of
recency scales, with more resolution near zero where the precious long-range heads live — which a geometric
sequence gives (equal ratios = equal log-spacing, dense near the small end). The clean choice is start =
ratio = `2^{−8/n}`; for 4 heads that is `2^{−2} = 1/4`, giving slopes `1/4, 1/16, 1/64, 1/256`.

Turning those into distances is what decides whether `reverse` is reachable. A key at distance `d` carries
bias `−m·d`, so after the softmax its weight is multiplied by `e^{−m d}`; the half-weight distance solves
`e^{−m d} = 1/2`, `d = ln 2 / m ≈ 0.693/m`. For the four slopes: `m = 1/4 → d ≈ 2.8`, `1/16 → 11`, `1/64 →
44`, `1/256 → 177`. So the heads form a ladder of reach — one effectively local, one mid-range, one that
spans the training length, one whose half-weight distance sits far beyond the OOD ceiling. That top head is
what `reverse` lives or dies on: emitting the last output symbol needs to reach back `~2L ≈ 80` positions at
OOD length 40. The local head applies `−80/4 = −20`, so `e^{−20} ≈ 2·10^{-9}` — the far source token is
annihilated, that head cannot reach the input start. The long-range head applies `−80/256 = −0.31`, so
`e^{−0.31} ≈ 0.73` — the far token keeps 73% of an adjacent one's weight, enough to attend to. So the
geometric slope set is exactly what buys `reverse` a fighting chance, and it is the precise reason I expect
`reverse` to be the *weakest* variant rather than a total loss: it has one head that can reach, but only
one, so it runs on a thin margin the recency prior is actively taxing.

Should the slopes be learned? For extrapolation, no — NoPE is the cautionary tale. Learned slopes would be
tuned to the distances seen in `[0, 20)`, with no training signal about behaviour at the far distances that
only appear at evaluation, so they could overfit the in-range recency statistics and generalise *worse* out
of range — the same "learned from nothing reliable past the training length" weakness I am fixing. A fixed
geometric set is the same line everywhere and cannot be miscalibrated. So I fix the slopes before training,
cached as a parameter-free buffer in a tiny `nn.Module` so they follow the model to the GPU, registered via
`scheme.extra_modules`, carrying no gradient.

One scaling subtlety. Standard attention divides scores by `√head_dim = √32 ≈ 5.66` to tame the variance of
a random dot product, but my bias is not a random dot product — it is a deterministic penalty I choose
directly in score-space. If it were folded *inside* that scaling, every slope would be divided by 5.66, my
`1/4` head would behave like `1/22.6`, and the whole half-attenuation ladder (2.8, 11, 44, 177) would stretch
by 5.66× to roughly (16, 63, 250, 1000), making even the "local" head span the entire sequence and
destroying the local/long-range spread that `reverse` and `delim` need to coexist. So the bias is added
*after* the `√head_dim` scaling. In this harness that is automatic — `CausalSelfAttention` computes
`attn_scores = q·k / √head_dim` first, then adds `scheme.attn_bias(T, …)` — so the ladder is preserved and
the slopes mean exactly what I designed them to.

Now the concrete fill, where the harness version differs from the canonical implementation. The `attn_bias`
hook returns a `[H, T, T]` tensor added to the scores *before* the causal mask (the loop applies the
lower-triangular `-inf` mask right after adding the bias). The canonical ALiBi exploits softmax's per-row
shift-invariance to replace the honest staircase with a single broadcast row `m·[0,1,…,T−1]`, since the two
differ only by a per-row constant softmax kills. This harness does **not** take that shortcut: it builds the
full signed relative-distance matrix `rel = idx[None,:] − idx[:,None]` (so `rel_{ij} = j − i`, negative below
the diagonal) and sets `bias = slope · rel`, i.e. `bias_{ij} = m·(j − i) = −m·(i − j)` — more negative for
keys farther back, exactly as the math says. Above the diagonal `rel` is positive, but those entries are
overwritten by the loop's `-inf` causal mask before the softmax, so they never reach the output and I do not
mask them myself. On the query at the last position of a `T = 3` block, the bias row is `[−2m, −m, 0]`:
farthest-back key `−2m`, next `−m`, adjacent `0` — the recency staircase the right way round, `−m·(distance
back)` on every row the model is actually allowed to attend over. `build_positional_scheme` returns the
scheme with only `attn_bias` set (`token_embedding_extra` and `rotary` are `None`); `build_model` returns the
plain `SeqModel(use_lstm=False)`; values and embeddings stay position-free. The attention weights are
identical to canonical ALiBi — only the arithmetic path (full matrix, no shift trick, two separate adds)
differs.

My falsifiable expectations against the NoPE numbers. In-distribution should stay perfect on all three (the
bias is negligible at short distances). The whole bet is OOD, and the prediction is specific: ALiBi should
beat NoPE where NoPE's *learned* relative code was softest — the biggest lift on `delim` (NoPE only `0.297`)
and a solid lift on `repeat`, because the linear recency bias is precisely the well-shaped relative prior
NoPE had to induce from scratch, defined identically at length 40 as at length 15. The risk I have to name
is `reverse`: the recency bias is a *monotone* preference for nearby tokens, but `reverse` requires attending
from the end of the output all the way back to the start of the input — the longest offset, exactly the
direction the penalty fights. So `reverse` is where a built-in recency prior could backfire, and I expect it
to be ALiBi's weakest variant, possibly at or below NoPE there, with the long-range head (`slope = 1/256`)
doing the heavy lifting if it works at all. The honest claim is aggregate, not uniform: ALiBi should lift the
geometric-mean score above NoPE's `0.706` by winning clearly on `delim`/`repeat` even if it pays a little on
`reverse`. Were `reverse` to collapse far enough to drag the geomean below NoPE, that would falsify "monotone
recency is good enough here" and point the next step toward a non-monotone relative scheme. (The full
scaffold module is in the answer.)
