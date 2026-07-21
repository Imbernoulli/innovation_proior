The diversified constructor confirmed the asymptote rather than breaking it. The multi-start over the
boundary-spike seeds settled at `1.517146` — essentially the rung-3 basin, which the mirror seeds agreed was the
lowest of those tried — and the long full-constraint polish ground it to `1.517040`, a total of `0.000197`
below the previous rung, with the `N = 1200` repeat-lift not paying within budget. So the whole apparatus of
breadth-plus-polish bought under two ten-thousandths, exactly the few-thousandths-or-less the diminishing-
returns column predicted, and it leaves the constructor at `1.5170` while AlphaEvolve reached `1.5053` at the
same `600` pieces and the record stands at `1.5028628969`. The residual is `~0.014`. This rung is about
understanding that number honestly: why my own engine saturates where it does, and what a method reaching the
record must do differently, because "run the SLP harder" is now provably not the answer.

Take the saturation first. My SLP is a minimax linear program — epigraph variable, linearized self-convolution
constraints, trust region, accept only if true `R` drops. That is the correct *local* move, pressing the whole
near-tight plateau down together, but it is still a local move from *one* parametrization. The objective is
non-convex, its good regions narrow, asymmetric, irregular valleys, and a trust-region LP can only follow the
valley it is already in; the restart kicks jostle it locally but do not carry it to a structurally different
valley. I confirmed the symptom directly — six diverse starts, both mirror orientations and a balanced
two-spike seed, all funnelled into the same basin the single rung-3 warm start found. Diversity helped a little
and then stopped, which is what "one engine, one reachable basin" looks like from outside.

But there is a second, decisive reason that is not about search at all — it is *expressivity*, and I can pin it
with a hard count. The record-grade solution needs its autoconvolution flattened into a vast plateau of
near-equal top nodes; the finer that plateau, the lower the peak can be pressed. When I look at the record
sequence (below), its autoconvolution has on the order of `18000` nodes within `10^{−4}` of the peak. But the
autoconvolution of a `600`-piece vector has `2·600 − 1 = 1199` nodes *in total*. A plateau of eighteen thousand
near-tight nodes cannot exist on a node set of size twelve hundred — I would need at least `N ≈ 9000` pieces,
fifteen times my grid, before the record's flat top is even representable. This is not a tuning gap or a
search-breadth gap; `600` pieces physically cannot express the object the record is built from. My rung-4
profile already packed roughly a third of its `1199` nodes within `10^{−3}` of the peak — near the expressivity
limit of the grid — and that third of twelve hundred is a coarse shadow of a third of sixty thousand.

The trust region compounds this, and it shows the local engine could not build the record shape even if it knew
the target. In the normalized profile the record's spike reaches height `~0.91` while the mean is `~0.008` — a
spike `111×` the mean. A trust-region step moves each coordinate by at most `|d_j| ≤ tr ≈ 10^{−4}`, so growing
one coordinate from the mean up to the spike takes on the order of `(0.91 − 0.008)/10^{−4} ≈ 9000` monotone
accepted steps for that height alone — ignoring the coordinated reshaping of tens of thousands of others that
must happen simultaneously to keep the autoconvolution flat. Thousands of full-LP rounds to erect one spike is
not reachable in budget, and it is not what the trust region is for: it exists to keep the linearization valid,
which forces the steps small, which forbids the large structural moves the record shape requires. So even
setting the coordinate count aside, the local step size is an independent reason the record is out of reach.

What does the method that reaches the record do? It attacks both fronts at once. It scales the construction to
`30000` pieces, so the profile can carry the finely irregular structure and the autoconvolution can host a
plateau of tens of thousands of near-tight nodes. And it replaces the single local engine with a large-scale
search over the *form of the constructor itself*: an agentic coding loop where a strong model repeatedly
proposes and edits the construction program, scores it through this same `R` evaluator, and keeps what scores
lower, over tens of hours. This is the AutoEvolver line — Claude/Opus via "aspiration prompting" — at the end of
a visible chain: a prior published `1.5098`, then AlphaEvolve's `600`-piece `1.5053`, then TTT-Discover's
`30000`-piece `1.5028628983`, then the AutoEvolver `30000`-piece `1.5028628969`. The jump that matters is
`1.5053 → 1.50286`, and it coincides exactly with `600 → 30000` pieces: the gain lives entirely below the third
decimal and was bought with scale, not a cleverer local step.

Why does searching over program forms escape the valleys my SLP is trapped in, when "more search" is too vague
to be the reason? My SLP searches over height *vectors*, continuously and locally — every step an infinitesimal
trust-bounded reshaping — so it can only reach shapes connected to its start by a path of small improving moves.
An entire family of constructions ("put the spike two-thirds along instead of at the boundary," "make the
interior a decaying comb instead of a plateau") is unreachable, not because it scores worse but because no
continuous downhill path leads to it. An evolutionary search over the construction *program* has no such limit:
a single edit — a placement rule, a sparsity pattern, an index — produces a discretely different family of
shapes in one move, a jump across the landscape no sequence of small vector steps can make. That is the
mechanism, and it separates the two methods not only by scale but by the topology of what they can reach — the
deepest sense in which the last `0.014` is a different-method gap rather than a tuning gap. (The scale gap is
itself steep: twenty minutes on `600` pieces against tens of hours on `30000`-piece constructions, where each
scoring call is `~60×` heavier and the search issues an enormous number of them — three to four orders of
magnitude more computation, aimed by a model that could restructure the constructor rather than reshape a fixed
vector. And it rests on the same `fftconvolve` this evaluator uses, whose `O(N log N)` is what makes a
`30000`-piece scoring call cheap enough to issue tens of thousands of times.)

So this final rung does not pretend a cleverer single LP closes the gap. It reproduces the actual record and
scores it through the very same FFT autoconvolution evaluator this ladder has used — there is no optimizer here,
the work is the verification. I load the published `30000`-piece sequence from `record_sequence.json` under the
key `sequence`, confirm its length is `30000`, and apply the frozen functional `R = 2N · max_k(v*v)_k /
(Σ v)^2`. I compute the peak two ways, with `np.convolve` and with `fftconvolve`, as a deliberate cross-check:
an FFT convolution accumulates floating-point error differently from a direct one, so if the two agreed only to
a few digits I would suspect the tenth-digit value I report. They agree to about `2·10^{−16}` — machine
round-off — so the reported `R` is not an FFT artifact, and I pin it with an assertion to ten digits so the
number cannot silently drift. The result is `R = 1.5028628969`, the record, reproduced through the identical
evaluator that scored the flat indicator at exactly `2` and my SLP frontier at `1.5170`.

And the record-grade solution looks exactly like the family my SLP was drifting toward but could not reach,
scaled up and refined. Loading the sequence: a single enormous spike at one boundary — the tallest height
`~111×` the mean, at the last index `29999` — over an interior roughly `38%` near-zero (about `37%` below
`10^{−6}`, rising to `~39%` under a percent of the mean), its autoconvolution flattened into a plateau of about
`18000` of its `~60000` nodes within `10^{−4}` of the peak. That is the same peak-suppressing, boundary-heavy,
sparse-and-irregular structure my `600`-piece runs hinted at, but at `50×` the resolution, so the plateau is
`18000` nodes wide instead of `~400` and the peak it presses down is correspondingly lower. The near-zero
interior is the mechanism, and it is exactly what the sparsity ratchet was reaching for at coarse resolution:
because every autoconvolution node is a sum of non-negative products, the only way to keep the *maximum* node
low is to arrange the support so no shift of the profile against itself lines up much overlap, and a sparse,
boundary-loaded, deliberately irregular support scatters the self-overlap across many shifts so the tallest is
barely above the average. My anneal's one-way ratchet toward zeros and my SLP's driving of a couple hundred
heights near zero were coarse instances of the same principle, carried here to `30000` pieces where the
arrangement can be tuned finely enough to hold `18000` nodes at a common near-maximal level.

Read as one story through the peak/mean lens, the whole ladder is a single quantity driven down: the flat
ceiling at `peak/mean = 2`, breaking the symmetry on `50` heights to `~1.53×`, the minimax LP at `600` reaching
`~1.516×`, and the record at `30000` reaching `~1.5028×`. Every rung is the same move — flatten the top of the
node profile so the peak sits closer to the average, under non-negativity and fixed mass — carried further by a
little more resolution and a little more search. The record is not a different quantity; it is the same descent
at a resolution and search breadth my constructor could not command.

The remaining distance is the honest measure of how open this problem still is. `1.5028628969` is an *upper*
bound on `C1` — every admissible construction, this one included, only certifies `C1 ≤ R(f)` — and the best
proved *lower* bound is `1.28` (Cloninger–Steinerberger). So the true constant lives somewhere in `[1.28,
1.5028628969]`, a window of width `~0.223`, and this whole ladder has pushed only its *upper* end: from `2` down
through `1.537`, `1.517`, and now `1.50286`. Nothing here moves the lower end, and I cannot tell from this side
whether the true `C1` sits near the floor or near the record. What reproducing the record settles is everything
this ladder could: that the `~0.014` gap from my SLP frontier was a real different-method gap — `600` pieces
cannot express an `18000`-node plateau and one local trust-region engine cannot cross to a structurally
different valley — and that a `30000`-piece construction from a long LLM-guided evolutionary program search can,
scoring the record to ten digits through the identical FFT `R` that pinned my own constructions. The distance
from `1.5028628969` down to `1.28` is the part of the first autocorrelation inequality that remains genuinely
open.
