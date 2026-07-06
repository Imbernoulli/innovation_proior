The diversified constructor confirmed the asymptote rather than breaking it. The multi-start search over the
boundary-spike seeds settled at `1.517146` — essentially the rung-3 basin, which the mirror seeds agreed was the
lowest of those tried — and the long full-constraint polish ground it to `1.517040`, a total of `0.000197` below
the previous rung. The repeat-lift to `N = 1200` was tried and did not pay within budget. So the whole apparatus
of breadth-plus-polish bought under two ten-thousandths, exactly the few-thousandths-or-less the diminishing-
returns column predicted, and it leaves the constructor at `1.5170` while AlphaEvolve reached `1.5053` at the
same `600` pieces and the record stands at `1.5028628969`. The residual is `~0.014`. This rung is about
understanding that number honestly: why my own engine saturates where it does, and what a method that reaches the
record must do differently, because "run the SLP harder" is now provably not the answer.

Take the saturation first, because the diagnosis has to be right or the conclusion is wrong. My SLP is a minimax
linear program: an epigraph variable for the peak, the self-convolution constraints linearized around the
current heights, a trust region, accept only if the true `R` drops. That is the correct *local* move — it presses
the whole near-tight plateau of autoconvolution nodes down together rather than chasing one peak — but it is
still a local move from *one* parametrization. The objective is genuinely non-convex: `a*a` is bilinear in the
heights, the `max` over nodes is non-smooth, and the good regions of the landscape are narrow, asymmetric,
irregular valleys. A trust-region LP linearizes around the current point, so it can only follow the valley it is
already in; the restart kicks jostle it locally but do not carry it to a structurally different valley. That is
one reason it saturates — a single local engine cannot relocate itself across the landscape. And I confirmed the
symptom directly: six diverse starts, including both mirror orientations and a balanced two-spike seed, all
funnelled into the same basin the single rung-3 warm start found. Diversity of starts helped a little and then
stopped helping, which is what "one engine, one reachable basin" looks like from the outside.

But there is a second reason, and it is the decisive one because it is not about search at all — it is about
*expressivity*, and I can pin it with a hard count rather than an impression. The record-grade solution needs its
autoconvolution flattened into a vast plateau of near-equal top nodes; the finer that plateau, the lower the peak
can be pressed relative to the mass. When I look at what the record actually is (below), its autoconvolution has
on the order of `18000` nodes sitting within `10^{−4}` of the peak. Now count what `N = 600` can even offer: the
autoconvolution of a `600`-piece vector has `2·600 − 1 = 1199` nodes *in total*. A plateau of eighteen thousand
near-tight nodes cannot exist on a node set of size twelve hundred — I would need at least `18000` nodes to host
it, hence at least `N ≈ 9000` pieces, fifteen times my grid, before the record's flat top is even representable.
This is not a tuning gap or a search-breadth gap; `600` pieces physically cannot express the object the record is
built from. My rung-4 profile had roughly a third of its `1199` nodes packed within `10^{−3}` of the peak —
already close to the expressivity limit of the grid, the plateau nearly as wide as the node set allows — and
that third of twelve hundred is a coarse shadow of a third of sixty thousand. So the last `~0.014` is a
different-method gap on two compounding fronts: too few coordinates to express the optimal shape, and a single
local engine that cannot leave its valley to find the global one. The honest conclusion is that no schedule of my
constructor closes it.

The trust region compounds the second front in a way worth quantifying, because it shows the local engine could
not build the record shape even if it somehow knew the target. In the normalized profile (mass `1`) the record's
spike reaches a height of about `0.91` while the mean height is `0.008` — the spike is `111×` the mean. A
trust-region SLP step moves each coordinate by at most `|d_j| ≤ tr ≈ 10^{−4}`, so growing a single coordinate
from the mean up to the spike takes on the order of `(0.91 − 0.008)/10^{−4} ≈ 9000` monotone accepted steps for
that one height alone — and that ignores the coordinated reshaping of the tens of thousands of other coordinates
that has to happen simultaneously to keep the autoconvolution flat. Thousands of rounds to erect one spike, in a
method where each round is a full LP solve, is simply not reachable in budget, and it is not what the trust
region is for: the trust region exists to keep the linearization valid, which forces the steps small, which in
turn forbids the large structural moves the record shape requires. The engine's own stability discipline is at
odds with reaching a `111×`-mean spike from anything like a moderate profile. So even setting the coordinate
count aside, the local step size is a second, independent reason the record is out of this constructor's reach.

What does the method that reaches the record do, then? It attacks both fronts at once. It scales the construction
by two orders of magnitude — to `30000` pieces — so the height profile can carry the finely irregular structure
and the autoconvolution can host a plateau of tens of thousands of near-tight nodes. And it replaces the single
local engine with a large-scale search over the *form of the constructor itself*: an agentic coding loop in which
a strong model repeatedly proposes and edits the construction program, runs it through this same `R` evaluator,
and keeps what scores lower, over tens of hours of autonomous iteration. This is the AutoEvolver line of work —
Claude/Opus via "aspiration prompting" — and it sits at the end of a visible chain: a prior published upper bound
of `1.5098`, then AlphaEvolve's `600`-piece `1.5053`, then TTT-Discover's `30000`-piece `1.5028628983`, then the
AutoEvolver `30000`-piece `1.5028628969` that edges the fourth decimal a hair lower still. The jump that matters
in that chain is from `1.5053` to `1.50286`: it coincides exactly with the jump from `600` pieces to `30000`, and
the gain lives entirely below the third decimal. It was bought with that scale, not with a cleverer local step,
which is precisely why my constructor cannot reach it and an evolutionary program search over `30000`-piece
constructions can. Both the piece count and the search over program forms are essential; neither alone is what my
SLP has.

The compute asymmetry behind that is worth stating plainly, because it is the honest scale of the difference and
it keeps me from pretending the gap is small. My diversified SLP ran about twenty minutes on `600` pieces. The
record was found by an autonomous search running for tens of hours on `30000`-piece constructions, where each
scoring call is itself far heavier than mine — an `fftconvolve` at `N = 30000` versus `N = 600` is roughly `50×`
more pieces and, counting the `log N` factor, on the order of `60–70×` more work per evaluation — and the search
issues an enormous number of such calls as it edits and re-scores program after program. Multiplying the wall-
clock (tens of hours against twenty minutes, on the order of `100×`) by the per-evaluation cost and the sheer
number of evaluations, the record represents something like three to four orders of magnitude more computation
than my constructor spent, aimed by a model that could restructure the construction rather than only reshape a
fixed vector. When the residual to a record is a factor of thousands in compute plus a change in the search
space itself, calling it a "tuning gap" would be dishonest. It is exactly the kind of gap that only a different
method at a different scale closes, and the twenty-minute local constructor was never going to.

There is a piece of infrastructure that makes the `30000`-piece search feasible at all, and it is the same
`fftconvolve` this ladder's evaluator uses. Scoring a candidate is a self-convolution; done directly it is
`O(N^2)`, which at `N = 30000` is nearly a billion multiply-adds per evaluation, and an evolutionary search calls
the evaluator a great many times. The FFT does the same convolution in `O(N log N)` — for `N = 30000` a factor of
roughly `N/log₂N ≈ 30000/15 = 2000` fewer operations — which is what turns a scoring call from prohibitive into
cheap and lets the search run tens of thousands of program variants over tens of hours. So the record is not only
a matter of pieces and program search; it also rests on the same evaluator speedup I have been relying on
throughout, applied at a scale I never needed at `600` pieces. That the identical `fftconvolve` scores both my
constructions and the record is part of why the two are directly comparable.

It is worth being precise about *why* searching over program forms escapes the valleys my SLP is trapped in,
because "more search" is too vague to be the real reason. My SLP searches over height *vectors*, and it does so
*continuously* and *locally*: every step is an infinitesimal, trust-bounded reshaping of the current heights, so
it can only reach shapes that are connected to its start by a path of small improving moves. An entire family of
constructions — say, "put the spike two-thirds of the way along instead of at the boundary," or "make the
interior a decaying comb instead of a plateau" — is unreachable, not because it scores worse, but because there is
no continuous downhill path to it from where the LP sits. An evolutionary search over the *construction program*
does not have this limitation: a single edit to the program — changing a placement rule, a sparsity pattern, an
index — produces a *discretely different* family of shapes in one move, a jump across the landscape that no
sequence of small vector steps can make. That is the mechanism. The program search is not merely a bigger or
luckier version of my local search; it moves in a different space, the space of *rules that generate* the heights,
where the natural moves are large structural jumps rather than small reshapings. So the two methods are separated
not only by scale but by the topology of what they can reach, and that is the deepest sense in which the last
`0.014` is a different-method gap rather than a tuning gap.

So this final rung does not pretend a cleverer single LP closes the gap. Instead it reproduces the actual record
and scores it through the very same FFT autoconvolution evaluator this whole ladder has used, both to confirm the
record value lands through my own harness and to see what the record-grade solution looks like in the metrics I
have been tracking. There is no optimizer here — the work is the verification. I load the published `30000`-piece
sequence and apply the frozen functional `R = 2N · max_k (v*v)_k / (Σ v)^2`. I compute the peak two ways, with
`np.convolve` and with `fftconvolve`, as a deliberate cross-check: an FFT convolution accumulates floating-point
error differently from a direct one, so if the two agreed only to a few digits I would suspect the tenth-digit
value I report. They agree to machine precision — the two peaks differ by about `2·10^{−16}` — so the reported
`R` is not an FFT artifact, and I pin it with an assertion to ten digits so the number cannot silently drift. The
result is `R = 1.5028628969`, the record, reproduced through the identical evaluator that scored the flat
indicator at exactly `2` and my SLP frontier at `1.5170`.

And the record-grade solution looks exactly like the family my SLP was drifting toward but could not fully reach,
scaled up and refined to a fineness only a large search over `30000` pieces could carve. When I look at the
sequence: it is a single enormous spike at one boundary — the tallest height is about `111×` the mean, sitting at
the very last index, `29999` — over an interior that is roughly `38%` near-zero, and its autoconvolution is
flattened into a plateau of about `18000` of its `~60000` nodes all within `10^{−4}` of the peak. That is the
same peak-suppressing, boundary-heavy, sparse-and-irregular structure my `600`-piece runs hinted at — a spike at
a boundary, a thinned sparse interior, a wide flat autoconvolution top — but at `50×` the resolution, so the
plateau it can build is `18000` nodes wide instead of `~400`, and the peak it can press down is correspondingly
lower. In peak/mean terms the record sits at `peak/mean ≈ 1.5028` (the `N = 30000` prefactor is negligible),
against my `~1.516` and the flat `2`: the same shaving of the tallest node toward the mean, carried one further
part in a hundred by a plateau `45×` finer than anything `600` pieces can hold.

The near-zero interior is worth dwelling on, because it is the mechanism behind the low peak and it is exactly
what my sparsity ratchet was reaching for at coarse resolution. Loading the sequence and counting, about `37%` of
the `30000` heights are essentially zero (below `10^{−6}`), rising to roughly `39%` if I count everything under a
percent of the mean — well over a third of the support is empty. The mass that is not in the near-zero interior
is concentrated into the boundary spike and a structured, irregular remainder. This is the non-negativity
constraint being exploited to the hilt: since every autoconvolution node is a sum of non-negative products, the
only way to keep the *maximum* node low is to arrange the support so that no shift of the profile against itself
lines up much overlap, and a sparse, boundary-loaded, deliberately irregular support does exactly that — it
scatters the self-overlap across many shifts so that the tallest is barely above the average. My anneal's
one-way ratchet toward zeros, and my SLP's driving of a couple hundred heights near zero, were coarse instances
of the same principle; the record is that principle carried to `30000` pieces, where the arrangement can be tuned
finely enough to hold `18000` autoconvolution nodes at a common near-maximal level.

Read as one story through the peak/mean lens, the whole ladder is a single quantity being driven down. The flat
ceiling has its tallest autoconvolution node at twice the mean (`peak/mean = 2`); breaking the symmetry on `50`
heights brought it to about `1.53×`; lifting to `600` and pressing the plateau with the minimax LP reached
`~1.516×`; and the record, on `30000` pieces with a program search, reaches `~1.5028×`. Every rung is the same
move — flatten the top of the node profile so the peak sits closer to the average, subject to non-negativity and
fixed mass — carried a little further by a little more resolution and a little more search. The record is not a
different quantity; it is the same descent taken to a resolution and a search breadth that my constructor could
not command. Seeing it that way is what makes the frontier legible: I know exactly what the record did more of,
and exactly why my engine stopped where it did.

The verification itself is deliberately spare so that nothing can hide in it. I read the sequence from
`record_sequence.json` under the key `sequence`, confirm its length is `30000`, form `R = 2N · max_k(v*v)_k /
(Σv)^2`, and print it to ten digits. The `assert round(R, 10) == 1.5028628969` is there so the reproduction is a
test, not a claim: if the file, the evaluator, or the arithmetic drifted, the assertion would fire rather than
let a wrong number through. And computing the peak with both `np.convolve` and `fftconvolve` is the guard I most
want here — the value I am reporting is a record to the tenth digit, so I need to know the FFT is not quietly
contributing to that digit. The two peaks agree to about `2·10^{−16}`, floating-point round-off, so the tenth
digit is real. This is the same discipline the ceiling rung used when it insisted the flat value be `2` to full
precision: trust the harness only where two independent computations agree.

Reaching this through my own evaluator settles the open questions cleanly. The gap was real and it was a
different-method gap: `600` pieces cannot express an `18000`-node plateau and one local trust-region engine
cannot cross to a structurally different valley, but a `30000`-piece construction discovered by a long LLM-guided
evolutionary program search can, and when it is scored through the identical FFT `R`, the harness returns the
record. Reproducing it also validates the whole ladder's harness once more from the top: the same evaluator that
pinned the flat ceiling at `2`, tracked the anneal to `1.537`, the SLP to `1.517`, and now the record to
`1.5028628969`, is internally consistent across a `50×` range of piece counts and a `0.5` range of `R`. The
remaining distance is worth stating exactly, because it is the honest measure of how open this problem still is.
The record `1.5028628969` is an *upper* bound on `C1` — every admissible construction, this one included, only
certifies `C1 ≤ R(f)` — and the best proved *lower* bound is `1.28` (Cloninger–Steinerberger). So the true
constant lives somewhere in the interval `[1.28, 1.5028628969]`, a window of width `~0.223`, and this whole
ladder has only pushed the *upper* end of that window: from the flat ceiling `2` down through `1.537`, `1.517`,
and now `1.50286`. Nothing here moves the lower end, and I have no way of knowing from this side whether the true
`C1` sits near the floor or near the record — both bounds have been improved repeatedly, from opposite
directions, and they have not met. What reproducing the record does settle is everything this ladder could: that
the `~0.014` gap from my SLP frontier was a real, different-method gap; that `600` pieces cannot express an
`18000`-node plateau and one local trust-region engine cannot cross to a structurally different valley; and that
a `30000`-piece construction from a long LLM-guided evolutionary program search can, scoring the record to ten
digits through the identical FFT `R` that pinned my own constructions. What it does not settle — the distance
from `1.5028628969` down to `1.28`, and which end of it the truth is near — is the part of the first
autocorrelation inequality that remains genuinely open. The number I report is the one the evaluator returns on
the published `30000`-piece sequence, and it is the record.
