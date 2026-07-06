The inscribed eleven-gon gave me `0.0215`, a bit over half the record, and the lesson from it was
blunt: no closed-form configuration I can name reproduces the irregular `1/27` optimum, so I have to
start *searching* — actually trying configurations and letting the evaluator pick the best. Before I
reach for anything with a temperature or a gradient, I want to know what the raw landscape hands me
for free, because that number is the real thing every clever method has to beat, and if I skip it I
will not know whether a later method's gains come from being clever or merely from trying more
configurations than the baseline did. The very simplest search, the one that needs no schedule and no
acceptance rule, is to throw darts: sample many point sets uniformly at random in the square, score
each one exactly, and keep the best minimum-area I ever see. It is brute force, but it is honest, it
is trivially correct — there is no way to implement `max over samples` wrong — and it gives me a clean
reading of the landscape before I invest in anything with moving parts.

The appeal is that it is completely stateless. Every trial is independent: draw eleven points
uniformly in `[0,1]²`, compute the minimum over all `165` triangles, and if that minimum beats my
running best, store the configuration; otherwise discard it and draw again. There is no acceptance
rule, no temperature, no memory beyond the single incumbent, and therefore no way to get *stuck* — a
random dart cannot fall into a local trap because it does not care where the previous dart landed. The
flip side of that virtue is the defect I will diagnose at the end: statelessness also means no way to
*improve* a good draw. But first I want the number, and the only real decision standing between me and
it is the budget, so I should make the budget large and make it affordable.

Affordability forces the one non-trivial implementation choice, because the naive loop is hopeless.
The exact evaluator loops over `165` triples per configuration, and if I want millions of
configurations that is `4{,}000{,}000 × 165 ≈ 6.6 × 10⁸` small cross-product evaluations. In a pure
Python loop, at the interpreter's rough throughput of order a million simple operations per second
once you count the per-triple overhead, that is on the order of ten minutes to hours — absurd for a
baseline I want to run and forget. The fix is to vectorize the whole thing. I precompute the `165`
index triples once as three integer arrays `I, J, K`, and then for a whole *batch* of `B`
configurations at once I gather the three vertices of every triple, compute all `165` cross products
`(b_x−a_x)(c_y−a_y) − (c_x−a_x)(b_y−a_y)` in a single array expression, take half the absolute value
for the areas, reduce along the triples axis with a `min` to get each configuration's score, and read
off the batch maximum. A batch of tens of thousands of configurations collapses into a handful of
array operations, and NumPy runs those at hardware speed — the `6.6 × 10⁸` cross products become a few
sweeps over contiguous float64 arrays, which a single core clears in tens of seconds, so I expect the
whole run in well under a minute rather than the loop's hours.

But I cannot make the batch the whole four million, because of memory, and the arithmetic decides the
batch size for me. A single `(4{,}000{,}000, 165)` float64 array of areas is `4×10⁶ · 165 · 8 ≈ 5.3`
gigabytes, and that is before the three gathered vertex arrays of shape `(4×10⁶, 165, 2)`, each
another `10.6` gigabytes — the full-fat approach needs tens of gigabytes and will not fit. Batching at
`B = 50{,}000` brings the areas array to `50{,}000 · 165 · 8 ≈ 66` megabytes and each gathered vertex
array to `50{,}000 · 165 · 2 · 8 ≈ 132` megabytes, so the working set per batch is a few hundred
megabytes — comfortable — and I process `4{,}000{,}000 / 50{,}000 = 80` batches in sequence. That
keeps the memory bounded while still amortizing the per-call overhead over fifty thousand
configurations, and it finishes in well under a minute, the right scale for a baseline I want to
trust. I fix the RNG seed so the reported number is reproducible.

Now the honest question, which is really the whole point of running this rung: how good do I *expect*
random multi-start to be? My worry is the curse of dimensionality, and I can make it concrete. A
configuration of eleven points is a single point in *twenty-two* dimensions, and the configurations
with a fat minimum triangle occupy a vanishingly small corner of that space. It helps to separate two
scales. A *typical* triangle from three uniform points in the square is not small at all — the
expected area of one such triangle is the classical Sylvester value `11/144 ≈ 0.0764`, which is *twice*
the record `1/27 = 0.037` and far above any sliver. So the problem is not that random triangles are
thin on average; the difficulty has a very specific shape. The record is asking that even the
*thinnest* of `165` triangles be about half the *mean* of a single random triangle — and I am taking
a `min` over `165` of them, where the count of triples grows like `n³/6`. With `165` chances, it is
overwhelmingly likely that *some* trio happens to fall nearly on a line — three points that nearly line
up make a sliver — and that one sliver, being the minimum, sets the score. The expected smallest
triangle among many is governed by the *lower* tail of the area distribution, which has positive
density all the way down to zero, so the smallest of `165` correlated triangles is routinely a tiny
fraction of the record. Random sampling therefore spends almost all its trials on configurations worth
almost nothing, and only occasionally stumbles onto a lucky draw in which no trio happens to be too
collinear.

I can push that from a worry into a quantitative prediction, and this is the part I actually care
about, because it tells me *how* the best-so-far will grow with the budget and where it will stall.
The right frame is extreme-value theory. Fix a configuration's minimum-triangle-area and call it `M`;
random multi-start reports the maximum of `M` over `N` independent draws, which is essentially the
upper `(1 − 1/N)` quantile of `M`'s distribution. So I need the *upper* tail of `M`. Here is the model
I trust for small `M`: for three uniform points the area density `f_A(a)` is finite and positive at
`a = 0` (a random triangle is degenerate with vanishing but nonzero density), so the probability a
single triangle is below a small threshold `t` grows linearly, `P(A < t) ≈ f_A(0)·t`. If the `165`
triangle areas behaved independently, then `P(M > t) = P(A > t)^{165} ≈ (1 − f_A(0) t)^{165} ≈
exp(−165 · f_A(0) · t)`. So `M`'s distribution near zero is approximately *exponential*, with some rate
`λ` of order `165 · f_A(0)`. That single fact drives everything.

I want at least an independent order-of-magnitude for `f_A(0)` so that `λ` is a *prediction*, not a
number I quietly back-fit. The one moment I hold is the Sylvester mean, `E[A] = 11/144 ≈ 0.0764`. If
the single-triangle area distribution were even roughly exponential with that mean, its density at the
origin would be `1/E[A] ≈ 13`; the true area density is not exactly exponential, but for a positive
quantity concentrated near a mean of order `0.076` a density at zero of order ten is the right scale.
Take `f_A(0) ~ 10`. Then the *nominal* rate is `165 · 10 ≈ 1.6×10³`, and the triple-correlation
discount (each point spoiling `45` triangles at once cuts the effective number of independent chances
well below `165`) pulls that down toward `~10³`. So the model predicts `λ ~ 10³` from first principles,
which is the same `1.4×10³` I will read back out of the `0.011` estimate below — the two routes agree
to a factor of order one, which is all I need. `λ` of order a thousand is the load-bearing constant,
and it comes out of the Sylvester mean, not out of the answer.

Two consequences follow, and I care about the second far more than the first. First, the ceiling: `M`
is bounded above by the best possible configuration, `≈ 0.037`, so best-of-`N` can only *approach* that
asymptote, never break it — random sampling cannot exceed the true optimum no matter how large `N` is.
Second, and this is the limiting one, the *rate* of approach. With an exponential tail `P(M > t) ≈
exp(−λ t)`, the value the best-of-`N` reaches is the `t*` where `P(M > t*) ≈ 1/N`, i.e.
`exp(−λ t*) = 1/N`, which gives `t* ≈ ln(N)/λ`. The best-so-far grows like the *logarithm* of the
budget. That is a brutal law of diminishing returns, and I can put numbers on it. `ln(4{,}000{,}000)
= 15.2`. Doubling to eight million adds `ln 2 = 0.69` to the numerator — a `4.5%` bump. A full
*tenfold* increase to forty million adds only `ln 10 = 2.30`, a `2.30/15.2 = 15%` gain in `t*`. So
each order of magnitude of extra compute buys the same small absolute slice of tail and a shrinking
relative one. This is exactly why I do not bother with forty million draws: the tail law says
best-of-`40M` would beat best-of-`4M` by about `15%`, not by a factor, so four million is the sweet
spot where I have squeezed most of what raw sampling offers while still finishing in under a minute.
I therefore expect the best-so-far curve to rise quickly over the first million draws and then visibly
flatten into a `ln N` crawl.

The logarithmic law also lets me answer, before running anything, the two questions that decide
whether random search is even in the game — and the answers are damning. Suppose my `4M`-draw best
lands around `t* ≈ 0.011` (I will justify that magnitude in a moment). Then `λ = ln(4M)/0.011 ≈
15.2/0.011 ≈ 1.4 × 10³`. Now invert the log law to ask how many draws it would take just to *tie the
structured eleven-gon* at `0.0215`: I need `ln N = λ · 0.0215 ≈ 1.4×10³ · 0.0215 ≈ 29.7`, so `N ≈
e^{29.7} ≈ 8 × 10¹²` — eight trillion draws, two million times my budget, merely to match a
parameter-free baseline I wrote down in one line. And to reach the record `0.037`? `ln N ≈ 1.4×10³ ·
0.037 ≈ 51.7`, so `N ≈ e^{51.7} ≈ 1.6 × 10²²` draws — astronomically, permanently out of reach. This
is the sharpest statement of why random multi-start is a floor and not a method: it is not that it
*cannot* approach the optimum, it is that the compute to get anywhere near it grows *exponentially* in
the target value, so within any realistic budget it is pinned in the low `0.01`s. The eleven-gon's
`0.0215` is safe from random search by a factor of a trillion draws.

That same rate `λ ≈ 1.4×10³` gives me two more concrete pictures I can check my prediction against,
both just re-readings of `P(M > t) ≈ exp(−λt)`. The *typical* single draw's minimum is where the
expected number of triangles below threshold crosses one — the expected count below `t` is `λt`, so it
crosses one at `t ≈ 1/λ ≈ 1/1400 ≈ 0.0007`. A random configuration's worst triangle is therefore
around seven ten-thousandths, a sliver about `2%` of the record, which is exactly the near-collinear
accident I keep describing. Best-of-`4M` lifts that typical `0.0007` up to `≈ 0.011`, and the lift
factor is `0.011/0.0007 ≈ 15`, which is precisely `ln(4M) = 15.2` — the log law is nothing but "the
best of `N` draws pushes the min out by a factor `ln N` past the typical draw." And the same `λt`
count says why the *average* configuration is hopeless at the target: at `t =` the record `0.037`, the
expected number of triangles below it is `λ · 0.037 ≈ 1400 · 0.037 ≈ 52`. So a typical random draw has
on the order of *fifty* of its `165` triangles already thinner than the record's single worst triangle;
its minimum is the thinnest of those fifty, hence far below `0.037`. Random sampling is not losing by a
little — the median configuration fails the target dozens of times over, and I am asking the luckiest
of four million to fail zero times, which the exponential tail says essentially never happens.

So I know the *shape* of the best-so-far curve before I run it, and that shape is itself a falsifiable
test. If `t* ≈ ln(N)/λ`, then plotting the running best against `ln N` should be a straight line of
slope `1/λ`: the best rises fast while `ln N` is still gaining quickly (the first million draws take
`ln N` from `0` to `13.8`), and then crawls, because getting `ln N` from `13.8` to `15.2` requires the
last three million draws. So I predict a curve that reaches most of its final height in the first
million or so and then visibly flattens, still creeping but barely, through the remaining three
million. If instead the curve is still climbing *steeply* at four million, my `λ` is too large and the
landscape is friendlier than the independent-tail model claims; if it has gone completely flat well
before the budget, `λ` is even larger and random search is even more hopeless than I think. Either way
the `min`-area number the feedback reports, read together with where the climb saturated, tells me
whether the extreme-value picture is right.

I should be honest about the two soft spots in that model, because I do not want to oversell a
back-of-envelope. First, the `165` triangle areas are *not* independent — they share the eleven
points, and each point sits in `C(10,2) = 45` triples, so the near-degeneracies are strongly
correlated (one badly-placed point spoils forty-five triangles at once). Correlation reduces the
*effective* number of independent chances below `165`, which lowers the effective rate `λ` and nudges
`t*` upward a little — so the true random-search number may sit a touch above my exponential-model
estimate, but the *log-in-N scaling* is unchanged, and that scaling is the whole argument. Second, the
model is a large-`N` asymptotic: at `N = 1` it wrongly predicts `t* = ln(1)/λ = 0`, when the real
single-draw median min is small but positive. So I only trust `t* ≈ ln(N)/λ` in the regime `N ≫ 1`,
which four million comfortably is. Neither caveat rescues random search; they only blur the constant,
not the verdict.

Where does the magnitude `≈ 0.011` come from, independent of the fitting I just did? From two anchors I
already hold. The upper anchor is the eleven-gon at `0.0215`: that configuration earns its number by a
*structural* guarantee — no three points ever collinear, worst triangle a controlled `R² sin α (1−cos
α)` — whereas a uniform random draw has no such protection and almost always carries a near-collinear
sliver somewhere among its `165` triples. It would be very surprising for the best of a few million
*unprotected* draws to beat one carefully collinearity-free symmetric configuration; the log law above
says it would need trillions of draws to. So I expect random *below* `0.0215`. The lower anchor is that
random must still clear the noise floor by a wide margin — best-of-millions is far above a single
draw's near-zero median. Splitting those, my honest, falsifiable prediction is that random multi-start
lands in the neighborhood of `0.010` to `0.012`, a fraction of the record around `0.29` to `0.32`, and
— the riskier, sharper call — *below* the eleven-gon rather than above it. If the feedback shows random
*above* `0.0215`, my whole picture of why the `min` objective is hard, and of the log-in-`N` ceiling,
is wrong. I expect it below, with the climb saturating well before the budget runs out.

Before I accept pure uniform sampling as the baseline I should ask whether a smarter *sampler* changes
the story, because if it did I would use it instead. Two tempting variants: draw from a low-discrepancy
(Sobol) sequence so the points are more evenly spread, or use rejection sampling to throw away any draw
whose points come too close together. Both would slightly raise the *average* draw by suppressing the
most obvious clustering — but neither touches the actual limitation, and the eleven-gon already told me
why. The killer here is not clustering of points, it is near-collinearity of *triples*, and a
low-discrepancy fill of the square does not prevent it: three well-separated, evenly-spread points can
still be nearly collinear, and indeed a *perfectly* even fill is a lattice, which from the previous
rung I know is riddled with *exactly* collinear triples — the space-filling instinct actively
*creates* the degeneracy the `min` punishes. Rejection on pairwise distance is worse than useless here
because it tests pairs, not triples, and collinearity is a three-point property invisible to any
pairwise screen. More fundamentally, both variants are still *sampling*: they still discard every
configuration the instant the next is scored, so they still have no mechanism to improve a promising
draw, and the extreme-value ceiling `t* ≈ ln(N)/λ` still applies with essentially the same `λ`. They
would buy a small constant factor at the cost of turning the honest zero-knowledge baseline into
something with tuned machinery, which defeats the purpose. So I keep the sampler dumb and uniform, and
let the number stand as the cleanest possible reading of the raw landscape.

I can even sanity-check the extreme-value frame against a case I can reason about exactly, to be sure I
have the direction of the limit right. Take `n = 3`: then there is a single triple, `M` *is* that one
triangle's area, and best-of-`N` is the upper quantile of a single uniform-triangle area, which climbs
toward the *largest* triangle three points can span in the square — half the square, area `1/2`. So at
`n = 3` random search saturates high, because there is no `min`-over-many to drag it down. As `n` grows
the `min` runs over more and more chances to find a sliver, the tail rate `λ ∝ n³/6` blows up, and the
same log law `t* ≈ ln(N)/λ` sends the achievable value down like `1/n³` at fixed budget. At `n = 11`
that collapse is already severe, and it is exactly the regime where the best-of-`4M` sits far below
even a hand-named baseline. The limit check confirms the sign of every dependence in the model: more
points, more triples, smaller achievable `min`; more draws, only a `ln N` crawl upward. Nothing here
rescues sampling.

That last defect — that random multi-start wastes all of its information — is the whole opening for the
next rung, and I can already see the shape of it. Random sampling finds a decent configuration and then
immediately forgets it and draws a fresh random one; every good draw is thrown away the moment the next
is scored, and nothing carries forward. The log-in-`N` ceiling is a *direct* consequence of that
amnesia: with no way to walk from a `0.011` draw toward a better neighbour, all I can do is wait for an
even luckier independent draw, and luck of that size is exponentially rare. What I want instead is to
*hold onto* a good configuration and nudge it — move one point a little, see whether the minimum
triangle got bigger, and if so keep the nudge. That is local search, and the danger with the naive
version, greedy hill-climbing, is specific to this objective: because the score is the *minimum* over
`165` triangles, fixing the current worst triangle by moving a point almost always shrinks some *other*
triangle, which becomes the new worst, so a purely greedy climber trades one sliver for another and
freezes almost immediately in a shallow trap. The standard cure is to accept some worsening moves on
purpose, with a probability that cools over time — simulated annealing — so the search can walk out of
shallow traps early and settle into a deep basin late. So the next rung keeps the random *starting*
points but adds the two things this rung provably lacks: memory, and a principled rule for moving one
point at a time toward a larger minimum triangle. If the memory-plus-improvement story is right, it
should not creep up the `ln N` crawl — it should jump clean past the eleven-gon and the trillion-draw
wall in a single run, because it climbs a landscape instead of resampling one.
