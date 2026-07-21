The inscribed eleven-gon gave me `0.0215`, a bit over half the record, and its lesson was blunt: no
closed-form configuration I can name reproduces the irregular `1/27` optimum, so I have to start
*searching*. Before I reach for anything with a temperature or a gradient, I want to know what the raw
landscape hands me for free, because that number is what every clever method has to beat — if I skip it
I will not know whether a later method's gains come from being clever or merely from trying more
configurations. The simplest search, needing no schedule and no acceptance rule, is to throw darts:
sample many point sets uniformly in the square, score each exactly, keep the best minimum-area. It is
brute force but honest and trivially correct — there is no way to implement `max over samples` wrong —
and it reads the landscape cleanly before I invest in machinery.

It is completely stateless: each trial is independent, and there is no memory beyond the single
incumbent, so there is no way to get *stuck*. The flip side, which is the defect I will diagnose at the
end, is that statelessness also means no way to *improve* a good draw. But first the number, and the
only real decision is the budget, which I should make large and affordable.

Affordability forces the one non-trivial implementation choice, because the naive loop is hopeless: the
exact evaluator loops over `165` triples per configuration, and four million configurations is
`≈ 6.6×10⁸` cross-product evaluations — minutes to hours in pure Python. The fix is to vectorize:
precompute the `165` index triples once, then for a whole batch gather the three vertices of every
triple, compute all `165` cross products in one array expression, reduce along the triples axis with a
`min` to get each configuration's score, and read off the batch maximum. I cannot make the batch the
whole four million, though — a `(4×10⁶, 165)` float64 area array alone is `~5.3 GB`, and the gathered
vertex arrays several times that. Batching at `B = 50,000` brings the working set to a few hundred
megabytes and processes `80` sequential batches, finishing in under a minute. I fix the RNG seed so the
number is reproducible.

Now the honest question: how good do I *expect* random multi-start to be? A configuration of eleven
points is a single point in *twenty-two* dimensions, and the fat-minimum configurations occupy a
vanishingly small corner. The difficulty has a specific shape. A *typical* triangle from three uniform
points is not small — its expected area is the Sylvester value `11/144 ≈ 0.076`, twice the record. So
random triangles are not thin on average; the problem is that I take a `min` over `165` of them, and
with that many chances it is overwhelmingly likely some trio falls nearly on a line, and that one
sliver sets the score.

I can turn that into a quantitative prediction with extreme-value theory. Call a configuration's
minimum-triangle-area `M`; random multi-start reports the max of `M` over `N` draws, essentially the
upper `(1 − 1/N)` quantile, so I need the upper tail of `M`. For three uniform points the area density
is finite and positive at `a = 0`, so `P(A < t) ≈ f_A(0)·t` for small `t`. If the `165` areas behaved
independently, `P(M > t) ≈ (1 − f_A(0)t)^{165} ≈ exp(−165·f_A(0)·t)` — so near zero `M` is
approximately exponential with rate `λ` of order `165·f_A(0)`. I can anchor `f_A(0)` independently from
the Sylvester mean: a positive quantity with mean `0.076` has density-at-origin of order
`1/0.076 ≈ 13`, so `f_A(0) ~ 10`, giving nominal rate `165·10 ≈ 1.6×10³`, pulled down toward `~10³` by
triple correlation (each point sits in `45` triples, so the effective number of independent chances is
below `165`). So `λ ~ 10³` from first principles.

Two consequences, and I care far more about the second. First, `M` is bounded above by the best
possible configuration `≈ 0.037`, so best-of-`N` can only approach that asymptote. Second, the *rate*:
with `P(M > t) ≈ exp(−λt)`, the value best-of-`N` reaches is where `exp(−λt*) = 1/N`, i.e.
`t* ≈ ln(N)/λ`. The best-so-far grows like the *logarithm* of the budget — a brutal law of diminishing
returns. A tenfold increase from four to forty million adds only `ln 10 = 2.3` to `ln N = 15.2`, a
`15%` gain, which is exactly why I do not bother with forty million draws: four million squeezes most
of what raw sampling offers while finishing in under a minute.

The log law also answers, before I run anything, whether random search is even in the game — and the
answer is damning. With `λ` of order `10³`, to merely *tie* the structured eleven-gon at `0.0215` I
would need `ln N = λ·0.0215 ≈ 20`–`30`, so `N` in the billions to trillions just to match a one-line
baseline; to reach the record `0.037` takes astronomically more. The compute to get anywhere grows
*exponentially* in the target value, so within any realistic budget random multi-start is pinned in the
low `0.01`s: a *typical* single draw's worst triangle sits near `1/λ ~ 10⁻³`, a near-collinear sliver a
few percent of the record, and best-of-`4M` lifts it only by the factor `ln(4M) ≈ 15`. It is a floor,
not a method.

The model's soft spots blur the constant but not the scaling. The `165` areas are *not* independent —
each point sits in `45` triples, so near-degeneracies are correlated, which lowers the effective `λ`
and nudges `t*` up a little; and the `t* ≈ ln(N)/λ` form is a large-`N` asymptotic, trustworthy only
for `N ≫ 1`, which four million is. Neither touches the log-in-`N` scaling that carries the whole
argument.

The magnitude follows from two anchors. The eleven-gon at `0.0215` earns its number by a *structural*
guarantee — no three points ever collinear — whereas a uniform draw has no such protection and almost
always carries a near-collinear sliver; the log law says it would take trillions of draws to beat one
collinearity-free symmetric configuration, so I expect random *below* `0.0215`. And best-of-millions
sits far above a single draw's near-zero median. So my falsifiable prediction is the low `0.01`s —
around a third of the record or less and, the sharper call, *below* the eleven-gon. If the feedback
shows random above `0.0215`, my whole picture of why the `min` objective is hard is wrong. I also
expect the running-best curve to reach most of its height in the first million draws and then flatten
into the `ln N` crawl.

Before accepting uniform sampling I should ask whether a smarter *sampler* changes the story. Two
tempting variants: a low-discrepancy (Sobol) sequence, or rejection sampling that throws away draws
with points too close together. Both would slightly raise the *average* draw by suppressing obvious
clustering — but neither touches the actual limitation. The killer is not clustering of points, it is
near-collinearity of *triples*: three well-separated, evenly-spread points can still be nearly
collinear, and a *perfectly* even fill is a lattice, which the previous rung showed is riddled with
exactly collinear triples — the space-filling instinct actively creates the degeneracy. Rejection on
pairwise distance is worse than useless because collinearity is a three-point property invisible to any
pairwise screen. And both variants are still sampling: they discard every configuration the instant the
next is scored, so the extreme-value ceiling applies with essentially the same `λ`. So I keep the
sampler dumb and uniform.

That last defect — random multi-start wastes all of its information — is the whole opening for the next
rung. It finds a decent configuration and immediately forgets it; the log-in-`N` ceiling is a *direct*
consequence of that amnesia, because with no way to walk from a `0.011` draw toward a better neighbour,
all I can do is wait for a luckier independent draw. What I want instead is to *hold onto* a good
configuration and nudge it — move one point, keep the nudge if the minimum triangle grew. That is local
search, and the danger with the greedy version is specific to this objective: because the score is the
`min` over `165` triangles, fixing the worst triangle by moving a point almost always shrinks another,
which becomes the new worst, so a greedy climber trades one sliver for another and freezes. The
standard cure is to accept some worsening moves with a probability that cools over time — simulated
annealing. So the next rung keeps the random starts but adds the two things this rung provably lacks:
memory, and a rule for moving one point toward a larger minimum triangle. If that story is right it
should not creep up the `ln N` crawl — it should jump clean past the eleven-gon in a single run.
