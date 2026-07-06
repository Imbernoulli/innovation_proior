The single SLSQP run did its job and then named its own limitation in the cleanest possible way. It
lifted the sum from the grid's `2.5414` to `2.5949` — `+0.0535`, a real geometric gain — by doing
exactly the two things the grid could not: it made the radii unequal and it slid circles into the
corners and edges where a disk braced by free walls can grow past `0.1`. The optimizer is plainly the
right engine; its returned point was a clean KKT optimum feasible far inside tolerance
(`maxviol ~10⁻¹⁴`). And it stopped `~0.041` short of the `2.636` frontier not because SLSQP failed but
because one random start drops into one basin of a landscape with an enormous number of basins of very
different quality. The bottleneck is the *single initialization*, and the textbook cure for a
nonconvex problem whose entire difficulty is where you start is to start many times and keep the best.
So this rung is multi-start SLSQP: draw many random centre scatters, refine each with the same joint
centre-and-radius SLSQP, LP-re-tighten the radii, verify feasibility, keep the best sum seen. There is
no new algorithmic idea here beyond "restart," and I want to be honest that this rung buys its
improvement with compute and breadth, not with a cleverer search — which makes it all the more
important to understand precisely how fast that trade stops paying, because that is the only thing this
rung really has to teach.

I want to be precise about *why* restarting helps and about how fast it stops helping, because that
governs both the budget I should spend and the honest expectation I should set. Each random scatter,
refined to convergence, produces one local optimum whose sum-of-radii value I can treat as a draw from
some fixed distribution `F` — the distribution of SLSQP-optimum values induced by uniform starts. The
single-start rung handed me exactly one sample from `F`, and it read `2.5949`. Best-of-`K` restarts is
then the maximum order statistic of `K` independent draws from `F`, and essentially everything about
this rung's behaviour is a statement about how that maximum grows with `K`.

The growth is logarithmic, and it is worth doing the extreme-value arithmetic rather than waving at
"diminishing returns," because the specific rate decides where to stop. For a distribution with a
roughly Gaussian upper tail, the maximum of `K` samples sits about `√(2 ln K)` standard deviations
above the mean. That factor is the whole story: for `K = 10` it is `√(2·2.30) = 2.15`; for `K = 100`
it is `√(2·4.61) = 3.03`; for `K = 1000`, `√(2·6.91) = 3.72`. The increments are the tell — going from
`10` to `100` starts adds `0.88` of a standard deviation, and going from `100` to `1000`, a further
tenfold, adds only `0.69`. Each tenfold multiplication of compute buys a smaller absolute gain than
the last, and the curve has a distinct knee: the first hundred-ish starts capture most of what uniform
restarts can ever deliver, and everything past that is paying geometrically more compute for
arithmetically less climb. That is the quantitative reason to pick a budget in the low hundreds rather
than the thousands — I want to sit just past the knee, where the order statistic has mostly saturated,
and not burn compute chasing a tail that widens like `√(ln K)`.

So I set `K = 120` starts, and I can pin the expected climb to that same arithmetic, honestly and
without pretending to know `F`'s scale. The single draw at `2.5949` was one sample from the middle of
`F` — a typical draw, neither best nor worst. Best-of-`120` sits about `√(2 ln 120) = √9.57 ≈ 3.1`
standard deviations above `F`'s mean, so the improvement over a typical single draw is on the order of
two-to-three standard deviations of `F`. I do not know that standard deviation a priori, but I can
bound the outcome from both sides: the climb cannot exceed the `~0.041` gap to `2.636`, and the
order-statistic knee says most of the restart-reachable part of that gap is already captured by `120`
starts. So I expect best-of-`120` to clear the single-start value comfortably and settle into the low
`2.62`s — a real step toward the frontier bought entirely by sampling more basins — while stopping
visibly short of `2.636`. If the standard deviation of `F` is around `0.01`, that predicts a climb of
`~0.02`–`0.03` and a landing near `2.615`–`2.625`; the `n26` number will reveal the scale of `F` after
the fact, and I am deliberately predicting the band, not a digit.

The per-start cost, which I estimated on the previous rung, is what makes `K = 120` affordable. One
SLSQP descent costs a few million pairwise-distance evaluations and runs in a fraction of a second to a
couple of seconds in vectorised numpy, so `120` of them is a few minutes of wall clock — a budget I
can actually spend. And there is a deliberate trade inside that budget. I have three ways to spend a
fixed compute envelope, and I should price them rather than default to one. I could run a *few deep*
starts — say `12` starts at `maxiter = 2500` — but a basin is essentially determined by where the
scatter lands and is reached in the first hundred-odd iterations, so polishing each optimum to machine
precision spends the budget refining basins I will mostly discard, and `12` draws is far below the
order-statistic knee. I could run *very many shallow* starts — `1200` at `maxiter = 25` — but `25`
iterations is not enough to jam a `26`-disk configuration, so those runs return half-relaxed
arrangements whose radii the final LP cannot rescue into a real packing, and the sample would be drawn
from a *worse* distribution than `F`. The middle setting — `120` starts at `maxiter = 250` — sits
where each run reaches its basin and jams it adequately while the count sits just past the
order-statistic knee. So I cap each run at `maxiter = 250` rather than the previous rung's `300`,
spending the saved iterations on more starts, and I rely on the final LP re-tightening to recover the
radii cleanly regardless of exactly where SLSQP stopped. More, slightly coarser refinements dominate
fewer perfectly-converged ones when the entire goal is to *find a good basin* rather than to nail the
sixth digit of a basin I may throw away.

The rest of the mechanics carry over unchanged, and I want them unchanged so the only new thing this
rung adds is breadth. For each start I draw `26` centres uniformly in a slightly inset square
`[0.04, 0.96]²`, set the initial radii to their LP optimum for that scatter so SLSQP begins feasible,
run the joint SLSQP to a KKT point, re-tighten the radii by LP for the returned centres, check
feasibility against the real constraints, and keep the packing if it is legal and its sum beats the
running best. The inset plays exactly the role it did before — it keeps the initial circles off the
walls so the LP has a non-degenerate feasible start and SLSQP does not begin pinned in a constraint
corner — and I have widened it marginally to `0.04` only because the extra percent of interior gives
the scatters a touch more room to diversify without courting the boundary degeneracy. The explicit
feasibility re-check per start is not redundant with the LP: SLSQP can return a point a hair outside
its own constraint tolerance, and I want to keep only packings that pass the *real* checker at
`atol=1e-7`, so a start that lands marginally infeasible is discarded rather than reported. One master
seed is fixed so the whole `120`-start run is a single reproducible object; the reported number is the
deterministic output of that seed, not a lucky draw I cannot reproduce.

There is a pattern across the rungs I can already read off the two numbers I have, and it sets the
scale of what to expect here. The grid-to-single-start jump was `+0.0535`, a large step, because that
first optimization captured the biggest structural gains at once — it made the radii unequal and put
circles in the corners, the two moves worth the most. Multi-start cannot repeat a gain that has already
been banked; all it can do is choose a *better basin* than one random draw happened to find, and basin
quality varies far less than the grid-to-optimum gap did. So I expect the single-to-multistart
increment to be visibly *smaller* than `+0.0535` — on the order of `+0.02` to `+0.03` — and I expect
whatever rung follows this one to buy less still. That deceleration is not a disappointment; it is the
honest signature of a ladder approaching a frontier, where each successive rung works harder for a
thinner slice, and it is why the whole contest ultimately lives in the sixth-decimal band near `2.636`
rather than in the first or second decimal the early rungs moved.

It is worth grounding the abstract distribution `F` in the geometry of a single scatter, because that
picture explains where `F` sits and why. A uniform draw of `26` centres over the inset square of area
`0.92² = 0.8464` has point density `≈ 30.7` per unit area, so its typical nearest-neighbour spacing is
about `0.5/√30.7 ≈ 0.090` — half the grid's `0.2`. The radii LP can hand each circle at most about
half its gap to its nearest neighbour before that pair's `rᵢ + rⱼ ≤ dᵢⱼ` bites, so a typical circle
starts near radius `0.045` and the LP sum of a raw scatter lands down around `26 · 0.045 ≈ 1.2`,
comfortably *below* the grid's `2.5414`. And it is worse than that average suggests, because a scatter
clumps: the expected number of pairs closer than a distance `d` is `C(26,2) · πd²/0.8464`, which is
about `12` pairs within a tangency distance `0.1` and about `3` within `0.05`. So a typical start is a
cloud with a dozen knots and a matching set of voids — nothing like a packing — and SLSQP spends its
early iterations un-knotting clumps and sliding circles apart before it can grow anything. The whole
of a single descent is that lift, from a clumped `~1.2` cloud up through and past the grid floor into
the high `2.5`s. Multi-start is `120` such lifts, and `F` is the distribution of where they land; its
mean sits in the high `2.5`s precisely because every draw begins from the same blunt, clumped starting
quality and the descent can only do so much with it.

It is worth seeing how SLSQP dissolves those knots, because it explains why the engine is unbothered by
a bad start. At each iterate it linearises every too-close pair constraint
`‖cᵢ − cⱼ‖ − rᵢ − rⱼ ≥ 0` into a half-plane: the gradient of the distance with respect to `cᵢ` is the
unit vector `(cᵢ − cⱼ)/‖cᵢ − cⱼ‖` pointing from `j` to `i`, so the linear model says "to relieve this
overlap, move `cᵢ` along that direction or shrink the radii" — exactly the physically-correct repulsion
between two overlapping disks. The QP subproblem assembles all the active half-planes at once and takes
the step that satisfies them jointly, so a clump of several mutually-overlapping circles is pushed apart
in one coordinated move rather than one pair at a time. That is why even a scatter with a dozen knots
relaxes to a jammed packing in a few dozen iterations: SLSQP is not searching blindly, it is following
the linearised repulsion out of every clump simultaneously, and the only thing it cannot do is choose
*which* jammed configuration it relaxes into — that is fixed by where the scatter started, which is the
whole reason this rung multiplies the starts.

I can pin the expected climb to the area lever as a second, independent check on the prediction. By
Cauchy–Schwarz `Σ rᵢ² ≥ (Σ rᵢ)²/26`, so a landing in the low `2.62`s forces
`Σ rᵢ² ≥ 2.62²/26 ≈ 0.264`, above the grid's `0.25` and above single-start's implied
`2.5949²/26 ≈ 0.259`. In coverage terms that is the packing filling `π · 0.264 ≈ 0.83` of the square,
against the grid's `0.79` — best-of-`120` claws back another slice of the `21%` void the grid left, but
only a slice. The frontier's `2.636` demands `Σ rᵢ² ≥ 0.267`, more coverage still, and the `0.264`
versus `0.267` gap in the area lever is the very `0.014` in the sum that uniform breadth cannot buy: it
is the corner-heavy, coordinated coverage that a blunt scatter never seeds. So the plateau shows up as
an area deficit I can name before I run it, not merely as a number I fall short of.

Now the part I have to be honest about in advance, because it is the whole reason this is a rung and
not the endpoint: I expect it to *plateau* below `2.636`, and I can name the mechanism precisely from
what the earlier rungs already established. On the single-start rung I counted that a jammed `26`-disk
optimum is roughly isostatic — about `2N ≈ 52` active contacts drawn from the `429` constraints — and
that each distinct rigid contact graph is a distinct basin. The number of combinatorially-distinct
jammed graphs is astronomically large, and the basins that reach the very top of the frontier
correspond to a vanishingly small handful of them: specific irregular arrangements with a few large
disks in a particular pattern and the rest tuned as gap-fillers. A uniform random centre scatter has no
bias whatsoever toward those good graphs — it is as likely to seed a mediocre topology as a good one —
so the probability that any single draw lands in a top basin is tiny and, crucially, *fixed*.
Best-of-`K` improves the max by sampling the upper tail of `F` more thoroughly, but it cannot change
`F` itself, and if the top basins carry negligible probability mass under uniform scattering then no
affordable `K` reaches them. That is the precise sense in which random multi-start saturates: it is
drawing harder from a distribution whose good tail is thin, not shifting the distribution toward the
good region.

I can make the cost of trying to force it through pure restart concrete, which is what convinces me not
to. Suppose the top basins carry probability mass `q` per uniform draw. To hit one with probability
about a half needs `K ≈ 0.7/q` starts. If `q ~ 10⁻³` — one scatter in a thousand seeds a top basin —
that is `K ~ 700`; if `q ~ 10⁻⁴`, it is `K ~ 7000`; and the reward, once reached, is only the last
`~0.014` of sum. So the cost of reaching the frontier by restart grows like `1/q` while the payoff is
fixed and small, and by running `120` I am implicitly betting the good basins have mass above roughly a
percent — a bet I expect to lose, which is exactly why the plateau will appear. Cranking `K` is the
wrong lever: it pays `1/q` for a logarithmically-shrinking return. The right move, when it comes, will
not be to sample the same `F` harder but to change `F`.

Before I accept that `120` uniform starts is the right shape of experiment, I should ask whether a cheap
tweak to the *scatter* could shift `F` enough to matter, because if it could I would rather do that than
reach for a different method. The obvious candidate is to stop drawing centres independently and reject
any draw that lands too close to an existing one — Poisson-disk sampling — which removes the dozen clumps
a uniform scatter carries and hands the LP a more even, higher-radius start. It would help a little:
fewer clumps means SLSQP wastes fewer early iterations un-knotting, so the median refined optimum rises
and `F` shifts right. But it does not break the plateau, and I can say why precisely. Poisson-disk
sampling is still memoryless — every start is independent, so it is still a max order statistic that
saturates like `√(2 ln K)` — and it is still corner-blind, laying no special weight on the corners where
a disk braced by two free walls does the most work. It removes one of the two named weaknesses (clumping,
a mild symptom of "no structure") while leaving the other (no memory) untouched, so it raises the ceiling
of random multi-start without changing its kind. That is enough to tell me the plateau this rung
establishes is real, and that the fix is not a better *scatter* but a different *procedure* — one that
seeds corner structure and remembers its incumbent — which is a larger change than a single rung of
restarts should smuggle in.

Two further facts sharpen what I should watch for. First, the effective diversity of the sample is
smaller than `120`. Many uniform scatters funnel into the same generic jammed packing — a
slightly-irregular near-grid arrangement — so a good fraction of the `120` draws are redundant hits on
a few common basins, and the number of genuinely distinct basins explored is well below the raw count.
On top of that, the square's `8`-fold symmetry and the free relabelling of similar-sized circles mean
every basin comes with an orbit of mirror-and-permutation copies that all score identically, so some
draws "rediscover" a basin the run has already scored under a different labelling. Both effects thin
the effective sample and make the order statistic saturate even faster than the naive `√(2 ln K)`
suggests. Second, there is a compensating fact I can lean on: the maximum of many draws is far *less*
variable than a single draw, because extreme-value distributions concentrate as `K` grows. Where the
single-start `2.5949` was one sample carrying the full basin-lottery variance of `F`, the best-of-`120`
is a stable statistic — a different master seed would move it only slightly, since it is already
averaging over the good tail of many draws. So the number this rung reports is a trustworthy
measurement of "what uniform multi-start reaches at `n = 26`," not an artefact of one lucky or unlucky
seed, which is exactly the property I want from a rung whose job is to establish the ceiling of a whole
family of methods.

What, concretely, would confirm or falsify this reading when I run it? Three things. First, the climb:
I predict best-of-`120` lands in the low `2.62`s, above single-start's `2.5949` by roughly two-to-three
standard deviations of `F` and below the `2.636` frontier. Second, the *timing* of new bests, which I
can sharpen using the theory of records rather than just saying "sporadic." In an i.i.d. sequence the
expected number of running-record values in `K` draws is the harmonic number `H_K = Σ_{k≤K} 1/k`,
which for `K = 120` is `≈ ln 120 + 0.577 ≈ 5.4`. So I expect on the order of *five* new bests across
the run, arriving with the record signature — the `k`-th draw sets a new best with probability `1/k`,
so improvements cluster early and thin out, with the last new best likely somewhere deep in the run
rather than near the start. If instead the best were set in the first few starts and never beaten, that
would say `F`'s good tail is even thinner than I think; if new bests kept arriving at a steady rate
late, that would say I am nowhere near the knee and should spend more. Third, the gap: I expect the
distance to the frontier to shrink from single-start's `~0.041` to something on the order of a third to
a half of that, but to remain stubbornly open — the signature of a saturating order statistic, not a
closing one.

Two consistency checks anchor this order-statistic model to what I already know. At `K = 1` the
`√(2 ln K)` factor is exactly `0`, so best-of-one sits at the mean of `F` — which is just the statement
that the single-start `2.5949` was a typical draw, neither tail, exactly as I read it. And at the other
end the factor grows without bound in units of `F`'s standard deviation, but the *value* is capped by
`F`'s upper endpoint `v*`, the best any uniform start can reach after SLSQP; since `v*` sits below the
frontier, best-of-`K` presses toward `v*` and saturates there, never at `2.636`, however large `K`
grows. So the model reproduces the one measured point and predicts the plateau in the same breath. A
small implementation fact follows from the same feasibility logic: because the LP-radii start on any
scatter is always feasible, every start yields *some* legal packing, so the running best is set from
the first iteration, and a start on which SLSQP stalls or errors is simply not kept — the loop loses
one draw out of `120` rather than corrupting the answer.

If that plateau materialises as I expect, it will have named the next move without ambiguity. Random
restarts fail for two separable reasons, and best-of-`K` addresses neither. The starts have no
*structure*: a uniform scatter looks nothing like a good packing, so most refinements waste themselves
converging to mediocre basins, and the way to fix that is to seed starts that already resemble good
sum-of-radii packings — a few disks pushed into the corners where two free walls let them grow, and a
smooth non-clumping interior spread — so that even a typical refinement lands well. That is a shift of
the whole distribution `F` rightward, which beats sampling the old `F` harder. And the search has no
*memory*: every refined packing is thrown away, so the best configuration found is never used to seed
anything better, when the obvious thing to do is to *exploit* the incumbent — perturb the best packing
found and re-refine it, walking from a good basin toward a nearby better one rather than re-rolling the
dice from scratch. Those are the two levers this rung cannot pull, and the plateau it is about to
establish is precisely the wall they exist to break.
