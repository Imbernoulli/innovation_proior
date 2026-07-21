The single SLSQP run did its job and named its own limitation. It lifted the sum from the grid's
`2.5414` to `2.5949` — `+0.0535`, a real geometric gain — by making the radii unequal and sliding
circles into the corners and edges where a disk braced by free walls grows past `0.1`. The optimizer
is plainly the right engine; its returned point was a clean KKT optimum feasible far inside tolerance
(`maxviol ~10⁻¹⁴`). It stopped `~0.041` short of `2.636` not because SLSQP failed but because one
random start drops into one basin of a landscape with an enormous number of basins of very different
quality. The bottleneck is the *single initialization*, and the textbook cure for a nonconvex problem
whose entire difficulty is where you start is to start many times and keep the best. So this rung is
multi-start SLSQP: draw many random centre scatters, refine each with the same joint SLSQP,
LP-re-tighten, verify feasibility, keep the best. There is no new algorithmic idea beyond "restart,"
so what this rung really has to teach is precisely how fast that trade stops paying.

Each random scatter, refined to convergence, produces one local optimum whose value I treat as a draw
from a fixed distribution `F` — the distribution of SLSQP-optimum values induced by uniform starts.
The single-start rung handed me exactly one sample from `F`, reading `2.5949`. Best-of-`K` restarts is
the maximum of `K` independent draws from `F`, and everything about this rung is a statement about how
that maximum grows. For a distribution with a roughly Gaussian upper tail, the max of `K` samples sits
about `√(2 ln K)` standard deviations above the mean: `√(2·2.30) = 2.15` at `K = 10`, `3.03` at
`K = 100`, `3.72` at `K = 1000`. The increments are the tell — `10 → 100` adds `0.88σ`, `100 → 1000`
adds only `0.69σ`. Each tenfold of compute buys less than the last, with a distinct knee: the first
hundred-odd starts capture most of what uniform restarts can ever deliver. So I sit just past the knee
at `K = 120` rather than chase a tail that widens like `√(ln K)`.

I can pin the expected climb to that arithmetic without pretending to know `F`'s scale. The single
draw at `2.5949` was one sample from the middle of `F`. Best-of-`120` sits about `√(2 ln 120) ≈ 3.1`
standard deviations above `F`'s mean, so the improvement over a typical draw is on the order of
two-to-three standard deviations of `F`. I cannot compute that standard deviation a priori, but I can
bound the outcome: the climb cannot exceed the `~0.041` gap to `2.636`, and the knee says most of the
restart-reachable part is already captured by `120` starts. So I expect best-of-`120` to clear
`2.5949` comfortably and settle in the low `2.62`s, stopping visibly short of `2.636` — a real step
bought entirely by sampling more basins. I am predicting the band, not a digit; the number will reveal
`F`'s scale after the fact.

The per-start cost makes `K = 120` affordable: one SLSQP descent runs in a fraction of a second to a
couple of seconds, so `120` is a few minutes of wall clock. And there is a deliberate trade inside that
budget. A few deep starts — say `12` at `maxiter = 2500` — polish each optimum to machine precision,
but a basin is essentially determined by where the scatter lands and reached in the first hundred-odd
iterations, so this refines basins I will mostly discard, and `12` draws is far below the knee. Very
many shallow starts — `1200` at `maxiter = 25` — return half-relaxed arrangements the final LP cannot
rescue into real packings, drawing from a *worse* distribution than `F`. The middle setting — `120` at
`maxiter = 250` — reaches each basin and jams it adequately while sitting just past the knee. So I cap
each run at `250` (down from the previous rung's `300`), spend the saved iterations on more starts, and
rely on the final LP re-tightening to recover radii cleanly regardless of exactly where SLSQP stopped.
More, slightly coarser refinements dominate fewer perfect ones when the goal is to *find a good basin*,
not to nail the sixth digit of one I may throw away.

The rest of the mechanics carry over unchanged, so the only new thing this rung adds is breadth. For
each start I draw `26` centres uniformly in `[0.04, 0.96]²`, set the initial radii to their LP optimum
so SLSQP begins feasible, run the joint SLSQP to a KKT point, re-tighten the radii by LP, check
feasibility against the real constraints, and keep the packing if legal and better than the running
best. The inset plays the same role as before, widened marginally to `0.04` only to give scatters a
touch more room. The explicit feasibility re-check is not redundant with the LP — SLSQP can return a
point a hair outside its own constraint tolerance, and I keep only packings that pass the real checker
at `atol=1e-7`. One master seed makes the whole `120`-start run one reproducible object.

It is worth grounding `F` in the geometry of a single scatter, because that explains where `F` sits. A
uniform draw of `26` centres over the inset square of area `0.8464` has density `≈ 30.7` per unit area,
so its typical nearest-neighbour spacing is about `0.5/√30.7 ≈ 0.090` — half the grid's `0.2`. The
radii LP can hand each circle at most about half its gap to its nearest neighbour, so a typical circle
starts near radius `0.045` and the LP sum of a raw scatter lands down around `26 · 0.045 ≈ 1.2`, well
below the grid's `2.5`. And a scatter clumps: the expected number of pairs closer than `d` is
`C(26,2)·πd²/0.8464`, about `12` pairs within tangency distance `0.1` and about `3` within `0.05`. So a
typical start is a cloud with a dozen knots and matching voids, nothing like a packing, and SLSQP
spends its early iterations un-knotting clumps before it can grow anything — it linearises each
too-close pair into the physically-correct repulsion and pushes a whole clump apart in one coordinated
QP step. Multi-start is `120` such lifts; `F` is the distribution of where they land, and its mean sits
in the high `2.5`s precisely because every draw begins from the same blunt, clumped quality.

Now the part I have to be honest about in advance, because it is the whole reason this is a rung and
not the endpoint: I expect it to *plateau* below `2.636`, and I can name the mechanism. A jammed
`26`-disk optimum is roughly isostatic — about `2N ≈ 52` active contacts — and each distinct rigid
contact graph is a distinct basin. The number of such graphs is astronomically large, and the basins
reaching the top of the frontier are a vanishingly small handful: specific irregular arrangements with
a few large disks in a particular pattern and the rest tuned as gap-fillers. A uniform scatter has no
bias toward those good graphs, so the probability that any single draw lands in a top basin is tiny and
*fixed*. Best-of-`K` samples the upper tail of `F` more thoroughly but cannot change `F`, and if the
top basins carry negligible mass then no affordable `K` reaches them. Concretely, if the top basins
carry mass `q` per draw, hitting one with probability `~½` needs `K ≈ 0.7/q`: `q ~ 10⁻³` gives
`K ~ 700`, `q ~ 10⁻⁴` gives `K ~ 7000`, and the reward is only the last `~0.014` of sum. The cost of
reaching the frontier by restart grows like `1/q` while the payoff is fixed and small, so cranking `K`
is the wrong lever. The right move, when it comes, will not be to sample the same `F` harder but to
change `F`.

I should ask whether a cheaper tweak to the *scatter* could shift `F` enough to matter, before reaching
for a different method. The obvious candidate is Poisson-disk sampling — reject any draw too close to an
existing one, removing the clumps and handing the LP a higher-radius start. It would help a little:
fewer clumps means fewer wasted early iterations, so `F` shifts right. But it does not break the
plateau, because it is still memoryless — every start independent, still a max order statistic
saturating like `√(2 ln K)` — and still corner-blind, laying no special weight on the corners where a
disk on two free walls does the most work. It removes one named weakness (clumping) while leaving the
other (no memory) untouched. So the fix is not a better *scatter* but a different *procedure* — one that
seeds corner structure and remembers its incumbent — which is a larger change than a single rung of
restarts should smuggle in.

Two further facts sharpen what to watch. The effective diversity is smaller than `120`: many scatters
funnel into the same generic near-grid jammed packing, and the square's `8`-fold symmetry plus free
relabelling means every basin comes with an orbit of identically-scoring copies, so some draws
rediscover a basin already scored — both thin the effective sample and make the order statistic
saturate even faster than `√(2 ln K)`. Against that, the maximum of many draws is far *less* variable
than a single draw — extreme-value distributions concentrate as `K` grows — so best-of-`120` is a
stable statistic, not an artefact of one lucky seed, exactly the property I want from a rung whose job
is to establish the ceiling of a whole family of methods.

What would confirm or falsify this reading? Three things. The climb: best-of-`120` in the low `2.62`s,
above `2.5949` by two-to-three `σ` of `F` and below `2.636`. The timing of new bests: in an i.i.d.
sequence the expected number of running records in `K` draws is the harmonic number
`H_120 ≈ ln 120 + 0.577 ≈ 5.4`, so I expect on the order of five new bests, arriving with the record
signature — the `k`-th draw sets a new best with probability `1/k`, so improvements cluster early and
thin out, the last likely deep in the run. And the gap: I expect the distance to the frontier to shrink
from `~0.041` to a third-to-half of that but stay stubbornly open — the signature of a saturating order
statistic, not a closing one. The model reproduces its two endpoints too: at `K = 1` the `√(2 ln K)`
factor is `0`, so best-of-one sits at `F`'s mean, which is just the statement that `2.5949` was a
typical draw; and as `K` grows the value is capped by `F`'s upper endpoint `v*`, which sits below the
frontier, so best-of-`K` presses toward `v*` and saturates there, never at `2.636`.

If that plateau materialises it names the next move. Random restarts fail for two separable reasons that
best-of-`K` addresses neither. The starts have no *structure* — a uniform scatter looks nothing like a
good packing — and the fix is to seed starts that already resemble good sum-of-radii packings (a few
disks in the corners, a smooth non-clumping interior), which shifts the whole distribution `F`
rightward, beating sampling the old `F` harder. And the search has no *memory* — every refined packing
is thrown away — when the obvious thing is to exploit the incumbent: perturb the best packing found and
re-refine, walking from a good basin toward a nearby better one rather than re-rolling the dice. Those
are the two levers this rung cannot pull, and the plateau it establishes is precisely the wall they
exist to break.
