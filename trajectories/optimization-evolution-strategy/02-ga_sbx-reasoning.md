The DE numbers came back and they are a clean confirmation of exactly the worry I flagged, plus one
surprise that is worse than I feared. Read the *mean* against the *seed-42* value, because the gap
between them is itself a measurement. Ackley at seed 42: 0.488 — not the near-zero I hoped for, but
close. On the mean it is 10.0, and that jump is structure, not noise: if one seed is 0.488 and the mean
is 10.008, the other two seeds average `(3·10.008 − 0.488)/2 ≈ 14.76`. Ackley's plateau sits around 20
(the exponential basin term vanishes far from the origin and only the constant offset survives), so a
value near 14.8 is a search that came partway off the plateau and stalled. Two seeds partway down, one at
the bottom — a bimodal, basin-found-or-not outcome, exactly what a slow `rand` base does on a flat
plateau where every difference vector points across an essentially constant surface. Rosenbrock tells the
same story louder: seed 42 is 32.2, respectable — the anisotropic difference vectors did track the curved
valley as predicted — but the mean is 48302, forcing the other two seeds to average
`(3·48302 − 32.2)/2 ≈ 72437`, a population that never entered the valley and is still scattered across the
box. So DE's real weakness on the smooth problems is variance, not the median case: when it finds the
structure it exploits it beautifully, and when the `rand` base fails to seed the basin it scatters and
the mean explodes.

And Rastrigin is the catastrophe I called, with no bimodality to hide behind — uniformly bad: 172 at 30D
(mean 257) and **926** at 100D, where the single seed equals the mean, so this is the method's honest
level, not a lucky-seed story. Turn those residuals into geometry. On Rastrigin a coordinate at integer
pit `m` contributes `m²`, a coordinate solved to the origin contributes 0. So the 30D residual of 172
over 30 coordinates is an average `m²` of about `5.7`, the typical coordinate stranded two to three
lattice steps out; the mean 257 pushes that to `8.6`. At 100D the residual 926 over 100 coordinates is
`9.3` per coordinate — three pits away, on average. That is the fingerprint of the mechanism I derived:
`CR=0.9` forces roughly `1 + (D−1)·0.9 ≈ 27` of 30 coordinates (and `≈ 90` of 100) to move together every
trial, which is precisely wrong when the coordinates are independent — the search cannot fix one axis at
a time, it must coordinate a ninety-coordinate jump that almost always lands in a worse pit and is
rejected, so most coordinates never get walked in toward the origin. The convergence column confirms no
rescue was coming from more time: Rosenbrock and Ackley report convergence at generation 500, the wall,
and Rastrigin-100D at 968 of 1000 — still improving when the budget ran out. So the failure is not a bug
in DE; it is the absence of a *local, per-coordinate* perturbation and a base that does not bet the whole
vector on every move. That is what I build now.

Start from what actually hurt the *ancestor* GA, because the scaffold default already hands me a
real-coded GA (SBX + polynomial mutation + tournament) and I need to understand why those operators have
the form they do — and whether that form fixes DE's separability problem. Binary GAs search well on
discrete problems because single-point crossover propagates building blocks, short high-fitness
substrings. When the variables are reals, the standard old move is to binary-encode each variable in a
fixed number of bits and run an ordinary binary GA. Every time I do that I pay: precision is capped by the
bit length, so I can never get arbitrarily close to an optimum between two representable values; the
string→real mapping is fixed and cannot bend; and worst is the Hamming cliff — `0111` and `1000` are
adjacent integers but differ in every bit, so to nudge a variable across that boundary the operators must
flip several bits at once, which they almost never do, and the search jams on the wrong side. On
Rastrigin's grid of pits, where the good move is often a small shift in one coordinate, the Hamming cliff
is fatal. So I want to work directly on the reals — the regime where DE just failed for a *different*
reason (too-coupled moves), so the operator I design must be local in a way DE was not.

Real-coded crossovers exist, and I should walk the tempting ones so the choice is earned. Wright's linear
crossover forms three points from two parents — the midpoint and two reflections — and keeps the best
two; but from any pair it can produce only three candidate children, almost deterministic, so if the good
region is off those three points the operator cannot reach it, far too coarse on a rugged surface. BLX-α
draws the child uniformly from a widened interval `[p1 − α(p2−p1), p2 + α(p2−p1)]`; genuinely stochastic,
and its width scales with the parent span, so it self-anneals as the population converges (the property
DE got from difference vectors). But the density inside that interval is *flat* — a child far out near
the edge is exactly as probable as one hugging a parent — and flat is suspicious, because it is not how
the binary operator I am replacing behaves. Let me pin down how single-point crossover actually
distributes its children on the decoded reals, so I have a target to match rather than a feeling.

Take two parent bitstrings, cut at a random site `k` from the right, swap the tails. Write a decoded
value as `x = B·2^k + A`, `A` the integer from the right `k` bits, `B` from the left. Each child keeps
its own parent's `B` and takes the other's `A`: `y1 = B1·2^k + A2`, `y2 = B2·2^k + A1`. Add them:
`y1 + y2 = (B1+B2)·2^k + (A1+A2)`, identical to `x1 + x2`, so `(y1+y2)/2 = (x1+x2)/2` — the children's
mean equals the parents', and under a linear bit-to-real map this carries to the reals: children sit
symmetrically about the parents' midpoint, equidistant. Single-point crossover does not drift the
per-variable centroid, it spreads around it. How much does it spread? Compare children separation to
parents separation, the spread factor `β = |c1 − c2| / |p1 − p2|`: β<1 contracts, β>1 expands, β=1
stationary. Because it is a *ratio*, the same β gives close children when parents are close and distant
children when far apart — so an operator built around a fixed β-distribution automatically explores while
the population is diverse and narrows as it converges, the same self-annealing DE got from difference
vectors. BLX-α has this width-scaling too; what it lacks is the *shape*.

What shape does single-point crossover produce in β? Compute it on the cleanest case, all-ones crossed
with all-zeros at every site `k`; sweeping `k` traces β(k). To turn that into a density over β, ask how
many sites land in a small window `dβ`: `dk = (1/|dβ/dk|)·dβ`, so the density is proportional to the
reciprocal of the slope. Differentiating gives, for the contracting range `0 ≤ β ≤ 1`, a density that
*increases* toward β=1 — single-point crossover overwhelmingly produces children *close to* their
parents, with a thinning chance of children further out. BLX's flat density gets this wrong. There is a
symmetry to exploit: crossing the two children of a contracting crossover at the same site returns the
parents, so contracting at β corresponds one-to-one with expanding at `1/β`; the mass on each side is one
half, and if the contracting density is `C(β)` on `[0,1]`, the expanding density is forced to
`E(β) = β^{−2}·C(1/β)`. I only choose `C` on `[0,1]`.

Now design `C(β)` to (1) increase toward β=1, (2) be dead simple to sample — this runs on every
coordinate of every mating every generation, on the order of `10^7` coordinate-crossovers per run, so any
special function in the inner loop is a real cost — and (3) carry a concentration knob. The simplest
increasing-toward-1 family is a power law `C(β) = c·β^η` with η ≥ 0. Fix `c` by the half-mass
requirement: `∫₀¹ c·β^η dβ = c/(η+1) = 0.5`, so `c = 0.5(η+1)` and `C(β) = 0.5(η+1)β^η`. Pushing the
symmetry through gives `E(β) = 0.5(η+1)/β^{η+2}` for β>1, whose mass integrates to exactly 0.5 — so the
power law is consistent with the contracting/expanding symmetry, not just convenient. η controls focus:
large η peaks sharply at β=1 (children hug parents, exploitative), small η flattens; η plays the role 1/T
plays in annealing. Sampling is closed-form: the contracting CDF is `0.5·β^{η+1}`, so for `u ≤ 0.5`,
`β = (2u)^{1/(η+1)}`; the expanding branch gives `β = (1/(2(1−u)))^{1/(η+1)}` for `u > 0.5`. Two ops and
one power per coordinate.

I fix a moderate η=20, and I want to know concretely what "children hug parents" means at that value. At
the median `u=0.5`, `β = 1` exactly. At `u=0.25`, `β = 0.5^{1/21} = 0.968`; at `u=0.75`,
`β = 2^{1/21} = 1.034` — across the entire middle half of the draw, the children stay within about 3% of
the parent separation. How often does a coordinate get a genuinely large spread, `|β−1| > 0.1`? On the
contracting side `(2u)^{1/21} < 0.9` needs `2u < 0.9^{21} = 0.109`, so `u < 0.055`; the expanding side
adds another `≈ 0.068`, together about `0.12`. So roughly 88% of coordinate crossovers keep the child
within 10% of the parent separation and only about one in eight reaches further — the exploitative-but-
not-frozen regime, and crucially a *local* operator in the sense DE was not, because a β near 1 changes
each coordinate only slightly, exactly the per-coordinate locality Rastrigin demanded and DE's `CR=0.9`
denied.

Turn β into children with the two requirements already nailed — equidistant from the midpoint, separation
`β·|x1−x2|`: `c1 = 0.5[(1+β)x1 + (1−β)x2]`, `c2 = 0.5[(1−β)x1 + (1+β)x2]`. The sum is `x1+x2` (mean
preserved); the difference is `β(x1−x2)` (spread honored). This reproduces single-point crossover
directly in real space — simulated binary crossover, the scaffold's
`tools.cxSimulatedBinary(ind1, ind2, eta=20.0)`, applied to every coordinate once the outer mating gate
fires.

Crossover alone is not a complete real-coded GA. I need mutation to keep diversity and let the population
escape a basin it collapsed into prematurely — and on Rastrigin, escaping pits is the whole game, the
thing DE's monotone greedy acceptance struggled with. I want mutation with the same character: perturb a
variable to a nearby value, small perturbations more likely than large, never stepping outside
`[lo, hi]`. The same polynomial idea transplants: a perturbation `δ` with density `∝ (1 − |δ|)^{η_m}` on
`[−1,1]`, peaked at 0, inverted by the same trick, then scaled by the range and added,
`x' = x + δ·(hi − lo)`. Large `η_m` makes the nudge small, small `η_m` allows bigger jumps. Size that
step for Rastrigin, the acid test. With `η_m = 20` the typical `|δ|` is on the order of `1/21`, so the
perturbation is about `(hi − lo)/21`, and on Rastrigin's box of width 10.24 that is roughly `0.49`.
Rastrigin's pits are spaced exactly one unit apart with a barrier crest halfway between, so a step of
`0.49` from a coordinate at `x≈1` lands near `x≈0.5`, *on the crest*, worse than it started, and
selection will tend to drop it. So the typical mutation does not itself descend the pit ladder. What
descends it is the *tail*: for a clean hop from pit `m` to pit `m−1` the step must span roughly a full
unit, `|δ|·10.24 ≥ 1`, i.e. `|δ| ≥ 0.098`. On the lower branch that needs `(2u)^{1/21} ≤ 0.902`, so
`u ≤ 0.057`, and by symmetry the upper branch adds another `0.057` — about `0.115`, roughly one mutation
in nine produces a step large enough to carry a coordinate cleanly over the crest into the adjacent lower
pit. Rare per event, but the crucial difference from DE is that it is *isolated*: because `indpb=1/n`
changes one coordinate at a time, a successful hop improves that axis without disturbing the ~29
already-good coordinates, so the recombinant that catches the hop keeps its other gains and wins
tournaments, and selection ratchets the hop into the population. DE's `CR=0.9` move hopped 27 coordinates
simultaneously and had them accepted or rejected as one indivisible bundle, so a single good axis was
always outvoted by the 26 that worsened. The GA descends Rastrigin one ratcheted axis at a time; DE could
not isolate the axis at all. One detail: a bare perturbation can step outside the box near a boundary,
and clipping piles mass on the wall, so I bend the density itself — folding the boundary distances
`δ1 = (x−lo)/(hi−lo)` and `δ2 = (hi−x)/(hi−lo)` into the inversion so that at the lower bound the downward
move vanishes and as `u→0` the branch reaches `x'=lo` exactly, the wall touched but never overshot. This
is the scaffold's `tools.mutPolynomialBounded(individual, eta=20.0, low=lo, up=hi, indpb=1/n)`.

The per-coordinate rate `indpb = 1/n` is the real-coded analogue of the binary `1/L`, and its count is
the whole separability fix. With `indpb = 1/n`, the expected number of coordinates touched in a mutation
pass is `n·(1/n) = 1`. And mutation itself only fires with `mut_prob = 0.2`, so the expected number of
coordinates perturbed per individual per generation is `0.2`. Set that beside DE, which overwrote about
27 coordinates on every trial: the GA changes about one coordinate per five individuals per generation, a
hundred-fold quieter per-coordinate churn — exactly the "change one axis at a time and let the good
values ride along" policy the separable egg-carton rewards, delivered not by design intent but as a side
effect of `indpb=1/n`.

Selection I do not invent anything clever for — I want robustness and no fitness scaling. Tournament
selection: grab `t` individuals at random, keep the best, repeat. No sorting, no normalization, just
comparisons; "best" is whatever the `FitnessMin` weight says, so minimization is handled by the weight.
The tournament size *is* the selection-pressure knob, and I can put a number on `t=3`: with `N`
tournaments each sampling 3 members, the single best is present with probability about `3/N` and wins
whenever present, so its expected number of copies in the mating pool is `N·(3/N) = 3`. Three copies of
the best, not thirty — mild pressure that keeps the population converging without collapsing diversity
too fast. The scaffold default already uses `tools.selTournament(population, k, tournsize=3)`.

So the literal edit here is *exactly the scaffold default fill* — tournament(3) + SBX(η=20) +
polynomial(η=20, indpb=1/n) inside the standard generational loop — which is the point: this establishes
the operator-suite GA the scaffold ships with, the one DE restructured away. The loop: tournament-select
a full parent pool; clone so I do not trample the current population; walk pairs and with probability
`cx_prob=0.9` apply SBX; walk individuals and with probability `mut_prob=0.2` send one through polynomial
mutation; clip everyone into the box; re-evaluate the changed; replace the population generationally.
One budget check, because the comparison to DE only means something if the two spend the same
evaluations: an offspring is re-evaluated if crossed (0.9) or mutated (0.2), a fraction
`1 − (1−0.9)(1−0.2) = 0.92` per generation, about 184 of 200, so roughly 92000 evaluations over 500
generations — the same order as DE's 100000, a fair fight. The full module is in the answer.

Now the predictions against the DE numbers, and where I expect this to also fall short. The per-coordinate
polynomial mutation, sized at half a lattice cell, should rescue Rastrigin from DE's catastrophe:
changing one axis at a time on a separable egg-carton is the right move, so I expect Rastrigin to drop
from DE's 172/926 to single-digit at 30D and roughly an order of magnitude better at 100D — the headline
bet, its mechanism the `0.2`-coordinate-per-generation churn against DE's 27. But I expect two
*regressions* relative to DE, both from the operators being axis-aligned. First, Rosenbrock: SBX and
polynomial mutation have no notion of the diagonal valley — every β spreads children along the coordinate
axes, never along the ribbon — so without DE's anisotropic differences I expect Rosenbrock *worse* than
DE's seed-42 32.2, into the dozens or higher (DE's mean of 48302 is not the honest comparison; its
seed-42 32.2 is, the run where DE's mechanism actually worked, and that mechanism is what I am giving up).
Second, Ackley: the broad smooth basin is where DE's difference vectors shine and axis-aligned SBX does
not, so I expect Ackley better than DE's noisy 10.0 mean but still far from solved — a full unit or so of
error left on the table. So the bet: GA wins decisively on Rastrigin (the separability fix), loses on
Rosenbrock (no anisotropy), roughly ties or modestly improves Ackley. And I expect the convergence
columns to keep telling the budget story — with mild tournament pressure and one coordinate nudged per
five individuals, the GA is a slow refiner and will likely still be improving at the wall, just as DE
was. If that pattern holds, the diagnosis is sharp: what I would want is DE's self-scaling anisotropic
mutation *back*, but carrying the per-coordinate flexibility that fixed Rastrigin, so one operator set is
not forced to choose between the separable egg-carton and the diagonal valley.
