The DE numbers came back and they are a clean confirmation of exactly the worry I flagged, plus one
surprise that is worse than I feared. Read them in the order the diagnosis suggested, and read the *mean*
against the *seed-42* value, because the gap between them is itself a measurement. Ackley at seed 42:
0.488 — not the near-zero I hoped for, but close. On the mean over the three internal seeds it is 10.0,
and that jump is not noise, it is structure: if one seed is 0.488 and the mean is 10.008, the other two
seeds average `(3·10.008 − 0.488)/2 ≈ 14.76`. Ackley's plateau sits around 20 (the exponential basin term
vanishes far from the origin and only the constant offset survives), so a value near 14.8 is a search that
came partway off the plateau and stalled. Two seeds partway down, one seed at the bottom — that is a
bimodal outcome, a basin-found-or-not coin flip, exactly what a slow `rand` base does on a flat plateau
where every difference vector points across an essentially constant surface and there is no gradient of
population fitness to align with. Rosenbrock tells the same bimodal story louder: seed 42 is 32.2, a
respectable result — the anisotropic difference vectors did track the curved valley as I predicted — but
the mean is 48302, which forces the other two seeds to average `(3·48302 − 32.2)/2 ≈ 72437` each. A
Rosenbrock value in the tens of thousands is a population that never entered the valley at all and is still
scattered across the box. So DE's real weakness on the smooth problems is variance, not the median case:
when it finds the structure it exploits it beautifully, and when the `rand` base fails to seed the basin it
scatters and the mean explodes.

And Rastrigin is the catastrophe I called, with no bimodality to hide behind — it is uniformly bad. 172 at
30D (mean 257) and **926** at 100D, where the single seed equals the mean, so this is not a lucky-seed
story, it is the method's honest level. Turn those residuals into geometry. On Rastrigin each coordinate
sitting at the integer pit `m` contributes `m²` to the cost, and a coordinate solved to the origin
contributes 0. So the seed-42 30D residual of 172 spread over 30 coordinates is an average `m²` of about
`5.7`, i.e. the typical coordinate is stranded two to three lattice steps out from the origin; the mean 257
pushes that to `8.6` per coordinate. At 100D the residual 926 over 100 coordinates is `9.3` per
coordinate — the coordinates are, on average, three pits away from where they need to be. That is the
fingerprint of the mechanism I derived last rung: `CR=0.9` forces roughly `1 + (D−1)·0.9 ≈ 27` of 30
coordinates (and `≈ 90` of 100) to move together every trial, which is precisely the wrong move when the
coordinates are independent — the search cannot fix one axis at a time, it must coordinate a
ninety-coordinate jump that almost always lands in a worse pit and is rejected, so most coordinates never
get walked in toward the origin. The convergence column confirms there was no rescue coming from more time:
Rosenbrock and Ackley at seed 42 report convergence at generation 500, the wall, and Rastrigin-100D at 968
of 1000 — the search was still improving when the budget ran out. So the failure is not a bug in DE; it is
the absence of a *local, per-coordinate* perturbation and a base learner that does not bet the whole vector
on every move. That is what I will build now.

Let me start the GA from what actually hurt the *ancestor* GA, because the scaffold default already hands
me a real-coded GA (SBX + polynomial mutation + tournament) and I need to understand why those operators
have the form they do — and whether that form fixes DE's separability problem. Binary GAs search well on
discrete problems because single-point crossover propagates building blocks, short high-fitness
substrings. When my variables are reals, the standard old move is to binary-encode each variable in a
fixed number of bits and run an ordinary binary GA. Every time I do that I pay: precision is capped by the
bit length so I can never get arbitrarily close to an optimum between two representable values; the
string→real mapping is fixed and the algorithm cannot bend it; and worst is the Hamming cliff — `0111` and
`1000` are adjacent integers but differ in every bit, so to nudge a variable across that boundary the
operators must flip several bits at once, which they almost never do, and the search jams on the wrong side.
On Rastrigin's grid of pits, where the good move is often a small shift in one coordinate, the Hamming
cliff is fatal. So I want to work directly on the reals — and that is exactly the regime where DE just
failed for a *different* reason (too-coupled moves), so the operator I design must be local in a way DE was
not.

The wish is a crossover that acts on reals directly. Real-coded crossovers exist, and I should walk the
tempting ones before committing, so the choice is earned. Wright's linear crossover forms three points from
two parents — the midpoint and two reflections — and keeps the best two; but from any pair it can produce
only three candidate children, almost deterministic, so if the good region is off those three points the
operator cannot reach it, and on a rugged surface three fixed candidates per mating is far too coarse.
BLX-α draws the child uniformly from a widened interval `[p1 − α(p2−p1), p2 + α(p2−p1)]`; genuinely
stochastic, and its width scales with the parent span (so it self-anneals as the population converges, the
property DE got from difference vectors). But the density inside that interval is *flat* — every point
equally likely, so a child far out near the interval edge is exactly as probable as one hugging a parent —
and I have a nagging feeling flat is wrong, because it is not how the binary operator I am replacing
behaves. Let me pin down how single-point crossover actually distributes its children on the decoded
reals, so I have a target to match rather than a feeling.

Take two parent bitstrings, cut at a random site `k` from the right, swap the tails. Write a decoded value
as `x = B·2^k + A`, `A` the integer from the right `k` bits, `B` from the left. Each child keeps its own
parent's `B` and takes the other's `A`: `y1 = B1·2^k + A2`, `y2 = B2·2^k + A1`. Add them:
`y1 + y2 = (B1+B2)·2^k + (A1+A2)`, identical to `x1 + x2`. So `(y1+y2)/2 = (x1+x2)/2` — the children's
mean equals the parents' mean, and under a linear bit-to-real map this carries to the reals: children sit
symmetrically about the parents' midpoint, equidistant. That is a structural property: single-point
crossover does not drift the per-variable centroid, it spreads around it. How much does it spread?
Compare children separation to parents separation: the spread factor `β = |c1 − c2| / |p1 − p2|`. β<1
contracts (children enclosed), β>1 expands, β=1 is stationary. Because it is a *ratio*, the same β gives
close children when parents are close and distant children when far apart — so an operator built around a
fixed β-distribution automatically explores while the population is diverse and narrows as it converges,
the same self-annealing DE got from difference vectors. BLX-α has this width-scaling too; what it lacks is
the *shape*.

What shape does single-point crossover produce in β? Compute it on the cleanest case, all-ones crossed
with all-zeros at every site `k`; sweeping `k` traces β(k). To turn that into a density over β, ask how
many sites land in a small window `dβ`: `dk = (1/|dβ/dk|)·dβ`, so the density is proportional to the
reciprocal of the slope. Differentiating gives, for the contracting range `0 ≤ β ≤ 1`, a density that
*increases* toward β=1 — single-point crossover overwhelmingly produces children *close to* their parents,
with a thinning chance of children further out. BLX's flat density gets this wrong; it makes a far-out
child as likely as a near one. There is a symmetry to exploit: crossing the two children of a contracting
crossover at the same site returns the parents, so contracting at β corresponds one-to-one with expanding
at `1/β`; the mass on each side is one half, and if the contracting density is `C(β)` on `[0,1]`, the
expanding density is forced to `E(β) = β^{−2}·C(1/β)`. I only choose `C` on `[0,1]`.

Now design `C(β)` to (1) increase toward β=1, (2) be dead simple to sample (this runs on every coordinate
of every mating every generation — 200 matings × up to 100 coordinates × 500 generations, on the order of
`10^7` coordinate-crossovers per run, so any special function in the inner loop is a real cost), and (3)
carry a concentration knob. The simplest increasing-toward-1 family is a power law `C(β) = c·β^η` with η ≥
0 the distribution index. Fix `c` by the half-mass requirement: `∫₀¹ c·β^η dβ = c/(η+1) = 0.5`, so
`c = 0.5(η+1)` and `C(β) = 0.5(η+1)β^η`. Push through the symmetry: `E(β) = 0.5(η+1)/β^{η+2}` for β>1, and
its mass integrates to exactly 0.5 — so the power law is consistent with the contracting/expanding
symmetry, not just convenient. η controls focus: large η peaks sharply at β=1 (children hug parents,
exploitative), small η flattens; η plays the role 1/T plays in annealing. I fix a moderate value, η=20, and
I want to check that this number actually means "children hug parents" rather than assert it.

Sampling β is closed-form, no special functions, which is what makes it affordable. Contracting CDF:
`∫₀^β 0.5(η+1)t^η dt = 0.5·β^{η+1}`; for a uniform `u ≤ 0.5`, set `0.5·β^{η+1} = u`, so
`β = (2u)^{1/(η+1)}`. Expanding CDF totals `1 − 0.5·β^{−(η+1)}`; for `u > 0.5`,
`β = (1/(2(1−u)))^{1/(η+1)}`. Two ops and one power per coordinate. Now put η=20 through it numerically. At
the median `u=0.5`, `β = 1` exactly — children land right at the parent separation. At `u=0.25`,
`β = 0.5^{1/21} = 0.968`; at `u=0.75`, `β = 2^{1/21} = 1.034`. So across the entire middle half of the
random draw, the children stay within about 3% of the parent separation. How often does a coordinate get a
genuinely large spread, say `|β−1| > 0.1`? On the contracting side I need `(2u)^{1/21} < 0.9`, i.e.
`2u < 0.9^{21} = 0.109`, so `u < 0.055`; on the expanding side `1/(2(1−u)) > 1.1^{21} = 7.40`, so
`u > 0.932`. Together that is about `0.055 + 0.068 ≈ 0.12`. So roughly 88% of coordinate crossovers keep the
child within 10% of the parent separation, and only about one in eight reaches further. That is what η=20
buys concretely: an operator that overwhelmingly preserves the parents' structure and only occasionally
probes outward — the exploitative-but-not-frozen regime, and crucially a *local* operator in the sense DE
was not, because a β near 1 changes each coordinate only slightly, which is exactly the per-coordinate
locality Rastrigin demanded and DE's `CR=0.9` denied.

Turn β into children with the two requirements already nailed — equidistant from the midpoint, separation
`β·|x1−x2|`: `c1 = 0.5[(1+β)x1 + (1−β)x2]`, `c2 = 0.5[(1−β)x1 + (1+β)x2]`. Sum is `x1+x2` (mean preserved); difference
is `β(x1−x2)` (spread honored). Let me verify on numbers with the `β = 0.968` that `u=0.25` produced:
parents `x1 = 2.0`, `x2 = 3.0` give `c1 = 0.5[1.968·2 + 0.032·3] = 2.016` and `c2 = 0.5[0.032·2 + 1.968·3]
= 2.984`. Their sum is `5.0 = x1 + x2`, the midpoint held at `2.5`; their separation is `0.968 = β·1`, the
parents' gap contracted by exactly β. The children `(2.016, 2.984)` sit just inside the parents `(2, 3)` —
that is the "hug" the 88%-within-10% figure predicted, made concrete. This reproduces single-point
crossover directly in real space — simulated binary crossover. The scaffold's `tools.cxSimulatedBinary(ind1, ind2, eta=20.0)` is exactly this
kernel, applied to every coordinate once the outer mating gate fires.

Crossover alone is not a complete real-coded GA. I need mutation to keep diversity and let the population
escape a basin it collapsed into prematurely — and on Rastrigin, escaping pits is the whole game, the thing
DE's monotone greedy acceptance struggled with. I want mutation with the same character: perturb a variable
to a nearby value, small perturbations more likely than large, never stepping outside `[lo, hi]`. The same
polynomial idea transplants: a perturbation `δ` with density `∝ (1 − |δ|)^{η_m}` on `[−1,1]`, peaked at 0,
inverted by the same trick — for `u < 0.5`, `δ = (2u)^{1/(η_m+1)} − 1`; for `u ≥ 0.5`,
`δ = 1 − (2(1−u))^{1/(η_m+1)}` — then scaled by the range and added, `x' = x + δ·(hi − lo)`. Large `η_m`
makes the nudge small, small `η_m` allows bigger jumps. Let me size that step for Rastrigin specifically,
because it is the acid test. With `η_m = 20` the typical `|δ|` is on the order of `1/(η_m+1) = 1/21`, so
the perturbation is about `(hi − lo)/21`, and on Rastrigin's box of width 10.24 that is roughly `0.49`.
Rastrigin's pits are spaced exactly one unit apart with a barrier crest halfway between, so I have to be
careful here rather than wave: a step of `0.49` from a coordinate sitting at `x≈1` lands near `x≈0.5`,
which is *on the barrier crest*, worse than where it started, and a per-individual selection will tend to
drop it. So the typical mutation does not itself descend the pit ladder. What descends it is the *tail* of
the same distribution. For a clean hop from pit `m` to pit `m−1` the step must span roughly a full unit,
`|δ|·10.24 ≥ 1`, i.e. `|δ| ≥ 0.098`. Sizing that tail: on the lower branch `|δ| ≥ 0.098` needs
`(2u)^{1/21} ≤ 0.902`, so `2u ≤ 0.902^{21} = 0.115`, `u ≤ 0.057`, and by symmetry the upper branch adds
another `0.057`, so about `0.115` — roughly one mutation in nine produces a step large enough to carry a
coordinate cleanly over the crest into the adjacent, lower pit. That is rare per event, but the crucial
difference from DE is that it is *isolated*: because `indpb=1/n` changes one coordinate at a time, a
successful hop improves that axis without disturbing the ~29 already-good coordinates, so the recombinant
that catches the hop keeps its other gains and wins tournaments, and selection ratchets the successful hop
into the population. DE's `CR=0.9` move, by contrast, hopped 27 coordinates simultaneously and had them
accepted or rejected as one indivisible bundle, so a single good axis was always outvoted by the 26 that
worsened. The GA descends Rastrigin one ratcheted axis at a time; DE could not isolate the axis at all.
But a bare perturbation can
step outside the box near a boundary; clipping piles mass on the wall and biases toward it, so I bend the
density itself. With `δ1 = (x−lo)/(hi−lo)` the fraction of range below `x` and `δ2 = (hi−x)/(hi−lo)` above,
fold the boundary distance into the inversion: for `u < 0.5`,
`δ_q = [2u + (1−2u)(1−δ1)^{η_m+1}]^{1/(η_m+1)} − 1`; for `u ≥ 0.5`,
`δ_q = 1 − [2(1−u) + 2(u−0.5)(1−δ2)^{η_m+1}]^{1/(η_m+1)}`. Check the boundary: at the lower bound `δ1=0`
gives `δ_q=0` (no downward move possible — right); as `u→0` the downward branch gives `δ_q=−δ1` so `x'=lo`
exactly, the wall reached but never overshot. This is the scaffold's
`tools.mutPolynomialBounded(individual, eta=20.0, low=lo, up=hi, indpb=1/n)`.

The per-coordinate rate `indpb = 1/n` is the real-coded analogue of the binary `1/L`, and it is worth
counting what it actually does, because that count is the whole separability fix. With `indpb = 1/n`, the
expected number of coordinates touched in a mutation pass is `n·(1/n) = 1` — one variable changes per pass,
on average. And mutation itself only fires with `mut_prob = 0.2`, so the expected number of coordinates an
individual has perturbed per generation is `0.2·1 = 0.2`. Set that beside DE, which overwrote about 27
coordinates on every trial: the GA changes on the order of one coordinate per five individuals per
generation, a hundred-fold quieter per-coordinate churn. That is exactly the "change one axis at a time and
let the good values of the others ride along" policy the separable egg-carton rewards and DE's coupled
`CR=0.9` denied — the operator suite delivers it not by design intent but as a side effect of `indpb=1/n`.

Selection: I do not invent anything clever, I want robustness and no fitness scaling. Tournament selection
— grab `t` individuals at random, keep the best, repeat to fill the pool. No sorting, no fitness
normalization, just comparisons; "best" is whatever the `FitnessMin` weight says, so minimization is
handled by the weight rather than rewriting the rule. The tournament size *is* the selection-pressure knob,
and I can put a number on what `t=3` costs in greediness: with `N` tournaments each sampling 3 members, the
single best individual is present in a given tournament with probability about `3/N` and wins whenever
present, so its expected number of copies in the mating pool is `N·(3/N) = 3`. Three copies of the best,
not thirty — mild pressure that keeps the population converging without collapsing diversity too fast, a
good match for an operator suite whose own job is to balance exploration and exploitation. The scaffold
default already uses `tools.selTournament(population, k, tournsize=3)`.

So the literal edit here is *exactly the scaffold default fill* — tournament(3) + SBX(η=20) +
polynomial(η=20, indpb=1/n) inside the standard generational loop — which is the point: this rung
establishes the operator-suite GA that the scaffold ships with, the one DE restructured away. The loop:
tournament-select a full parent pool; clone so I do not trample the current population; walk pairs and with
probability `cx_prob=0.9` apply SBX to each pair; walk individuals and with probability `mut_prob=0.2` send
one through polynomial mutation (whose own per-coordinate rate is `1/n`); clip everyone into the box;
re-evaluate the changed; replace the population generationally; record the best. One budget check before I
commit, because the comparison to DE only means something if the two spend the same evaluations: an
offspring is re-evaluated if it was in a crossed pair (probability 0.9) or mutated (probability 0.2), so the
fraction re-evaluated per generation is `1 − (1−0.9)(1−0.2) = 0.92`, about 184 of 200, i.e. roughly 92000
evaluations over 500 generations — the same order as DE's 100000, so this is a fair fight on budget. The
full scaffold module is in the answer.

Now the falsifiable predictions against the DE numbers, and where I expect this rung to *also* fall short
so the next one is motivated. The per-coordinate polynomial mutation, sized at half a lattice cell, should
rescue Rastrigin from DE's catastrophe: changing one axis at a time on a separable egg-carton is the right
move, so I expect Rastrigin to drop from DE's 172/926 (30D seed / 100D) to something single-digit at 30D
and roughly an order-of-magnitude better at 100D — that is the headline I am betting on, and its mechanism
is the `0.2`-coordinate-per-generation churn against DE's 27. But I expect two *regressions* relative to DE,
and both follow from the operators being axis-aligned. First, Rosenbrock: SBX and polynomial mutation have
no notion of the diagonal valley — every β spreads children along the coordinate axes, never along the
ribbon — so without DE's anisotropic difference vectors I expect Rosenbrock to get *worse* than DE's
seed-42 32.2, likely into the dozens or higher, because the GA must crawl down a curved valley with moves
that are never aligned to it. (DE's mean of 48302 is not the honest comparison there; its seed-42 32.2 is,
because that is the run where DE's mechanism actually worked, and it is that mechanism I am giving up.)
Second, Ackley: the broad smooth basin is exactly where DE's difference vectors shine and axis-aligned SBX
does not, so I expect Ackley to land around 1 rather than near machine precision — better than DE's noisy
10.0 mean but far from solved. So the concrete bet is: GA *wins decisively on Rastrigin* (the separability
fix), *loses on Rosenbrock* (no anisotropy), and *roughly ties or modestly improves Ackley*. And I expect
the convergence columns to keep telling the budget story — with mild tournament pressure and a single
coordinate nudged per five individuals, the GA is a slow refiner and will likely still be improving at the
wall, just as DE was. If that pattern holds, the diagnosis is
sharp — what I would want is DE's self-scaling anisotropic mutation *back*, but carrying the per-coordinate
flexibility and the local refinement that fixed Rastrigin, so that one operator set is not forced to choose
between the separable egg-carton and the diagonal valley.
