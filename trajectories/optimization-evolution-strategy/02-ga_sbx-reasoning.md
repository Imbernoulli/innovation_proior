The DE numbers came back and they are a clean confirmation of exactly the worry I flagged, plus one
surprise that is worse than I feared. Read them in the order the diagnosis suggested. Ackley at seed 42:
0.488 — not the near-zero I hoped for, but on the *mean* over the three internal seeds it is 10.0, which
means two of the three seeds did not find the basin at all and one did; Ackley's plateau is flat enough
that DE/rand/1 with a fixed `F=0.5` and a slow `rand` base can sit on the plateau, where every difference
vector points across an essentially constant surface and there is no gradient of population fitness to
align with. Rosenbrock at seed 42: 32.2, a respectable result — the anisotropic difference vectors did
track the curved valley as I predicted — but the *mean* is 48302, an enormous number, so again the
aggregate is dominated by seeds where the population scattered and never entered the valley. And Rastrigin
is the catastrophe I called: 172 at 30D (mean 257) and **926** at 100D. That 100D number is the loudest
signal in the table. On a separable egg-carton, `CR=0.9` forces nine of ten coordinates to move together
every trial, which is precisely the wrong move when the coordinates are independent — the search cannot
fix one axis at a time, it must coordinate a 90-coordinate jump that almost always lands in a worse pit.
DE's one global setting, right for Rosenbrock, is catastrophic for Rastrigin, and the budget runs out
(convergence around generation 968 at 100D, i.e. still improving at the wall) before it can grind through.
So the failure is not a bug in DE; it is the absence of a *local, per-coordinate* perturbation and a base
learner that does not bet the whole vector on every move. That is what I will build now.

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

The wish is a crossover that acts on reals directly. Real-coded crossovers exist. Wright's linear
crossover forms three points from two parents — the midpoint and two reflections — and keeps the best two;
but from any pair it can produce only three candidate children, almost deterministic, so if the good region
is off those three points the operator cannot reach it. BLX-α draws the child uniformly from a widened
interval `[p1 − α(p2−p1), p2 + α(p2−p1)]`; genuinely stochastic, and its width scales with the parent span
(so it self-anneals as the population converges, the property DE got from difference vectors). But the
density inside that interval is *flat* — every point equally likely — and I have a nagging feeling flat is
wrong, because it is not how the binary operator I am replacing behaves. Let me pin down how single-point
crossover actually distributes its children on the decoded reals, so I have a target.

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
of every mating every generation — 200 matings × up to 100 coordinates × 500 generations), and (3) carry a
concentration knob. The simplest increasing-toward-1 family is a power law `C(β) = c·β^η` with η ≥ 0 the
distribution index. Fix `c` by the half-mass requirement: `∫₀¹ c·β^η dβ = c/(η+1) = 0.5`, so
`c = 0.5(η+1)` and `C(β) = 0.5(η+1)β^η`. Push through the symmetry: `E(β) = 0.5(η+1)/β^{η+2}` for β>1, and
its mass integrates to exactly 0.5 — so the power law is consistent with the contracting/expanding
symmetry, not just convenient. η controls focus: large η peaks sharply at β=1 (children hug parents,
exploitative), small η flattens; η plays the role 1/T plays in annealing. I fix a moderate value;
η=20 keeps children tightly near good parents, the exploitative-but-not-frozen regime, which is what I want
when I am trying to preserve good substructure rather than blow it apart — and note this is a *local*
operator in the sense DE was not: a β near 1 changes each coordinate only slightly, which is exactly the
per-coordinate locality Rastrigin demanded and DE's `CR=0.9` denied.

Sampling β is closed-form, no special functions, which is what makes it affordable. Contracting CDF:
`∫₀^β 0.5(η+1)t^η dt = 0.5·β^{η+1}`; for a uniform `u ≤ 0.5`, set `0.5·β^{η+1} = u`, so
`β = (2u)^{1/(η+1)}`. Expanding CDF totals `1 − 0.5·β^{−(η+1)}`; for `u > 0.5`,
`β = (1/(2(1−u)))^{1/(η+1)}`. Two ops and one power per coordinate. Turn β into children with the two
requirements already nailed — equidistant from the midpoint, separation `β·|x1−x2|`:
`c1 = 0.5[(1+β)x1 + (1−β)x2]`, `c2 = 0.5[(1−β)x1 + (1+β)x2]`. Sum is `x1+x2` (mean preserved); difference
is `β(x1−x2)` (spread honored). This reproduces single-point crossover directly in real space — simulated
binary crossover. The scaffold's `tools.cxSimulatedBinary(ind1, ind2, eta=20.0)` is exactly this kernel,
applied to every coordinate once the outer mating gate fires.

Crossover alone is not a complete real-coded GA. I need mutation to keep diversity and let the population
escape a basin it collapsed into prematurely — and on Rastrigin, escaping pits is the whole game, the thing
DE's monotone greedy acceptance struggled with. I want mutation with the same character: perturb a variable
to a nearby value, small perturbations more likely than large, never stepping outside `[lo, hi]`. The same
polynomial idea transplants: a perturbation `δ` with density `∝ (1 − |δ|)^{η_m}` on `[−1,1]`, peaked at 0,
inverted by the same trick — for `u < 0.5`, `δ = (2u)^{1/(η_m+1)} − 1`; for `u ≥ 0.5`,
`δ = 1 − (2(1−u))^{1/(η_m+1)}` — then scaled by the range and added, `x' = x + δ·(hi − lo)`. Large `η_m`
makes the nudge small (order `(hi−lo)/η_m`), small `η_m` allows bigger jumps. But a bare perturbation can
step outside the box near a boundary; clipping piles mass on the wall and biases toward it, so I bend the
density itself. With `δ1 = (x−lo)/(hi−lo)` the fraction of range below `x` and `δ2 = (hi−x)/(hi−lo)` above,
fold the boundary distance into the inversion: for `u < 0.5`,
`δ_q = [2u + (1−2u)(1−δ1)^{η_m+1}]^{1/(η_m+1)} − 1`; for `u ≥ 0.5`,
`δ_q = 1 − [2(1−u) + 2(u−0.5)(1−δ2)^{η_m+1}]^{1/(η_m+1)}`. At the lower bound `δ1=0` gives `δ_q=0` (no
downward move possible — right); as `u→0` the downward branch gives `δ_q=−δ1` so `x'=lo` exactly. This is
the scaffold's `tools.mutPolynomialBounded(individual, eta=20.0, low=lo, up=hi, indpb=1/n)`. The
per-coordinate rate `indpb = 1/n` is the real-coded analogue of the binary `1/L`, so on average one
variable changes per pass — enough to keep diversity without tearing good solutions apart, and crucially
*one coordinate at a time*, the local per-axis move Rastrigin rewards and DE could not make.

Selection: I do not invent anything clever, I want robustness and no fitness scaling. Tournament selection
— grab `t` individuals at random, keep the best, repeat to fill the pool. No sorting, no fitness
normalization, just comparisons; "best" is whatever the `FitnessMin` weight says, so minimization is
handled by the weight rather than rewriting the rule. The tournament size *is* the selection-pressure knob:
`t=1` is drift, large `t` is near-greedy, `t=3` is mild pressure that keeps the population converging
without collapsing diversity too fast — a good match for an operator suite whose own job is to balance
exploration and exploitation. The scaffold default already uses `tools.selTournament(population, k,
tournsize=3)`.

So the literal edit here is *exactly the scaffold default fill* — tournament(3) + SBX(η=20) +
polynomial(η=20, indpb=1/n) inside the standard generational loop — which is the point: this rung
establishes the operator-suite GA that the scaffold ships with, the one DE restructured away. The loop:
tournament-select a full parent pool; clone so I do not trample the current population; walk pairs and with
probability `cx_prob=0.9` apply SBX to each pair; walk individuals and with probability `mut_prob=0.2` send
one through polynomial mutation (whose own per-coordinate rate is `1/n`); clip everyone into the box;
re-evaluate the changed; replace the population generationally; record the best. The full scaffold module
is in the answer.

Now the falsifiable predictions against the DE numbers, and where I expect this rung to *also* fall short
so the next one is motivated. The per-coordinate polynomial mutation should rescue Rastrigin from DE's
catastrophe: changing one axis at a time on a separable egg-carton is the right move, so I expect Rastrigin
to drop from DE's 172/926 (30D/100D mean 257/926) to something single-digit at 30D and roughly an
order-of-magnitude better at 100D. That is the headline I am betting on. But I expect two *regressions*
relative to DE. First, Rosenbrock: SBX and polynomial mutation are axis-aligned operators with no notion
of the diagonal valley, so without DE's anisotropic difference vectors I expect Rosenbrock to get *worse*
than DE's seed-42 32.2 — likely into the dozens or higher — because the GA must crawl down a curved valley
with axis-aligned moves. Second, Ackley: the broad smooth basin is exactly where DE's difference vectors
shine and axis-aligned SBX does not, so I expect Ackley to land around 1 rather than near machine
precision — better than DE's noisy 10.0 mean but far from solved. So the concrete bet is: GA *wins
decisively on Rastrigin* (the separability fix), *loses on Rosenbrock* (no anisotropy), and *roughly ties
or modestly improves Ackley*. If that pattern holds, the lesson for the next rung is sharp — I want DE's
self-scaling anisotropic mutation *back*, but with the per-coordinate flexibility and the local refinement
that fixed Rastrigin, and with the global `F`/`CR` setting that doomed DE made *adaptive* so one algorithm
can be right on all four landscapes at once.
