The multi-start rung climbed to the low `2.62`s and stalled exactly as predicted: best-of-`120`
lifted the sum `+0.0272` to `2.6221`, the gains arrived with the record signature — sporadic new
bests thinning out — and the distance to `2.636` only shrank to `~0.014`. SLSQP is still the right
local engine; what saturated is the *sampling*. Best-of-`K` is a maximum order statistic over a fixed
distribution `F` of uniform-start optima, and its climb grows like `√(2 ln K)` — logarithmic, knee
already passed. So random restart has two structural weaknesses no `K` repairs: the starts have no
*structure* (a uniform scatter looks nothing like a good packing, so most refinements waste themselves
in mediocre basins) and the search has no *memory* (every refined packing is discarded). This rung
fixes both, built from those two holes.

Cranking `K` further is the wrong lever, and that forces a genuinely different design. Reaching the
frontier basins by restart costs `K ~ 1/q` while the payoff is the fixed `~0.014`, so the cost
explodes as the good tail thins. There are two ways out, and they are different operations. Change `F`
itself — start from configurations that already look like good packings, so the whole distribution of
refined optima shifts rightward; that attacks "no structure." Or stop drawing i.i.d. samples and do
*directed* local search — take the best packing found and perturb-and-re-refine it, walking from a good
basin to a nearby better one; that attacks "no memory" and is not sampling any distribution. I want
both: structure gets me into a strong basin quickly, and directed search mines that basin for the last
coordinated moves restart can never reach.

Start with structure in the initialization. I want starts that resemble a good sum-of-radii packing — a
few disks pushed hard into the corners and along the edges, where two free walls let a circle grow
toward the `0.5` ceiling, and the interior filled by a smooth quasi-uniform spread that does not clump,
because two close initial centres force both radii small. The construction that delivers this is the
golden-angle spiral. For uniform areal density in a disk the count inside radius `ρ` grows like `ρ²`,
so `i ∝ ρ_i²` and `ρ_i ∝ √i`; normalising the outermost point to fill radius `0.5` gives
`ρ_i = 0.5√(i/m)` for the `m` interior points. For the angle I want successive points as far from
resonance as possible so no two fall on the same spoke; the most-irrational rotation number is the
golden ratio, whose continued fraction is all `1`s, so the increment is `2π(2 − φ) = π(3 − √5) ≈ 137.5°`.
That is the golden angle — the sunflower-seed arrangement, no radial gaps or spokes — giving the
interior the non-clumping quasi-uniform spread I want.

The spiral specifically, rather than the other even spreads, because each alternative fails a test it
passes. A square or hexagonal lattice spreads perfectly evenly but is exactly the rigidity the grid
baseline showed to be a dead end — it would bias SLSQP back toward the symmetric near-grid basins I am
trying to escape. Concentric rings impose a rotational resonance whose interstitial gaps line up into
channels a good packing must break. A low-discrepancy sequence (Halton, Sobol) fills evenly and
aperiodically but is tuned for uniform coverage of a rectangle, with no radial bias toward the
corner-heavy layout a square packing wants. The golden-angle spiral is simultaneously even (equal-area
radii), aperiodic (the most-irrational angle kills every spoke and ring), and radial (it leaves the four
corners for me to seed by hand) — the combination the others each miss.

The spiral does not seed the corners, so I add them by hand: its outermost points only graze the
mid-edges and reach the corners barely at all, yet the corners are the prize real estate. So I plant
four circles near the corners at `(0.07, 0.07)` and its three mirror images, lay the remaining `22` on
the spiral, and apply a small random jitter so repeated structured starts are a family rather than one
point. I can put a number on the shift this buys. The `22` spiral points fill a disk of radius `0.5` at
density `22/(π·0.25) ≈ 28`, so their spacing is about `1/√28 ≈ 0.189` — right at the grid's tangency
scale, almost perfectly even, so essentially zero pairs start closer than tangency distance `0.1`. The
uniform scatter of the previous rung carried about a dozen such pairs and its clumps dragged its
LP-radii start down near `1.2`, below the grid. The spiral has no clumps to un-knot, so its LP-radii
start already sits up near the grid's `2.5` before SLSQP takes a step, with the four corner seeds free
to grow on two walls. The two kinds of start hand SLSQP configurations more than a full unit of
radius-sum apart — which is what "shifting `F` rightward" means concretely.

The local engine stays the joint centre-and-radius SLSQP, unchanged, and it stays joint for the reason
the single-start rung traced on the grid corner: growing a corner disk requires a coordinated shove of
a whole chain, and only a solver carrying centres and radii in one quadratic model can price that trade
in a single step. A split scheme structurally cannot make those moves, and the whole climb above the
grid lives in exactly them. Radii are re-tightened to their exact LP optimum after each refinement.

Now the memory, which is the real escape: iterated perturbation chains. Rather than discard refined
packings, I take the best so far, perturb it, and re-refine — repeating to form a chain that walks from
good basin to better basin. The perturbation jiggles a random subset of centres by a Gaussian kick,
then lets SLSQP repair and re-optimise; if the result is feasible and better it becomes the new
incumbent, else I try again from the same incumbent. Two design choices inside the kick both come from
what the chain needs to do. First, how many circles to perturb: I draw the subset size `k` uniformly
from `1` to `⌊N/2⌋ = 13`. A jammed disk touches four-to-six neighbours, and a kick of `σ = 0.06` — more
than half a typical radius — displaces a circle far enough to break essentially all its contacts. So
`k = 1` frees roughly five contacts, a surgical repair SLSQP re-seats without disturbing the rest of the
graph; `k = 13` frees on the order of `65` contacts, well past the `~50` that define the entire rigid
web, so the packing is effectively re-melted and re-jammed into a possibly-different topology. Drawing
`k` uniformly over `1..13` samples the full range from surgical re-seating to wholesale re-melting, and
the last `~0.014` of sum is won by a mix of both: mostly small rebalancings inside a fixed contact
graph, punctuated by the rare re-melt that finds a strictly better graph.

Second, how big a kick: an adaptive schedule. On an improvement I reset the kick to its full `σ = 0.06`
and explore boldly from the new incumbent; on a failure I cool it, `σ ← 0.94 σ`, down to a floor
`σ = 0.012`, searching finer around a configuration I cannot immediately beat. That reset-on-success,
shrink-on-failure rule is the classic exploration/exploitation adaptation, the same logic as the
`1/5`-success rule. The floor matters: below about `0.012` a kick is smaller than the scale at which a
circle can change which neighbours it touches, so it just jiggles within the same basin and wastes the
SLSQP repair. The timescale matches the chain length by construction: cooling by `0.94` per failure
reaches the `0.012` floor after `ln 0.2 / ln 0.94 ≈ 26` consecutive failures, so a chain of about `25`
steps that never improves sweeps the kick across its entire useful range exactly once, and
reset-on-success re-arms that sweep every time the chain finds a better basin. The kick is also not
optional — at `σ = 0` each step runs SLSQP from the incumbent's own centres, which, being a KKT point,
it returns unchanged, so the chain spins in place forever. The perturbation is the entire engine; the
re-SLSQP only repairs what the kick disturbs.

Why perturb-from-incumbent rather than restart is the crux, and it is mechanism, not preference. A
jammed packing is a rigid contact network of roughly `2N ≈ 52` contacts, and moving to a *better*
network cannot be done by sliding continuously — the two networks are different combinatorial objects,
separated by a barrier where at least one contact must break and another form. A pure local solver at a
jammed point cannot cross it: every improving direction is blocked by an active constraint. The Gaussian
kick is precisely the barrier-crossing move — it displaces a subset of centres far enough to break some
contacts, momentarily un-jamming the packing, and the re-SLSQP re-jams it into whichever network is now
nearest, which with some probability is better. That is why perturbing more circles enables a *topology*
change while perturbing one only re-optimises within the current network, and why the kick floors at the
sub-contact scale. The coordinated lateral moves that climb the last `~0.014` — long sequences of
mostly-sideways rearrangements each unlocking a sliver of radius — are reachable from a good
configuration by a small perturbation but astronomically unlikely from a cold scatter, because they
require already being *near* the good graph. The structured restarts find a strong basin; the chains
mine it. So there is a falsifiable prediction: where restart records are i.i.d. and thin out like `1/k`,
a chain that keeps improving can produce runs of correlated gains — one good perturbation opens several
more — so the perturbation phase should show clustered improvements rather than isolated thinning ones.

I run this as three phases under a fixed wall-clock budget of about `520` seconds, split `0.4 / 0.6 / 1.0`.
The first `~208` seconds are structured spiral-plus-corner restarts with joint SLSQP, to find a strong
basin quickly — at a second or two per refinement, on the order of a hundred-plus starts. The next `~104`
seconds mix in pure random restarts as cheap insurance against the opinionated structured prior having
boxed the search into one region. The final `~208` seconds are the perturbation chains from the
incumbent, each `~25` steps, grinding the best packing upward. Throughout I keep the single best feasible
packing, verify it against the real constraints, and re-tighten its radii by LP. The seed is fixed at `7`
so the whole run is one reproducible object.

A design-space check before I commit, since "perturbation chain" is not the only memory mechanism.
Simulated annealing on the `78`-variable configuration needs every proposal projected back to
feasibility, and that projection is itself the SLSQP solve I am already doing, so SA would just wrap the
same solver in a more fragile acceptance rule. A genetic method crossing two good packings produces a
child splicing half of one with half of another, almost always overlapping and infeasible, needing so
much repair the parents' structure is destroyed — crossover has no natural meaning for jammed disks.
Iterated local search with SLSQP as both repair and local optimiser is the clean choice precisely because
the one operation I trust — joint SLSQP to a KKT point — *is* the repair, so the perturbation only
proposes a nearby configuration.

What do I expect, and where do I stop honestly? Two checks against the phase trace. The structured init
alone should already clear the random-multi-start plateau of `2.6221` — the rightward shift of `F` means
even the structured-restart phase, before any chains, should land above where `120` uniform starts
saturated. Then the chains should add a further increment on top, mining the incumbent past what any
single refinement reaches, with the trace shaped structured-phase-plateaus-fast then
chains-grind-slowly-upward. But I am running one constructor under a bounded budget — minutes, not the
orders-of-magnitude more autonomous compute that produced the record `2.635988438568` — and the
published frontier is a band separated by only parts in the sixth decimal, each part costing a very long
chain of mostly-lateral perturb-and-re-SLSQP moves through near-degenerate basins. A nine-minute run
cannot make that many moves. So I expect to land in the frontier *neighbourhood* but below the record —
the honest outcome of running the right pipeline with a fraction of the compute.

I can predict the shape of the packing this returns, as a structural check. It should be irregular — the
grid analysis proved any substantial gain over `2.5414` cannot be structured — with a few large disks
pressed into the corners and along the edges and the interior filled by many unequal gap-fillers of
graded size. Every circle pinned: at a local maximum nothing can grow, so each radius is held by at least
one active wall or contact, and the active contacts should form a roughly isostatic web of order
`2N ≈ 50` constraints out of `429`. If instead the returned configuration had a slack circle floating
free, or looked grid-like, I would distrust the run.

So the pipeline is the endpoint method: structured init supplies the structure random restart lacked,
iterated chains supply the memory, joint SLSQP remains the engine underneath both. Run under a bounded
budget it should reach the frontier neighbourhood, and the residual gap from my number to
`2.635988438568` will be the honest measure of the part of this problem bought purely with search budget,
not a better algorithm — which names what the final rung must confront: not a cleverer local move, since
none closes a sixth-decimal gap within a bounded run, but the specific configuration that only a much
longer search reaches.
