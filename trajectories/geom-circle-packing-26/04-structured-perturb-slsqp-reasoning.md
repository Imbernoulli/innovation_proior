The multi-start rung climbed to the low `2.62`s and then stalled in exactly the way I predicted:
best-of-`120` lifted the sum `+0.0272` to `2.6221`, but the gains arrived with the record signature —
sporadic new bests, thinning out as the run went — and the distance to the `2.636` frontier only
shrank to `~0.014` rather than closing. SLSQP is still the right local engine; every rung has leaned on
it and it keeps returning clean local optima. What saturated is the *sampling*: best-of-`K` is a
maximum order statistic over a fixed distribution `F` of uniform-start optima, and I argued last time
that its climb grows like `√(2 ln K)` — logarithmic, knee already passed at a hundred-odd starts. So
the honest reading of the plateau is that random restart has two structural weaknesses that no amount
of `K` repairs: the starts have no *structure* (a uniform scatter looks nothing like a good packing,
so most refinements waste themselves in mediocre basins) and the search has no *memory* (every refined
packing is discarded, so the best configuration found is never used to seed anything better). This rung
has to fix both, and I will build the fix from those two named holes rather than from a menu of tricks.

Before I build it, let me be precise about why cranking `K` further is the wrong lever, because that is
what forces a genuinely different design. Last rung I priced the frontier basins as carrying some
probability mass `q` under uniform scattering; reaching them by restart costs `K ~ 1/q` while the
payoff is the fixed `~0.014`, so the cost explodes as the good tail thins and the return shrinks like
`√(ln K)`. There are only two ways out of that trap, and they are genuinely different operations.
One is to change `F` itself — start from configurations that already look like good packings, so the
whole distribution of refined optima shifts rightward and even a typical start lands well; that
attacks the "no structure" hole. The other is to stop drawing i.i.d. samples from any `F` at all and
instead do *directed* local search — take the best packing found and perturb-and-re-refine it, walking
from a good basin to a nearby better one; that attacks the "no memory" hole and is not sampling a
distribution in any sense. I want both, because they fix different things: structure gets me into a
strong basin quickly, and directed search mines that basin for the last coordinated moves that restart
can never reach.

Start with structure in the initialization. I want starts that already resemble a good sum-of-radii
packing, and from the earlier rungs I know what that looks like: a few disks pushed hard into the
corners and along the edges, where two free walls let a circle grow toward the `0.5` ceiling I derived
on the single-start rung, and the interior filled by a smooth, quasi-uniform spread that does not
clump. Clumping is the enemy of a good LP-radii start, because two close initial centres force both
their radii small, so I want an interior spread with roughly equal local spacing and no long-range
regularity that SLSQP would just have to break. The construction that delivers exactly that is the
golden-angle spiral, and it is worth deriving rather than invoking. If I place point `i` at polar
radius `ρ_i` and want uniform areal density in the disk, the count of points inside radius `ρ` must
grow like `ρ²`, so `i ∝ ρ_i²` and `ρ_i ∝ √i`; normalising the outermost point to fill radius `0.5`
gives `ρ_i = 0.5 √(i/m)` for the `m` interior points. For the angle I want successive points to land
as far from resonance as possible, so that no two ever fall on the same spoke and the local
neighbourhood of every point looks the same. The most-irrational rotation number is the golden ratio,
whose continued fraction is all `1`s and is therefore the hardest to approximate by rationals, so the
angular increment is `2π(1 − 1/φ) = 2π(2 − φ) = π(3 − √5) ≈ 137.5°`. That is the golden angle, the same
construction that arranges sunflower seeds without radial gaps or spokes, and it gives the interior the
non-clumping quasi-uniform spread I want.

I should say why the spiral specifically, rather than the other obvious ways to spread points evenly,
because the choice is not arbitrary and each alternative fails a test the spiral passes. A regular
square or hexagonal lattice spreads points perfectly evenly, but that is exactly the rigidity the grid
baseline already showed to be a dead end — a lattice start biases SLSQP back toward the symmetric
near-grid basins whose ceiling I measured at `88.6%` of the area bound, so I would be seeding the search
inside the very region I am trying to escape. Concentric rings spread evenly too, but they impose a
rotational resonance — points fall on shared radii and shared spokes — so their interstitial gaps line
up into channels that a good packing has to break, wasting the first refinement iterations undoing the
structure. A low-discrepancy sequence like Halton or Sobol gives an even, aperiodic fill and is a
reasonable alternative, but it is tuned for uniform coverage of a rectangle, with no radial bias toward
the corner-heavy layout a square packing wants. The golden-angle spiral is the one construction that is
simultaneously even (equal-area radii), aperiodic (the most-irrational angle kills every spoke and
ring), and radial (it naturally leaves the four corners for me to seed by hand) — so it seeds diversity
without seeding symmetry, which is precisely the combination the lattice and the rings each fail.

The spiral alone does not seed the corners, and I have to add them by hand, which the earlier corner
analysis makes obvious. The spiral is a disk of radius `0.5` centred at `(0.5, 0.5)`, so its outermost
points only just graze the mid-edges and it reaches the four corners barely if at all — yet the corners
are precisely the prize real estate, since a corner disk is confined by two free walls that recede as
it grows. So I plant four circles explicitly near the corners, at `(0.07, 0.07)` and its three mirror
images, and lay the remaining `22` on the golden-angle spiral. A small random jitter on the whole
layout means repeated structured starts are not identical, so I get a family of good-looking starts
rather than one, and SLSQP refining any of them lands in a strong basin far more often than a uniform
scatter does. This is the rightward shift of `F` I was after: the structured start is already a
plausible packing, so the median refined optimum is good, not just the rare tail.

Let me put a number on that shift, because it is the whole justification for the extra machinery. The
`22` spiral points fill a disk of radius `0.5` at areal density `22/(π·0.25) ≈ 28` per unit area, so
their nearest-neighbour spacing is about `1/√28 ≈ 0.189` — right at the grid's tangency scale, and
almost perfectly even, so essentially *zero* pairs start closer than the tangency distance `0.1`.
Contrast the uniform scatter of the previous rung, which carried about a dozen pairs within `0.1` and
whose clumps dragged its LP-radii start down to a sum near `1.2`, below the grid. The spiral has no
clumps to un-knot, so its LP-radii start already sits up near the grid's `2.5` before SLSQP takes a
step, with the four corner seeds free to grow on two walls. So the two kinds of start hand SLSQP
configurations more than a full unit of radius-sum apart in quality, and the one that begins looking
like a packing is the one that ends in a strong basin — which is exactly what "shifting `F`
rightward" means in concrete geometric terms rather than as an abstraction about distributions.

The local engine stays the joint centre-and-radius SLSQP, unchanged, and I want to restate why the
joint form is non-negotiable rather than the tempting LP-radii-plus-separate-centres split. On the
single-start rung I traced, on the grid, a corner disk pinned at `0.1` by two walls and two neighbours,
and showed that growing it requires a *coordinated* shove of a whole chain of circles — the value of
moving one centre is how much radius that move unlocks across all its neighbours at once, and only a
solver carrying centres and radii in the same quadratic model can price that trade in a single step. A
split scheme freezes one block and optimises the other, and it structurally cannot make those
coordinated moves; the whole climb above the grid lives in exactly the moves it cannot make. So joint
SLSQP, with the radii re-tightened to their exact LP optimum after each refinement, stays the workhorse
inside every phase of this rung.

Now the memory, which is the real escape from the plateau: iterated perturbation chains. Rather than
discard refined packings, I take the best one found so far, perturb it, and re-refine — and repeat,
forming a chain that walks from good basin to better basin. The perturbation jiggles a random subset of
the centres by a Gaussian kick and then lets SLSQP repair and re-optimise; if the result is feasible
and better it becomes the new incumbent, and if not I try again from the same incumbent. The two design
choices inside the kick both come from thinking about what the chain needs to do. First, how many
circles to perturb: I draw the subset size `k` uniformly from `1` to `⌊N/2⌋ = 13`. Perturbing a single
circle is a local repair that can at most reshuffle one contact; perturbing thirteen rearranges half
the packing and can change the contact topology wholesale. The frontier gap needs *both* — tiny radius
rebalancing within a fixed contact graph *and* the occasional topology change that opens a better graph
— so sampling `k` across all scales lets one chain make moves of every size rather than committing to
one. I can put a number on why the subset size has to span those scales. A jammed disk touches on the
order of four to six neighbours, and a kick of `σ = 0.06` — more than half a typical radius — displaces
a circle far enough to break essentially all of its contacts. So perturbing `k = 1` circle frees
roughly five contacts, a genuinely local repair that SLSQP re-seats without disturbing the rest of the
graph; perturbing `k = 13` frees on the order of `13 × 5 ≈ 65` contacts — well past the `~50` that
define the entire rigid web — so the packing is effectively re-melted and SLSQP re-jams it into a
possibly-different topology. Drawing `k` uniformly over `1..13` therefore samples the full range from
surgical re-seating to wholesale re-melting, and the reason to want both is that the last `~0.014` of
sum is won by a mix of the two: most steps are small rebalancings inside a fixed contact graph,
punctuated by the rare re-melt that finds a strictly better graph.

Second, how big a kick: I run an adaptive schedule. On an improvement I reset the kick to its full
size `σ = 0.06` and explore boldly from the new, better incumbent; on a failure I cool it,
`σ ← 0.94 σ`, so the chain searches finer and finer around a configuration it cannot immediately beat,
down to a floor `σ = 0.012`. That reset-on-success, shrink-on-failure rule is the classic
exploration/exploitation adaptation — the same logic as the `1/5`-success rule in evolution strategies:
keep the step large while it is paying, shrink it when it stops. The floor matters too: below about
`0.012`, a kick is smaller than the scale at which a circle can change which neighbours it touches, so a
smaller kick just jiggles within the same basin and wastes the SLSQP repair; flooring the cool-down
keeps every perturbation capable of at least a small topology change.

The schedule's timescale is worth checking against the chain length, so the two are not accidentally
mismatched. From `σ = 0.06`, cooling by `0.94` per failure reaches the `0.012` floor when
`0.94ⁿ = 0.2`, i.e. `n = ln 0.2 / ln 0.94 ≈ 26` consecutive failures; with chains of about `25` steps,
a chain that never improves cools from `0.06` down to `0.06 · 0.94²⁵ ≈ 0.0128`, essentially the floor,
by its last step. So a single chain sweeps the kick across its entire useful range — from a coarse
`0.06`, more than half a typical radius and enough to eject a circle from its contacts, down to a fine
sub-contact `0.012` — exactly once, and the reset-on-success rule re-arms that sweep every time the
chain finds a better basin. The kick is also not optional, and a degenerate check makes that sharp: at
`σ = 0` each step would run SLSQP from the incumbent's own centres, and since the incumbent is already
a KKT point of the joint problem the solver returns it unchanged and the chain spins in place forever.
So the perturbation is the entire engine of the chain; the re-SLSQP only repairs and re-optimises what
the kick disturbs, and with the kick off there is nothing to repair and no progress to make.

Why perturb-from-incumbent rather than simply restart is the crux, and it is worth stating as
mechanism, not preference. A fresh random restart discards all information and draws again from `F`; a
perturbation starts from a *known* good basin and hops to a nearby one, so it explores the neighbourhood
of the frontier basins that uniform scatter reaches with probability `q ≈ 0`. The coordinated lateral
moves that climb the last `~0.014` — long sequences of mostly-sideways rearrangements that each unlock a
sliver of radius — are reachable from a good configuration by a small perturbation but are
astronomically unlikely to be hit by a cold scatter, because they require already being *near* the good
graph. The structured restarts find a strong basin; the chains then mine it. And there is a falsifiable
prediction in this: the chain's improvements should look *different* from the multi-start record
signature. Where restart records are i.i.d. and thin out like `1/k`, a chain that keeps improving from
its incumbent can produce runs of correlated gains — one good perturbation opens several more — so I
expect the perturbation phase to show clustered improvements rather than the isolated, thinning ones
the previous rung showed.

The perturbation's job is best understood in the language of the contact graph the SLSQP rung set up. A
jammed packing is a rigid contact network of roughly `2N ≈ 52` active contacts, and to move from one
such network to a *better* one the search cannot slide continuously — the two networks are different
combinatorial objects, separated by a barrier where at least one contact must break and another form. A
pure local solver started at a jammed point cannot cross that barrier: it is at a KKT point and every
direction that would improve the sum is blocked by an active constraint. The Gaussian kick is precisely
the barrier-crossing move — it displaces a subset of centres far enough to break some of their contacts,
momentarily un-jamming the packing, and then the re-SLSQP re-jams it into whichever network is now
nearest, which with some probability is a better one. That is why the kick has a floor at the
sub-contact scale, and why perturbing more circles (large `k`) is what enables a *topology* change
while perturbing one (small `k`) only re-optimises within the current network: the number of contacts a
kick can break scales with how many circles it moves. Seen this way the whole chain is a walk on the
graph of jammed networks, the kick size choosing how far along that graph each step reaches, and the
frontier is a network many such steps from any random start — reachable by walking from a good
incumbent, unreachable by a cold scatter that lands in a random network far away from it.

I run this as three phases under a fixed wall-clock budget of about `520` seconds, split
`0.4 / 0.6 / 1.0`. The first `~208` seconds are structured spiral-plus-corner restarts with joint
SLSQP, to find a strong basin quickly; at a second or two per refinement that is on the order of a
hundred-plus structured starts, plenty to find a good incumbent. The next `~104` seconds mix in pure
random restarts as cheap insurance — the structured prior is opinionated, and I want a guard against it
having boxed the search into one region of configuration space, so a slab of ordinary uniform scatter
buys diversity the spiral might miss. The final `~208` seconds are the iterated perturbation chains
launched from the incumbent, each chain `~25` perturb-and-re-SLSQP steps, grinding the best packing
upward. Throughout I keep the single best feasible packing ever seen, verify it against the real
constraints, and re-tighten its radii by LP, so the reported number is the genuine sum of a genuinely
feasible arrangement. The seed is fixed at `7` so the whole three-phase run is one reproducible object.

What do I expect, and where do I stop honestly? Two predictions I can check against the phase trace.
First, the structured init alone should already clear the random-multi-start plateau of `2.6221` — the
rightward shift of `F` means even the structured-restart phase, before any chains, should land above
where `120` uniform starts saturated. Second, the perturbation chains should then add a further
increment on top of that, mining the incumbent past what any single refinement reaches, and the shape
of the trace should be structured-phase-plateaus-fast then chains-grind-slowly-upward. In area terms I
can pin a rough target: a sum near `2.627` forces `Σ rᵢ² ≈ 2.627²/26 ≈ 0.2655`, disk area `≈ 0.834`, so
this rung should fill roughly `(0.834 − 0.791)/0.209 ≈ 21%` of the void the grid left — up from the
single start's `~9%` and multi-start's mid-teens, and closing on the frontier's `~23%`. But I am
running a single constructor under a bounded budget — minutes, not the orders-of-magnitude more
autonomous compute that produced the record `2.635988438568` — and the published frontier
(`2.63586` AlphaEvolve, `2.635983` ShinkaEvolve, `2.635988` AutoEvolver) is a band separated by only
parts in the sixth decimal place, each part costing a very long chain of mostly-lateral perturb-and-
re-SLSQP moves through near-degenerate basins. A nine-minute run cannot make that many moves. So I
expect to land *in the band's neighbourhood but below the record* — the honest outcome of running the
right pipeline with a fraction of the compute.

There is a design-space check I owe myself before I commit, because "perturbation chain" is not the
only memory mechanism I could bolt on. I could run simulated annealing on the full `78`-variable
configuration, but a temperature schedule over a continuous space with `429` hard constraints is
awkward — every proposal has to be projected back to feasibility, and that projection is itself the
SLSQP solve I am already doing, so SA would just wrap the same solver in a more fragile acceptance rule.
I could run a genetic or population method that crosses over two good packings, but a crossover child
that splices half of one packing with half of another is almost always overlapping and infeasible,
needing so much repair that the parents' structure is destroyed — crossover has no natural meaning for
jammed disks. Iterated local search with SLSQP as both the repair and the local optimiser is the clean
choice precisely because the one operation I trust — joint SLSQP to a KKT point — *is* the repair, so
the perturbation only has to propose a nearby configuration and SLSQP does the rest. That is why the
chain, not annealing or crossover, is the memory mechanism here.

I can also predict the *shape* of the packing this returns, as a structural check against everything
the earlier rungs established. It should be irregular — the grid analysis proved any substantial gain
over `2.5414` cannot be structured — with a few large disks pressed into the corners and along the
edges and the interior filled by many unequal gap-fillers of graded size, the vocabulary the grid
lacked. Every circle should be pinned: at a local maximum nothing can grow, so each radius is held by
at least one active wall or contact, and the active contacts should form a roughly isostatic web of
order `2N ≈ 50` constraints out of the `429`. If the returned configuration instead had a slack circle
floating free, or looked grid-like and regular, I would distrust the run — the LP re-tightening exists
precisely to grow any under-tight radius to its contact. So the falsifiable claim has a geometric half
as well as a numeric one.

So the pipeline is the endpoint method: structured init supplies the structure random restart lacked,
iterated chains supply the memory, and joint SLSQP remains the engine underneath both — exactly the
construction the frontier systems converged on. Run under a bounded budget it should reach the frontier
neighbourhood, and the residual gap from my measured number to `2.635988438568` will be the honest
measure of the part of this problem that is bought purely with search budget, not with a better
algorithm — the distance a nine-minute run leaves for sustained search to close. That residual,
whatever it turns out to be, names what the final rung must confront: not a cleverer local move, since
no such move closes a sixth-decimal gap within a bounded run, but the specific configuration that only
a much longer search reaches.
