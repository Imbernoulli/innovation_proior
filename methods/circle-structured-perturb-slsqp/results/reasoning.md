The multi-start rung climbed to the low `2.62`s and then stalled, and before I reach for anything
fancier I want to be sure I am reading the plateau correctly rather than just assuming it. So I run
60 random uniform starts through joint SLSQP and watch the best-of-`k` curve. The per-start optima
scatter widely — mean `2.41`, and several starts collapse to a near-zero sum where the solver gets
stuck against the constraints — but the running best is the telling number: best-of-1 already sits
at `2.611`, best-of-10 is *still* `2.611`, best-of-20 is *still* `2.611`, and only by best-of-40 does
it creep to `2.6222`, where it stays through best-of-60. So twenty extra restarts after the twentieth
buy essentially nothing. That is the signature of an order statistic saturating: the best of many
i.i.d. draws improves like the tail of the per-start distribution, and once I am sampling the good
tail, halving the gap to the true optimum costs an exponentially growing number of fresh draws.
Uniform center scatters are a blunt seed — they almost never land in the rare basins at the top of
the frontier, and every restart is independent, so nothing learned from a good draw informs the next.

That diagnosis names two distinct weaknesses, and I should treat them separately. The starts have no
*structure* — they look nothing like a good packing — and the search has no *memory* — the best
configuration found is thrown away instead of mined. SLSQP itself is not the problem; every rung has
leaned on it and it keeps returning clean local optima. So I want to keep SLSQP as the local engine
and attack the seed and the memory.

First, structure in the initialization. Instead of scattering centers uniformly I want a start that
already resembles a good sum-of-radii packing: a few circles pushed into the corners and along the
edges, where two walls let a circle grow on two sides, and the interior filled by a spread that does
not clump. The device I reach for is the golden-angle spiral — point `i` at radius `∝ √(i/m)` and
angle `i·ψ` with `ψ = π(3 − √5)` — because the golden angle is the worst-approximable irrational
multiple of a turn, so successive points never fall on a small number of rays. But "does not clump"
is exactly the kind of claim I keep telling myself not to assert without checking, so I lay down the
22 interior points and measure the nearest-neighbor spacings directly. The spiral gives mean NN
spacing `0.177` with standard deviation `0.0046` — a coefficient of variation of `0.026`. For
contrast I replace the golden angle with a resonant one, `2π/8`, which should pile points onto eight
rays: that gives mean spacing `0.119` with std `0.015`, CV `0.129`. So the spiral's local spacing is
about five times more uniform, and the resonant version both clumps and packs tighter (smaller mean
spacing) — points crowding onto rays, exactly the failure I wanted to avoid. The spiral check holds.
So my structured init seeds the four corners explicitly, lays the remaining circles on the spiral,
and adds a small random jitter so repeated structured starts differ.

Second, the local refinement. Here I have a real decision to make, not a foregone one. For fixed
centers the optimal radii are a linear program: maximize `Σrᵢ` subject to `rᵢ + rⱼ ≤ dᵢⱼ` and
`rᵢ ≤ wallᵢ`, all linear. I confirm this is wired correctly on a case I can do by hand — two circles
at `(0.25,0.5)` and `(0.75,0.5)`, distance `0.5`, each `0.25` from its near wall: the LP returns
`[0.25, 0.25]`, sum `0.5`, with wall and separation both tight, as I computed. Push them to
`(0.4,0.5),(0.6,0.5)`, distance `0.2`: the LP returns sum `0.2` (it happens to load it all on one
circle, which is fine — only the sum is scored), separation binding. Good, the LP is right.

That makes a split scheme tempting: solve the radii by LP, and optimize only the centers in an outer
loop. The reduced objective `f(centers) = max Σrᵢ` is just a function of the 52 center coordinates, so
I could run SLSQP on `f` directly and let the LP handle radii exactly at every center configuration.
Against that sits the joint form I have been using: hand SLSQP all `3N` variables at once, with the
pairwise and wall constraints written out, and re-tighten the radii by LP at the very end. My instinct
is that the joint form must be better because centers and radii couple — the value of moving a center
is the radius it unlocks across all its neighbors at once. But that is an assertion, and the split
solves the *same* coupled problem (the LP sees all neighbors too), so I should actually race them.

On `n = 10` from a shared random start, six trials: the fair split — SLSQP maximizing the LP value
`f(centers)` — is not worse. It ties or beats the joint form in five of the six (joint `1.539` vs
split `1.572` on average), and on one start the joint form falls short by `0.115`. So my coupling story is wrong as
stated; per start, optimizing the LP value directly is at least as good. The split is not the inferior
optimizer I assumed it was.

What kills the split is cost, not quality, and that only shows up at the real size under a clock. Each
SLSQP step on `f(centers)` needs a finite-difference gradient over 52 coordinates, i.e. ~52 fresh LP
solves per gradient, and SLSQP takes many steps. I time one refinement at `N = 26`: the joint solve
finishes in `0.76 s`; the split solve, even with its iterations capped, takes `12.9 s` — about
**17× slower** per refinement. (In fact an uncapped split blew a two-minute budget on four starts.)
The whole point of this endpoint is to run thousands of refinements — every restart, every
perturbation step is one — under a fixed wall-clock budget, and `17×` fewer of them is fatal. So the
joint SLSQP stays, not because it finds a better optimum per start, but because it finds an
essentially-as-good one ~17× faster, which under a budget translates into ~17× more basins examined.
I keep the final LP re-tightening as cheap insurance that the reported radii are exactly optimal for
the final centers.

Third, memory. The order-statistic plateau says fresh restarts stop paying; what I have not yet used
is the incumbent. So instead of discarding refined packings I take the best one, perturb a random
subset of its centers by a Gaussian kick, re-refine with SLSQP, and repeat — a chain that walks from
good basin to neighboring better basin. If the re-refined packing is feasible and better it becomes
the new incumbent and I reset the kick size; if not, I cool the kick so the chain settles into finer
local exploration around a strong configuration. This is iterated local search with an annealed
perturbation scale. The reason it can do what restarts cannot is structural: the coordinated radius
gains near the optimum come from long sequences of mostly-lateral center moves, and you can only make
such a sequence if you have a good configuration to move *from* — which a fresh i.i.d. restart, by
construction, does not. The structured restarts find a strong basin; the chains mine it.

I assemble this as three phases under a fixed time budget. First, structured spiral/corner-seeded
restarts with joint SLSQP, to reach a strong basin quickly. Then a phase mixing in some pure random
restarts, as cheap insurance that the structured prior has not boxed the search into one region.
Finally, iterated perturbation chains launched from the incumbent, perturb-and-re-SLSQP. Throughout I
keep the single best feasible packing ever seen, verify it against the real constraints at
`atol = 1e-7`, and re-tighten its radii by LP, so the reported number is the genuine sum of a
genuinely feasible arrangement.

Where do I expect to land, and where do I stop honestly? The plateau I measured sits at `~2.622`;
structured init plus chains should clear it and push into the frontier neighborhood near `2.636`. But
I am running a single constructor under a bounded wall-clock budget — minutes, not the ~16.6 hours of
autonomous compute behind the record `2.635988438568`. The published frontier band (`2.63586`,
`2.635983`, `2.635988`) is separated by only a few parts in a million, and the saturation arithmetic I
just watched — exponentially more search per halving of the gap — says those last digits are bought
with budget, not with a cleverer step. So I expect to clear my own `2.622` plateau and reach the
band's neighborhood but below the record, and the residual gap from my measured number to
`2.635988438568` is the honest price of running this same construction with a fraction of the compute.
