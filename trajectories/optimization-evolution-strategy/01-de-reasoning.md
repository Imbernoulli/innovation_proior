The honest place to start is the question that decides every direct-search method: by how much, and in
what direction, do I perturb a candidate point? I have a scalar `f(x)` over a box of reals, and the panel
is deliberately nasty — Rastrigin is a multimodal egg-carton, Rosenbrock a thin curved valley, Ackley a
flat plateau with a needle at the origin, two at 30D and one at 100D. Every method I could fill the
scaffold with runs the same two beats: take a vector, make a variation of it, decide whether to keep the
variation. Where they all live or die is the perturbation scale — too large and the search thrashes and
never settles, too small and it crawls or sinks into the first local pit and stays. The cruel part is
that the right amount is not a constant: early, scanning a wide box, I want big jumps; late, polishing, I
want tiny ones, and it differs per landscape. So before I pick an algorithm I want to be clear about
where the scale comes from in each candidate, because that is what separates a floor baseline from
something that actually moves.

And I should count the budget first, because scarcity governs which failures I can afford. The harness
gives `pop_size=200` and `n_generations=500` on the three 30D problems, `1000` on the 100D one. One
evaluation per member per generation is `200 × 500 = 100000` evaluations at 30D and `200 × 1000 = 200000`
at 100D — read the other way, each member gets at most 500 (or 1000) attempts to be improved over the
whole run. That is not many, when the job is to place 30 or 100 coordinates each near their optimum. So I
cannot afford a method that wastes evaluations on trials structurally doomed to be rejected, nor one so
timid it needs thousands of generations to refine a basin. Both are scale failures in disguise.

The scaffold's default fill answers the scale question the genetic-algorithm way: simulated binary
crossover plus polynomial mutation, both with a fixed distribution index η=20, under tournament
selection. Stare at how it sets scale. The SBX spread and the polynomial mutation width are governed by
η, a number I fix once by hand, which does not contract as the population converges and does not align
with the shape of the valley. Evolution strategies have the same disease in subtler form: σ *is* the
scale, but controlling it is a second optimization problem — the 1/5 rule, self-adapted per-coordinate σ,
or a full covariance, each with its own learning machinery that lags the true geometry. Simulated
annealing answers with a cooling schedule and an enormous evaluation appetite I cannot afford. The
pattern is everywhere the same: the perturbation's scale is set by an *external device* sitting outside
the actual configuration of the candidate points — a temperature, a controlled σ, a fixed mutation width
— brittle when the landscape violates its assumptions, and a tuning burden. I do not want a better
device. I want the scale to not be a device at all.

Nelder–Mead hints that this is possible: it keeps `D+1` vertices and reads the size of its next move
straight off how big the simplex currently is — reflect, expand, contract relative to its own spread, no
σ, no schedule. Big simplex, big moves; as it collapses into a basin the moves shrink with it. That is
exactly the self-scaling I want. But it is one tiny figure of `D+1` points, a purely local minimizer that
contracts into a basin and cannot climb back out — useless on a Rastrigin egg-carton. What it carries is
the idea: extract the move scale from the configuration of points you already hold. The scaffold hands me
a whole population of 200. What if I extract the move scale from the *population itself*?

Early in the run the 200 members are scattered across the box, so the cloud is large; late, as they
converge, the cloud is tight around the optimum. That spread — the typical distance between members — is
precisely the quantity I keep wanting σ to be: big while exploring, small while refining, and it tracks
the phase of the search on its own, for free, because the population literally contracts as it converges.
The only question is how to turn "the population is spread this much" into a perturbation vector I can
add.

The naive move is to estimate the population covariance and sample a Gaussian from it — but that is ES
with extra steps, a `D×D` matrix to estimate, store, and keep current. At 100D that is `10000` entries
re-estimated every generation; drawing a correlated Gaussian the clean way needs a factorization,
`O(D³) = 10^6` operations per refresh, and even a rank-one update is `O(D²) = 10^4` per sample times 200
samples per generation. And it lags: the covariance I sample this generation was fit to where the
population *was*. So what is the simplest object built from the population whose typical magnitude *is*
the population's spread? The difference between two members. Pick `x_r2` and `x_r3` at random and form
`x_r2 − x_r3`: its length is, by construction, a sample of the typical pairwise distance in the current
population — large when the cloud is large, small when it is small. I never estimated anything; I
subtracted two points I already hold, `O(D)` work and zero storage. Add a scaled copy to a third member:
`v = x_r1 + F·(x_r2 − x_r3)`.

Make the self-scaling quantitative, since the whole argument rests on it. Early, per coordinate the
members are independent draws from a uniform of width `L = hi − lo`; for two i.i.d. uniforms the mean
absolute difference is `E|X − Y| = L/3`. So the raw per-coordinate difference has typical magnitude
`L/3`, and with `F = 0.5` the step is about `L/6 ≈ 0.167 L`. On Rastrigin, `L = 10.24`, so an early
per-coordinate step is around `1.7` — a genuine hop across a couple of the unit-spaced pits. Run the
population forward to where it has clustered with per-coordinate spread `σ`: two members are then roughly
Gaussian, their difference Gaussian with standard deviation `σ√2`, mean absolute difference
`σ√2 · √(2/π) = 2σ/√π ≈ 1.13 σ`, and with `F = 0.5` the step is about `0.56 σ` — just over half the
current spread. So as the cloud contracts from box-width to `σ`, the step contracts with it, from
`~0.167 L` down to `~0.56 σ`, with no schedule anywhere. This is what ES needed a 1/5 rule or a
self-adapted σ to fake, falling out of pure subtraction.

And the difference vector inherits not just the cloud's size but its *shape*. Picture Rosenbrock's long
curved valley, a thin bent ribbon. The population, selected toward low cost, spreads out *along* the
ribbon and stays thin across it, so two random members differ much more along the valley than across it,
and `x_r2 − x_r3` points preferentially down the valley. The perturbation is automatically anisotropic in
exactly the way the landscape is — the correlated, rotation-aware search ES needs an explicit covariance
for, read off for the cost of one subtraction, because the differences sample a cloud that has already
aligned itself with the contours. This is the property I expect to matter on Rosenbrock, where the
scaffold default's axis-aligned polynomial mutation has no way to align with the diagonal valley.

Now the details, each a real choice with a failure mode. The base vector `x_r1` — random, or the current
best? Always perturbing the best converges fast on easy unimodal bowls but concentrates every trial
around one point and loses the diversity that lets the population escape local minima — dangerous on the
multimodal costs I care most about, Rastrigin with its grid of pits. A random base keeps exploration
broad and the population diverse. So the cautious default is a random base: the "rand" scheme. The factor
`F` then amplifies an already-adapted difference vector; I keep it because the raw typical difference is
not necessarily the *optimal* step length — I may want a bit less than a full inter-member hop to
converge cleanly, or a bit more on a rugged surface. Too small and the steps die before the population
reaches the optimum; too large and trials overshoot and never settle. A value around a half keeps the
mean step useful, and crucially I do *not* adapt `F` over time, because the difference vector already
carries the time-adaptation. `F = 0.5`.

The indices `r1, r2, r3` must be distinct from each other and from the target `i`. Distinct from each
other because if `r2 = r3` the difference is the zero vector and the mutant collapses to the bare base —
a wasted evaluation the budget cannot spare. Distinct from `i` because the move should be informed by
*other* members, not coupled to the vector it competes against. Three indices all different from `i`
means at least four distinct members, `pop_size ≥ 4`, comfortably satisfied at 200. The population
initializes uniformly over the box, the honest default with no prior.

The bare mutation, though, moves *all* `D` coordinates of `x_r1` at once, by the full difference vector.
On a separable problem like Rastrigin, where coordinates are independent, perturbing every coordinate
simultaneously is wasteful — I would rather sometimes change a few coordinates and let the good values of
the others ride along. More generally I want to *control* how many coordinates of the mutant reach the
trial, to inject diversity and tune to the dependency structure. So I will not let the mutant become the
trial wholesale; I will mix it with the target coordinate by coordinate. That is the crossover step.

Build the trial `u` from the mutant `v` and the target `x_i` per coordinate: `u_j = v_j` with probability
`CR`, otherwise `u_j = x_{i,j}` — each coordinate an independent coin. `CR ∈ [0,1]` is a dial on the
dependency structure: near 1, almost every coordinate comes from the mutant, so the trial is essentially
the full difference-vector move — right for a *non-separable* problem like Rosenbrock, where coordinates
must move together along the curved valley. Near 0, only a coordinate or two change per trial — suiting a
*separable* problem optimized axis by axis. For the general non-separable case I lean high, `CR = 0.9`,
accepting this is not ideal for separable Rastrigin (a point I will pay for and revisit). One degenerate
case: if every coin says "keep the target", `u = x_i` and the evaluation is wasted. I force at least one
mutant coordinate by picking a random `j_rand` up front: `u_j = v_j` if `random() < CR or j == j_rand`.
Now the all-target copy cannot occur.

Count exactly how many coordinates move, because that number is the mechanism that makes or breaks each
landscape. `j_rand` is always from the mutant; every other coordinate with probability `CR`, so the
expected mutant-coordinate count is `1 + (D − 1)·CR`. At `CR = 0.9` that is `1 + 29·0.9 = 27.1` of 30 on
the 30D problems and `1 + 99·0.9 = 90.1` of 100 on the 100D one — roughly 90% of the vector overwritten
by a coordinated move on every trial. On Rosenbrock this is exactly right: the coordinates *have* to move
together, and a trial that nudged only two of them would step off the ribbon. But on separable Rastrigin,
reason about acceptance: the target is a selection survivor, so most of its coordinates already sit near a
per-coordinate local minimum, and a random change to any one is more likely to raise its contribution
than lower it. On a separable cost the total change is the *sum* of per-coordinate changes, and
acceptance needs that sum `≤ 0`. Disturbing ~27 coordinates sums ~27 mostly-adverse independent changes,
and the chance they cancel to a net improvement falls off sharply as the count grows. Changing one or two
gives a genuinely-better coordinate a chance to carry the trial; changing twenty-seven buries it. That is
the precise sense in which `CR = 0.9` is wrong on a separable egg-carton, and it should be worst at 100D
where the coordinated move is 90 wide.

Now acceptance: greedy and one-to-one. Compare `u` against *its own target* `x_i` and let it take the
slot if `cost(u) ≤ cost(x_i)`; otherwise `x_i` stays. One-to-one keeps `pop_size` fixed and preserves
diversity far better than a global "keep the best `N`" truncation would — truncation lets a few good
basins crowd out everything and collapse the spread, and with it the self-scaling difference vectors;
one-to-one replacement lets a member that is merely best in its own lineage survive, so the cloud stays
spread and keeps supplying varied differences. It is also elitist per slot: a slot only ever changes to
something no worse, so per slot the cost is monotone non-increasing and the population minimum can only
fall or hold. This also lets the scale filter itself — once a slot is deep in a basin, a difference built
from far-flung members overshoots and is rejected, so only members whose neighbours have also contracted
propose steps small enough to be accepted there. Does greedy acceptance trap me the way plain greedy
search does? No, and this is where the population earns its keep: a single member sliding into a local
basin cannot drag the others, and difference vectors from members exploring other basins keep proposing
jumps that pull a trapped member out. Per generation this whole thing is `O(N·D)` arithmetic plus `N`
objective calls, storing exactly the `N·D` population reals — no `D²` matrix, no `O(D²)`–`O(D³)` refresh:
the same scale-and-orientation adaptation the covariance route buys, for a subtraction.

Now assemble the loop in the scaffold's vocabulary. I sweep the population once per generation; for each
target `i` I pick `r1, r2, r3` from the other indices, form the mutant, binomial-cross it with the target
(forcing `j_rand`), clip the trial into the box with the provided `clip_individual` logic, evaluate, and
— if no worse — overwrite `pop[i]` *in place, immediately*. In-place overwrite during the sweep lets a
later target draw `r1/r2/r3` from members already updated this generation; this is DE with immediate
replacement, a touch greedier than buffering a whole next generation, and it is the form the single-loop
structure invites. DE's selection is greedy inside the loop, its mutation is the difference vector, its
crossover is binomial, so the three operator stubs (`custom_select`, `custom_crossover`, `custom_mutate`)
go unused and the whole search lives in `run_evolution`. The control variables are just `F`, `CR`, and
`pop_size` — three numbers, none a schedule or an adapted matrix. The full module is in the answer.

What do I expect, and where should it hurt? The self-scaling anisotropic difference vector should make
this far more sample-efficient than the fixed-η operators on the smooth, non-separable problems: on
Ackley's single broad bowl once off the plateau, and on Rosenbrock's curved valley, the differences
should track the geometry and drive the best fitness down hard. But three things worry me. First,
`CR = 0.9` is wrong for separable Rastrigin — the 27-of-30 (90-of-100) coordinated move is exactly what a
separable survivor rejects, so I expect Rastrigin, and especially 100D Rastrigin, to be the worst
showings, possibly badly so. Second, a fixed `pop_size=200` with a random base is a slow converger — each
member gets only 500 (or 1000) attempts and `rand` trades speed for diversity, so I may run out of budget
with the search still creeping at the wall. Third, there is no adaptation of `F` or `CR` at all — the one
setting that suits Rosenbrock is the setting that fails Rastrigin. And with a `rand` base and no elite
pull, some seeds may scatter and never enter the basin on the harder surfaces, inflating the
seed-averaged mean well above a good seed's value. So the direction I expect: a clean, near-zero result
on Ackley, a respectable Rosenbrock, a poor Rastrigin at both dimensions — the failure being the fixed
`CR` colliding with separability and the slow `rand` base burning the budget. That diagnosis is exactly
what should push toward a mutation that is local and per-coordinate rather than globally scaled from one
fixed distribution.
