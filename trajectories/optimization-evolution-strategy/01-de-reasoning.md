I am starting the climb from the bottom, and the honest place to start is the question that decides every
direct-search method: by how much, and in what direction, do I perturb a candidate point? I have a scalar
`f(x)` over a box of reals, and the panel is deliberately nasty — Rastrigin is a multimodal egg-carton,
Rosenbrock is a thin curved valley, Ackley is a flat plateau with a needle at the origin, and two of
these run at 30D, one at 100D. The skeleton of every method I could fill the scaffold with is the same
two beats: take a vector, make a variation of it, decide whether to keep the variation. And the place
they all live or die is the perturbation scale. Too large and the search thrashes and never settles; too
small and it crawls or sinks into the first local pit and stays. The cruel part is that the right amount
is not a constant — early, scanning a wide box, I want big jumps; late, polishing, I want tiny ones — and
it differs per landscape. So before I pick an algorithm I want to be clear about where the scale comes
from in each candidate, because that is what will separate a floor baseline from something that actually
moves.

The scaffold's default fill answers the scale question the genetic-algorithm way: simulated binary
crossover plus polynomial mutation, both with a fixed distribution index η=20, under tournament selection.
That is a legitimate method and I will get to it on the next rung, but stare at how it sets scale. The SBX
spread factor and the polynomial mutation width are governed by η, a number I fix once, by hand, and which
does not contract as the population converges and does not align with the shape of the valley. The
mutation injects fresh variation from a fixed-width distribution that is right for at most one phase of one
problem. Evolution strategies have the same disease in a subtler form: σ *is* the scale, but controlling
it is a second optimization problem — the 1/5 rule, self-adapted per-coordinate σ, or a full covariance,
each with its own learning machinery that lags the true geometry and its own knobs. Simulated annealing
answers with a cooling schedule, a dozen control variables and an enormous evaluation appetite I cannot
afford. The pattern is the same everywhere: the perturbation's scale is set by an *external device* sitting
outside the actual configuration of the candidate points — a temperature, a controlled σ, a fixed
mutation width. Each device is brittle when the landscape does not match its assumptions, and each is a
tuning burden. I do not want a better device. I want the scale to *not be a device at all*.

Is that possible? Nelder–Mead hints that it is: it keeps `D+1` vertices and reads the size of its next
move straight off how big the simplex currently is — reflect, expand, contract relative to its own spread,
no σ, no schedule. When the simplex is big the moves are big; as it collapses into a basin the moves
shrink with it. That is exactly the self-scaling I want. But it is one tiny figure of `D+1` points, a
purely local minimizer that contracts into a basin and cannot climb back out — useless on a Rastrigin
egg-carton. What it carries, though, is the idea: extract the move scale from the configuration of points
you already hold. Nelder–Mead does this with one collapsing simplex. The scaffold hands me a whole
population of `pop_size=200`. What if I extract the move scale from the *population itself*?

Look at what a population actually contains as a distribution of points. Early in the run the 200 members
are scattered across the box, so the cloud is large. Late, as they converge, the cloud is tight around the
optimum. That spread — the typical distance between members — is precisely the quantity I keep wanting σ
to be: big while exploring, small while refining, and it already tracks the phase of the search on its own,
for free, because the population literally contracts as it converges. The information ES and SA pay for is
sitting in the geometry of the points I am already carrying. The only question is how to turn "the
population is spread this much" into a perturbation vector I can add.

The naive move is to estimate the population covariance and sample a Gaussian from it — but that is ES
with extra steps, a `D×D` matrix to estimate, store, and keep current, the very machinery I am trying to
delete, and at 100D that matrix is 10,000 entries. I want a perturbation of the right scale and
orientation without ever forming a distribution. So: what is the simplest object built from the population
whose typical magnitude *is* the population's spread? The difference between two members. Pick two members
`x_r2` and `x_r3` at random and form `x_r2 − x_r3`. Its length is, by construction, a sample of the typical
pairwise distance in the current population — large when the cloud is large, small when it is small. I
never estimated anything; I subtracted two points I already have. Add a scaled copy to a third member as
the perturbation: `v = x_r1 + F·(x_r2 − x_r3)`. Now the self-scaling is automatic and exact. Early, members
are far apart, the difference is a big vector, the step is exploratory; late, members have clustered, the
difference is tiny, the step refines. The scale rides the population's contraction with zero parameters and
zero schedule. This is what ES needed a 1/5 rule or a self-adapted σ to fake, falling out of pure
subtraction.

There is a second gift I almost miss, and it is the orientation. Difference vectors are not isotropic —
they inherit the *shape* of the population cloud, not just its size. Picture Rosenbrock's long curved
valley, a thin bent ribbon where the good region lives. The population, selected toward low cost, spreads
out *along* the ribbon and stays thin across it, so two random members differ much more along the valley
than across it, and `x_r2 − x_r3` points preferentially down the valley. The perturbation is automatically
anisotropic in exactly the way the landscape is — it proposes moves along directions where the population
has discovered it can move productively. This is the correlated, rotation-aware search ES needs an explicit
covariance for, and I get it for nothing because the differences are samples of a cloud that has already
aligned itself with the contours. The population does the covariance estimation implicitly, by where its
members sit. This is precisely the property I expect to matter on Rosenbrock, where the scaffold default's
axis-aligned polynomial mutation has no way to align with the diagonal valley.

So the difference-vector perturbation is the core. Now I nail the details, each a real choice with a
failure mode. The base vector `x_r1` — random, or the current best? Always perturbing the best converges
fast on easy unimodal bowls but concentrates every trial around one point and loses the diversity that lets
the population escape local minima — dangerous on the multimodal costs I care most about (Rastrigin, with
its grid of pits). A random base keeps exploration broad and the population diverse. So the cautious
general default is a random base: the "rand" scheme. The factor `F` amplifies an already-adapted difference
vector. Why have it at all if the difference is already the right scale? Because the raw typical difference
is not necessarily the *optimal* step length — I may want a bit less than a full inter-member hop to
converge cleanly, or a bit more to push exploration on a rugged surface. `F` is a single dial on a vector
that is already pointed and sized correctly, robust and easy to choose, unlike σ which had to *be* the
whole scale. Too small and the steps die before the population reaches the optimum (premature convergence);
too large and trials overshoot and the population never settles. The conventional range is `0 < F ≤ 2`; a
value around a half keeps the mean step useful, and crucially I do *not* adapt `F` over time, because the
difference vector already carries the time-adaptation. I take `F = 0.5` as the task default.

Now the indices. I need `r1, r2, r3` distinct from each other and from the target index `i` whose slot I
am trying to improve. Distinct from each other because if `r2 = r3` the difference is the zero vector — a
wasted trial. Distinct from `i` because the move should be informed by *other* members, not coupled to the
very vector it competes against. Three indices all different from `i` means at least four distinct members,
`pop_size ≥ 4`, which is comfortably satisfied at 200. Initialize the population uniformly over the box,
the honest default with no prior — the scaffold's `make_individual` does exactly this.

Let me sanity-check the mutation by itself. Suppose every trial were just `v = x_r1 + F·(x_r2 − x_r3)`,
accepted greedily if it beats its target. Does it work as a search? Members improve one slot at a time, the
cloud contracts toward low-cost regions, the differences shrink with it, the steps anneal themselves — the
dynamics are sound. But I have a worry: the perturbation moves *all* `D` coordinates of `x_r1` at once, by
the full difference vector. On a separable problem like Rastrigin, where coordinates are independent,
perturbing every coordinate simultaneously is wasteful — I would rather sometimes change a few coordinates
and let the good values of the others ride along. More generally I want to control *how many* coordinates
of the mutant make it into the trial, to inject diversity and tune to the dependency structure. So I will
not let the mutant become the trial wholesale; I will *mix* it with the target coordinate by coordinate.
That is the crossover step.

Build the trial `u` from the mutant `v` and the target `x_i` per coordinate: take `u_j = v_j` with
probability `CR`, otherwise `u_j = x_{i,j}` — each coordinate decided by an independent coin, a binomial
choice across the `D` coordinates. `CR ∈ [0,1]` is a dial on the dependency structure: near 1, almost every
coordinate comes from the mutant, so the trial is essentially the full difference-vector move — what I want
on a *non-separable* problem like Rosenbrock, where coordinates must move together along the curved valley.
Near 0, only a coordinate or two change per trial — suiting a *separable* problem optimized axis by axis.
For the general non-separable case I lean high, `CR = 0.9`, accepting that this is not the ideal setting for
separable Rastrigin (a point I will pay for in the numbers and revisit). There is a degenerate case in the
coin flipping: if every coin says "keep the target," `u = x_i` exactly and the evaluation is wasted. I
force at least one coordinate from the mutant by picking a random index `j_rand` up front and always taking
that coordinate from `v`: `u_j = v_j` if `random() < CR or j == j_rand`, else `u_j = x_{i,j}`. Now the
construction cannot produce the all-target copy.

Now acceptance. I have a trial `u` for target slot `i`. Greedy and one-to-one: compare `u` against *its
own target* `x_i`, and let `u` take the slot if its cost is no worse — `cost(u) ≤ cost(x_i)`; otherwise
`x_i` stays. Two things make this exactly right. First, it is one-to-one — each trial competes only with
the single member it was built to replace, not the whole population, which keeps `pop_size` fixed and, more
importantly, preserves diversity far better than a global "keep the best `N`" truncation would. Truncation
would let a few good basins crowd out everything and collapse the spread (and with it the self-scaling
difference vectors); one-to-one replacement lets a member that is merely best *in its own lineage* survive,
so the cloud stays spread and keeps supplying varied differences. Second, it is elitist per slot: a slot
only ever changes to something no worse, so the best cost is monotone non-increasing — the method never
throws away its best find. Does greedy acceptance trap me the way plain greedy search does? No, and this is
where the population earns its keep: a single member sliding into a local basin cannot drag the others, and
difference vectors from members exploring other basins keep proposing jumps that pull a trapped member out.

Now I assemble the loop in the scaffold's vocabulary, and here the literal edit diverges from the generic
textbook version in one detail worth stating, because the harness exposes the loop and I must fill it as it
runs. I sweep the population once per generation; for each target index `i`, I pick `r1, r2, r3` from the
other indices, form the mutant, binomial-cross it with the target (forcing `j_rand`), clip the trial back
into the box with the provided `clip_individual` logic, evaluate, and — if the trial is no worse —
overwrite `pop[i]` *in place, immediately*. In-place overwrite during the sweep means a later target in the
same generation can draw `r1/r2/r3` from members already updated this generation; this is the standard
"DE with immediate replacement" and it is a touch greedier than buffering a whole next generation, but it
is the form the scaffold's single-loop structure invites and it costs nothing. The scaffold's three
operator stubs (`custom_select`, `custom_crossover`, `custom_mutate`) are not used by DE — DE's
selection is greedy inside the loop, its mutation is the difference vector, its crossover is binomial — so
I leave them as no-ops and write the entire search inside `run_evolution`. The control variables are just
`F`, `CR`, and `pop_size`: three numbers, none a schedule or an adapted matrix. The full scaffold module is
in the answer.

What do I expect, and where do I expect it to hurt — these are the falsifiable predictions this rung must
hand the next one. The self-scaling difference vector should make DE far more sample-efficient than the
scaffold default's fixed-η operators on the smooth, non-separable problems: on Ackley, whose basin is a
single broad bowl once you are off the plateau, and on Rosenbrock's curved valley, the anisotropic
differences should track the geometry and drive the best fitness down hard. But I am uneasy about three
things. First, `CR = 0.9` is wrong for separable Rastrigin — changing nine of ten coordinates per trial
fights the egg-carton's independence, so I expect Rastrigin (and especially 100D Rastrigin) to be DE's
worst showings, possibly catastrophically so, because at 100D the difference vectors must coordinate a huge
non-separable move on a separable problem. Second, with a fixed `pop_size=200` and only 500 generations,
DE/rand/1 is a relatively slow converger; `rand` base trades speed for diversity, so I may simply run out
of budget before refining. Third, classical DE/rand/1/bin has no adaptation of `F` or `CR` at all — the one
setting must serve four very different landscapes, and I have just argued the setting that is right for
Rosenbrock is wrong for Rastrigin. So my concrete expectation entering the next rung is: DE will land a
clean, near-zero result on Ackley, a respectable Rosenbrock, and a *poor* Rastrigin at both dimensions —
the failure being the fixed `CR` colliding with separability and the slow `rand` base burning the budget.
That diagnosis is exactly what should push me toward an operator suite whose mutation is local and
per-coordinate next, and a self-adapting DE after that.
