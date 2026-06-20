The flat profile sat exactly at `2.0`, and the feedback made the obstruction concrete: with every piece
identical the autoconvolution is locked to a triangle with one sharp central peak, and there is no gradient to
follow — refining the grid does nothing. So the only way down is to introduce *variation* among the heights and
let some search find a non-flat profile that suppresses that peak. The question is how to search, and at what
scale.

Scale first, because it decides everything else. I am tempted to go straight to a large piece count, but that
is a trap at this stage. The functional is highly non-convex — many local optima, lots of symmetry to break —
and a long height vector is a high-dimensional search space where a blind local search wanders for a very long
time before finding anything good. The smart move is to find the *shape* at low resolution first, where the
vector is short enough to explore thoroughly, and only later lift that shape to finer grids. The literature
points exactly here: the published constructions are seeded from a small optimized profile and then refined on
fine grids. So I will work at `N` around `50`: short enough that a stochastic search can canvas the shape space,
long enough that the autoconvolution has real internal structure to exploit and that I can already read off
which direction the gains come from.

Now the search itself. The objective has a feature that rules out naive hill-climbing: the flat profile, and
indeed most "obvious" profiles, sit at local optima or near-flat regions where any small perturbation *raises*
the ratio before a coordinated reshape lowers it. A greedy descender that only accepts improving moves will
park itself at the first such point and stop. To cross those ridges I have to be willing to accept moves that
make the ratio temporarily *worse* — which is exactly what simulated annealing does. Propose a perturbation to
one height; if it lowers `R`, take it; if it raises `R`, take it anyway with a probability that depends on how
much worse and on a temperature I cool over the run. Hot early, so the search wanders freely and shakes loose
from the flat-triangle basin; cold late, so it settles into whatever better basin it has found. The whole bet
is that with enough wandering it discovers a profile whose autoconvolution has its peak pushed down relative to
the mass, well below `2`.

A few design decisions need care, and they come from the geometry of this particular objective. The first is
the perturbation. Heights are non-negative and I suspect the good profiles will span a wide dynamic range — not
a uniform spread but something with structure, perhaps heavier toward the ends and lighter in the middle, since
that is the kind of arrangement that reduces the central self-overlap. A single additive Gaussian kick of fixed
size would be far too coarse for the small heights and far too timid for the large ones. So I make the kick
*multiplicative in scale*: perturb a randomly chosen height by an amount proportional to its own magnitude, and
clip any negative result back to non-negative so the candidate stays legal. This lets the search adjust a tall
value and a thin one on comparable *relative* terms, which is the right invariance for a scale-free objective.

The second is what to anneal on. The objective sits in a bounded range above `1.28` and its changes under a
single height perturbation are small and well-scaled, so I do not need to take a log or rescale — I can anneal
directly on `R` with a temperature of order a few hundredths cooled geometrically, and the Metropolis acceptance
behaves sanely. I shrink the perturbation scale alongside the temperature, so that late in the run the search
is making fine adjustments to a settled shape rather than large jumps.

The third is the seeds, and here I should hedge because I do not yet know which way the optimal shape leans.
I will run several independent restarts from different initializations — the flat vector itself (so the anneal
can discover the descent direction from scratch), a monotone ramp, a smooth single hump, and a Gaussian bump —
and keep the best profile any restart ever reaches. The ramp and the bump are educated guesses: if the gains
come from breaking the left-right symmetry of the flat indicator, an asymmetric or concentrated seed should
land in the good basin faster than starting flat.

There is a real risk that annealing alone, perturbing one height at a time, leaves the profile rough and stops
short of the local optimum its basin contains. So after the anneal settles a shape I want a gradient polish to
slide it to the bottom of that basin. The obstruction is that `R` involves a `max` over the autoconvolution
nodes, which is not differentiable at the peak. I will handle it the standard way: replace the hard `max` by a
smooth softmax with a sharpness parameter `β`, making the objective differentiable, and anneal `β` upward —
starting soft, where the surrogate is smooth and the gradient points broadly toward better shapes, and ending
sharp, where the surrogate faithfully tracks the true peak. The gradient factors cleanly through the
autoconvolution by the chain rule, and everything is `O(N log N)` with FFTs, so the polish is cheap. I run
Adam on top of this gradient — its per-coordinate adaptive scaling suits the wide dynamic range of the heights
— clip to non-negative after each step, and track the best *true* `R` ever seen, not the surrogate's.

One thing I want to watch is whether the optimizer drives some heights to *zero* or concentrates mass at the
ends. That would be a real signal, not noise: it would mean the best coarse profile is genuinely asymmetric and
sparse — a specific structure, not an even spread — which is what I expect from a minimization that punishes
central self-overlap.

What do I expect? The annealing-plus-polish should clear the flat-triangle ceiling easily — that is the entire
reason to accept uphill moves — and drop well into the low `1.5`s, the band where coarse constructions for this
constant live. I do not expect to *reach* the `1.5053` of the `600`-piece AlphaEvolve construction, let alone
the `1.50286` record, from a short run on `50` heights; those numbers are the product of careful optimization on
far finer grids, and `50` pieces is a coarse grid that caps how low the peak can be pushed. So I expect this
rung to prove the principle — that breaking the flat symmetry buys a large drop, from `2.0` down past `1.55` —
and then to stall, limited not by the search idea but by the coarse resolution. That stall is the opening for the
next rung. If the only thing capping me is that `50` pieces cannot render a sufficiently fine, peak-suppressing
autoconvolution, then the move is to *lift* this optimized coarse shape onto a much finer grid and bring a much
stronger optimizer — one that attacks the whole flat top of near-peak nodes at once — to carve the fine
structure that annealing on `50` heights cannot represent.
