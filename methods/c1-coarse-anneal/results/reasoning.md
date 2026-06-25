The flat profile sat exactly at `2.0`, and the feedback made the obstruction concrete: with every piece
identical the autoconvolution is locked to a triangle with one sharp central peak, and there is no gradient to
follow — refining the grid does nothing. So the only way down is to introduce *variation* among the heights and
let some search find a non-flat profile that suppresses that peak. Before I commit to a search, I want to see
the mechanism on the smallest case I can compute by hand, so I actually know what "suppressing the peak" buys.

Take `N=2`, `a=(a_0,a_1)`. The self-convolution is `a*a=(a_0^2,\,2a_0a_1,\,a_1^2)` and the score is
`R = 4·max(a_0^2,2a_0a_1,a_1^2)/(a_0+a_1)^2`. Flat `(1,1)` gives `b=(1,2,1)`, peak `2`, sum `2`, so
`R=4·2/4=2` — the central node wins and pins the ratio at `2`, exactly as advertised. Now break it: try
`(2,1)`. Then `b=(4,4,1)`, and the peak is now `4`, *tied* between the endpoint node `a_0^2=4` and the
cross node `2a_0a_1=4`, against sum `3`, giving `R=4·4/9=1.778`. That is below `2`, and I can see *why*:
making `a_0` heavier pulls the cross term `2a_0a_1` down relative to where a symmetric profile would put it,
so the dominant node stops being the central overlap and starts being a corner. The score that this
minimization punishes is precisely the central self-overlap, and asymmetry is the lever that relieves it. So
the direction is real, and I have a concrete witness `R=1.778<2` to confirm it rather than just assert it.

That tempts an obvious guess for the larger profile: if asymmetry helps, pile mass at the ends and hollow out
the middle — a U-shape — to starve the central node directly. Let me check it before I believe it. At `N=50`,
a U-shape `1+3|x-1/2|` (heavy at both ends, light in the middle) scores `R=2.125` — *worse* than flat. A pure
steep ramp `x` is worse still at `2.23`. So the naive "concentrate at the ends" reading is wrong: a symmetric
U just builds two big endpoint nodes whose self-overlap is its own large peak, and an over-steep ramp does the
same at one end. What actually worked at `N=2` was *moderate, one-sided* asymmetry, and that does lift: a
half-`2`/half-`1` step at `N=50` reproduces `R=1.778` to the digit, and the gentle ramp `0.5+x` gives `1.864`.
The lesson I take from these five evaluations is that the good profile is some *tuned* intermediate amount of
asymmetry, not an extreme one — too little (flat) and too much (U, steep ramp) both sit at or above `2`. I
cannot guess that sweet spot from intuition; I have to search for it.

The question is how to search, and at what scale. Scale first, because it decides everything else. I am
tempted to go straight to a large piece count, but that is a trap at this stage. The functional is highly
non-convex — many local optima, lots of symmetry to break — and a long height vector is a high-dimensional
search space where a blind local search wanders for a very long time before finding anything good. The smart
move is to find the *shape* at low resolution first, where the vector is short enough to explore thoroughly,
and only later lift that shape to finer grids. The published constructions point the same way: they are seeded
from a small optimized profile and then refined on fine grids. So I will work at `N` around `50`: short enough
that a stochastic search can canvas the shape space, long enough that the autoconvolution has real internal
structure to exploit and that I can already read off which direction the gains come from.

Now the search itself. The objective has a feature that rules out naive hill-climbing, and the experiments
above already exhibit it: flat sits at `2`, and the nearby "obvious" profiles I tried (U-shape, steep ramp)
sit *higher*, so any small perturbation away from a simple profile tends to *raise* the ratio before a
coordinated reshape lowers it. A greedy descender that only accepts improving moves will park itself at the
first such point and stop. To cross those ridges I have to be willing to accept moves that make the ratio
temporarily *worse* — which is what simulated annealing does. Propose a perturbation to one height; if it
lowers `R`, take it; if it raises `R`, take it anyway with a probability that depends on how much worse and on
a temperature I cool over the run. Hot early, so the search wanders freely and shakes loose from the
flat-triangle basin; cold late, so it settles into whatever better basin it has found. The bet is that with
enough wandering it discovers a profile whose autoconvolution has its peak pushed down relative to the mass.

A few design decisions need care, and they come from the geometry of this particular objective. The first is
the perturbation. Heights are non-negative and, from the hand example, the good profiles span a wide dynamic
range — a tall value next to a thin one, as in `(2,1)`, rather than a uniform spread. A single additive
Gaussian kick of fixed size would be far too coarse for the small heights and far too timid for the large
ones. So I make the kick *multiplicative in scale*: perturb a randomly chosen height by an amount proportional
to its own magnitude, and clip any negative result back to non-negative so the candidate stays legal. This
lets the search adjust a tall value and a thin one on comparable *relative* terms, which is the right
invariance for a scale-free objective.

The second is what to anneal on. The objective sits in a bounded range above `1.28` and its changes under a
single height perturbation are small and well-scaled — at `N=50` flipping one of fifty heights moves `R` by a
few hundredths at most — so I do not need to take a log or rescale; I can anneal directly on `R` with a
temperature of order a few hundredths cooled geometrically, and the Metropolis acceptance behaves sanely. I
shrink the perturbation scale alongside the temperature, so that late in the run the search is making fine
adjustments to a settled shape rather than large jumps.

The third is the seeds, and here I should hedge because I do not yet know which way the optimal shape leans —
the experiments told me asymmetry helps but not its exact form. I will run several independent restarts from
different initializations — the flat vector itself (so the anneal can discover the descent direction from
scratch), a monotone ramp, a smooth single hump, and a Gaussian bump — and keep the best profile any restart
ever reaches. The ramp and the bump are educated guesses grounded in the checks above: the ramp `0.5+x`
already reaches `1.864` on its own, so seeding the anneal from it starts it inside a sub-`2` basin rather than
on the flat ridge.

There is a real risk that annealing alone, perturbing one height at a time, leaves the profile rough and stops
short of the local optimum its basin contains. So after the anneal settles a shape I want a gradient polish to
slide it to the bottom of that basin. The obstruction is that `R` involves a `max` over the autoconvolution
nodes, which is not differentiable at the peak. I will handle it the standard way: replace the hard `max` by a
smooth softmax with a sharpness parameter `β`, making the objective differentiable, and anneal `β` upward —
starting soft, where the surrogate is smooth and the gradient points broadly toward better shapes, and ending
sharp, where the surrogate faithfully tracks the true peak. The gradient factors cleanly through the
autoconvolution by the chain rule, and everything is `O(N log N)` with FFTs, so the polish is cheap. I run
Adam on top of this gradient — its per-coordinate adaptive scaling suits the wide dynamic range of the
heights — clip to non-negative after each step, and track the best *true* `R` ever seen, not the surrogate's.

One thing I want to watch is whether the optimizer drives some heights to *zero* or concentrates mass
asymmetrically. That would be a real signal, not noise: it would mean the best coarse profile is genuinely
sparse and one-sided — a specific structure, not an even spread, and not the symmetric U-shape that failed
above — which is what the `N=2` mechanism predicts.

What do I expect? The annealing-plus-polish should clear the flat-triangle ceiling easily — that is the entire
reason to accept uphill moves — and drop well into the low `1.5`s, the band where coarse constructions for
this constant live. I do not expect to *reach* the `1.5053` of the `600`-piece construction, let alone the
`1.50286` record, from a short run on `50` heights; those numbers are the product of careful optimization on
far finer grids, and `50` pieces is a coarse grid that caps how low the peak can be pushed.

Running it bears this out. The construction settles at `R ≈ 1.537`, and the profile is exactly the structure I
was watching for: of the fifty heights, about ten are driven to zero, the mass is spread unevenly with no
left-right symmetry, and the dominant autoconvolution node sits off the geometric center. So the search did
find the tuned, sparse asymmetry that the hand example pointed at and that the symmetric U-shape could not
reach — a large drop, from `2.0` down to `1.537` — and then stalled, limited not by the search idea but by the
coarse resolution. That stall is the opening for the next rung. If the only thing capping me is that `50`
pieces cannot render a sufficiently fine, peak-suppressing autoconvolution, then the move is to *lift* this
optimized coarse shape onto a much finer grid and bring a much stronger optimizer — one that attacks the whole
flat top of near-peak nodes at once — to carve the fine structure that annealing on `50` heights cannot
represent.
