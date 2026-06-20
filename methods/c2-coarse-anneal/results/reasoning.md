The flat profile sat exactly at `2/3`, and the feedback made the obstruction concrete: with every piece
identical the autoconvolution is locked to a triangle and there is no gradient to follow — refining the
grid does nothing. So the only way forward is to introduce *variation* among the heights and let some
search find a non-flat profile that bends the autoconvolution away from the tent. The question is how to
search, and at what scale.

Scale first, because it decides everything else. I am tempted to go straight to a large piece count, but
that is a trap at this stage. The functional is highly non-convex — many local optima, lots of symmetry to
break — and a long height vector is a high-dimensional search space where a blind local search wanders for
a very long time before finding anything good. The smart move is to find the *shape* at low resolution
first, where the vector is short enough to explore thoroughly, and only later lift that shape to finer
grids. The literature points exactly here: Matolcsi and Vinuesa got past `0.88` with only `20` steps, and
later works seeded their large-`N` constructions from a small optimized profile of this kind. So I will
work at `N` around `20`: short enough that a stochastic search can canvas the shape space, long enough that
the autoconvolution has real internal structure to exploit.

Now the search itself. The objective has a feature that rules out naive hill-climbing: the flat profile,
and indeed most "obvious" profiles, are local optima or near-flat regions where any small perturbation
*lowers* the ratio before a coordinated reshape raises it. A greedy climber that only accepts improving
moves will park itself at the first such point and stop. To cross those ridges I have to be willing to
accept moves that make the ratio temporarily *worse* — which is exactly what simulated annealing does.
Propose a perturbation to one height; if it improves `R`, take it; if it worsens `R`, take it anyway with a
probability that depends on how much worse and on a temperature that I cool over the run. Hot early, so the
search wanders freely and shakes loose from the flat-triangle basin; cold late, so it settles into whatever
better basin it has found. The whole bet is that with enough wandering it discovers a profile whose
autoconvolution has a flat cap and steep sides, well above `2/3`.

A few design decisions need care, and they come from the geometry of this particular objective. The first
is the perturbation. Heights are non-negative and span a wide dynamic range — the good profiles, I suspect,
will have a tall spike or two and a long shoulder of smaller values, not a uniform spread. A single
additive Gaussian kick of fixed size would be far too coarse for the small heights and far too timid for
the large ones. So I make the kick *multiplicative in scale*: perturb a randomly chosen height by an amount
proportional to its own magnitude plus a small floor, and reflect any negative result back to be
non-negative so the candidate stays legal. This lets the search adjust a tall spike and a thin shoulder
value on comparable *relative* terms, which is the right invariance for a scale-free objective.

The second is what to anneal on. Here the objective is already bounded in `[0,1]` and its changes under a
single height perturbation are small and well-scaled, so unlike some problems I do not need to take a log or
rescale — I can anneal directly on `R` with a temperature of order `0.01` down to a tiny floor, and the
Metropolis acceptance behaves sanely. I cool geometrically and shrink the perturbation scale alongside the
temperature, so that late in the run the search is making fine adjustments to a settled shape rather than
large jumps.

The third is restarts. A single annealing run from one random seed can get trapped; the landscape has many
basins of differing quality. So I run several independent restarts from different initializations — some
from a smooth single-bump seed (a Gaussian-shaped profile, which I expect to be in the right neighborhood
because the good autoconvolutions are unimodal), some from pure random heights — and keep the best profile
any restart ever reaches. The single-bump seed is the educated guess: if the autoconvolution wants a flat
cap, the height profile that produces it is likely concentrated, so starting concentrated should help the
search land in the good basin faster.

One thing I want to watch is whether the optimizer drives some heights to *zero*. That would be a real
signal, not noise: it would mean the best `20`-piece profile does not actually use all `20` pieces
uniformly — it wants a specific sparse support, a spike plus a shaped shoulder, with gaps. If that happens
it tells me the shape is genuinely non-trivial and that the gains over `2/3` come from a particular
asymmetric structure, not from spreading mass evenly.

What do I expect? The annealing should clear the flat-triangle floor easily — that is the entire reason to
accept downhill moves — and climb into the high `0.88`s, the band where the known `20`-step constructions
live. I do not expect to *beat* the published `0.88922` by much, if at all, from a short run on `20`
heights; that number is itself the product of careful optimization, and `20` pieces is a coarse grid that
caps how flat the autoconvolution's cap can get. So I expect this rung to prove the principle — that
breaking the flat symmetry by annealing the heights buys a large jump, from `0.6667` into the high `0.88`s —
and then to stall, limited not by the search idea but by the coarse resolution. That stall is the opening
for the next rung. If the only thing capping me is that `20` pieces cannot render a sufficiently flat-topped
autoconvolution, then the move is to *lift* this optimized coarse shape onto a much finer grid and let a
gradient-based refinement carve the fine structure that annealing on `20` heights cannot represent.
