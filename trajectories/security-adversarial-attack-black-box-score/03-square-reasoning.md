SPSA cleared the floor exactly where I predicted and stalled exactly where I feared, and reading the two
failures together tells me what the next rung has to be. The mean jumped from 0.474 to 0.816 — a real
descent direction, estimated from the queries the floor was wasting, beat blind noise everywhere. The
VGG11-BN pairs, the floor's collapse points, roughly doubled: 0.255 -> 0.605 on CIFAR-10, 0.300 -> 0.620
on CIFAR-100. So the floor's VGG failure *was* directional, confirmed. And the ResNet20 pairs saturated
toward the low-0.9s, 0.955 and 0.910. But two things are wrong with SPSA, and both come from the same
root. First, the headline regression I warned about: every row reads `avg_queries = 768`. SPSA spends the
whole budget but gets only three Adam steps for it, because one descent step costs `2 * nb_sample = 256`
queries. Second — and this is the more telling number — the VGG pairs only reached ~0.61, and the
CIFAR-100 pairs trail their CIFAR-10 ResNet siblings. Three noisy steps is simply not enough on the
harder boundaries; the residual weak spots are precisely the rows where the per-step query tax left too
few corrections. The SPSA bargain — a genuine direction, but `O(n)` queries per step — caps the number of
steps, and on hard boundaries the number of steps is what I am short of.

So the question for this rung is sharp: can I get many cheap, *effective* moves instead of a few
expensive ones? SPSA pays `256` queries to estimate one gradient and then takes one step on it. What if a
single query bought me a whole candidate move that I either keep or discard? That is one query per *step*,
not `2n` queries per step — a 256x improvement in step count at fixed budget, *if* the per-step move is
good enough to make progress. The floor already used one-query accept-if-better — and failed, because its
moves were structureless. So the entire game is: design a one-query proposal whose moves are *structured*
enough that greedy accept-if-better converges in tens of steps. SPSA's strength was the direction; the
floor's strength was the cheap step. I want both: cheap steps *and* moves aligned with the model.

There is a deeper reason to abandon gradient estimation entirely, beyond the query tax. SPSA, like any
finite-difference method, estimates and follows the *local gradient*. A large class of defenses do not
remove adversarial examples — they wreck the gradient signal (shatter it, randomize it, flatten it). PGD
and SPSA fail on those for the same reason, and a robustness probe that is fooled into calling a masked
model robust is worse than useless. So I want a method that never touches a gradient at all. That points
straight back to random search — pure greedy hill-climbing on the score, one query per candidate, immune
to masking by construction — but with the proposal distribution carrying all the structure the floor
lacked. The leverage is entirely in *what distribution I draw the proposed update from*. So: what proposal
makes one-query greedy accept-if-better converge in tens to low-hundreds of queries instead of the floor's
hopeless interior noise?

Let me hunt for structure I can exploit, starting from the constraint. The constraint is `L_inf`,
`|x_adv_i - x_i| <= eps` componentwise — a box. And there is a well-documented empirical fact: successful
`L_inf` perturbations almost always sit at a *corner* of that box, every component at `+/- eps`. That makes
sense — if a component can move by up to `eps` and moving it helps, you want it all the way out, not
halfway. This is precisely where the floor bled: it used `step = eps/2` and made *interior* moves,
spending only half its per-component budget. So the first design rule is: start on the boundary and stay
there. If a component is at `x_i + eps` and I add `+2eps` then re-project by clipping to the `L_inf` ball,
it clips back to `x_i + eps`; adding `-2eps` clips to `x_i - eps`. So updates of magnitude `2eps` followed
by projection land every touched component back on a corner — full budget spent every step, and a later
move can overwrite a region committed earlier (unlike SimBA's orthogonal small steps, which can never be
undone).

Now *where* do I put the nonzero entries, and *how many*? A uniform scatter of `+/-2eps` over the image is
back to the floor's hopeless high-dimensional random direction. So I have to think about what the model
actually is — and here is the structure SPSA never used: it is a *convolutional* network. Its first layer
correlates small `s x s` patches of the input against learned filters `w`. The change I induce in a
first-layer activation at position `(u,v)` is `z_{u,v} = (delta * w)_{u,v}`, summed over the `s x s`
receptive window. I do not know the filter weights — they are inside the black box — but I can bound how
much I could possibly move that activation: `|z_{u,v}| <= eps * sum_{i,j} |w_{i,j}| * 1[index in support
of delta]`. The bound is maximal when the receptive window is *fully covered* by my nonzero entries. So
for a fixed number `k` of changed pixels, I should shape them to maximize the number of `s x s` windows
fully covered. Build the shape greedily, one cell at a time, tracking the count `N` of fully-covered
`s x s` squares: from an `s x s` block (`N = 1`), extending as a long thin strip spends ~`s` cells per new
covered window, while keeping the shape near-square and adding a strip along the longer side creates many
covered windows at once. Carrying this through, the optimum for area `k` is the near-square rectangle; for
`k = l^2` it is a literal `l x l` square. So the convolutional structure forces the update's support to be
a **square** — the shape that maximizes the worst-case change in first-layer activations per pixel of
budget. And unlike fixed-grid corner-search attacks, I let the square's *position* be sampled freely
anywhere each iteration; freezing a grid throws away the freedom to choose where to spend budget.

Now the sign pattern inside the square. The cheap default is an independent random sign per pixel and
channel. But let me check whether a *spatially shared* sign does better, because greedy random search
makes progress when the proposed update correlates with the direction the loss responds to — which
behaves like a gradient `v` — and image gradients are approximately piecewise constant, so neighboring
pixels want to move the same way. Compare `E|<delta, v>|` over one channel of the square. Independent
signs: `<delta, v>` is a sum of independent signed terms, and by Khintchine its expected magnitude is
`Theta(||v_block||_2)` — the signs partially cancel, random-walk style. One shared sign `rho` across the
square: `<delta, v> = rho * sum_block v = rho * ||v_block||_1` for a constant-sign block, so
`E|<delta, v>| = Theta(||v_block||_1)`. And `||v_block||_1 >> ||v_block||_2` for a constant block (`h^2`
vs `h`), an entire factor of `h` better alignment. So I share one sign across the whole square *within*
each channel, but keep separate signs *per channel* (different first-layer color filters can want
different channel directions, and the implementation keeps that freedom for free). The update is a square
block, all spatial entries within a channel at `+/-2eps`, one sign per channel, at a uniformly random
location.

How big is the square, and does it stay fixed? Let `p in [0,1]` be the fraction of spatial pixels I touch
this step; the side is `s = round(sqrt(p * n_features / c))`, clamped to >= 1. Early in the search I am
far from a solution and want big, coarse moves that can change the prediction outright — large `p`. As I
close in, large squares overshoot and are more likely to be rejected, wasting a query. So `p` shrinks over
the budget, the direct analogue of step-size decay (the thing Adam gave SPSA, here built into the move
size). The fill starts at `p_init = 0.8` and halves `p` at fixed iteration breakpoints, with
`resc_schedule = True` rescaling the schedule to the actual budget by mapping `it -> int(it / n_queries *
10000)` so the same coarse-to-fine shape stretches to whatever `n_queries` is given. And the
initialization: rather than start at the clean image and spend the first moves finding the boundary, start
*already* on the boundary at a structured high-frequency point CNNs are known to be sensitive to — random
width-1 vertical stripes at full `L_inf` radius.

The objective is the margin `J = f_y - max_{k!=y} f_k`, used both as the thing to minimize and as the
success test (it is the same quantity SPSA descended, and the floor's plain-`f_y` choice is a degenerate
version of it). `loss = "margin"` in the fill. The accept rule has one extra clause beyond plain accept-
if-better: if a candidate is already misclassified (`margin <= 0`) it is force-accepted regardless of the
loss comparison, because crossing the boundary is the actual goal and I do not want a marginally worse
loss reading to throw away a successful flip. Once a sample's margin hits zero I stop spending queries on
it — exactly the early-exit the floor and SPSA also use, but now it matters far more because the steps are
cheap and many samples flip in the first dozen.

Now the economics, the axis on which SPSA bled. SPSA cost `256` queries per step and got 3 steps. Square
costs **one** query per candidate (one initial query of the stripe init, then one per iteration on the
still-unfooled samples). So with a 1000 budget Square can take *hundreds* of structured steps where SPSA
took three. But the more important number is `avg_queries`: because each step is one query and a sample
exits the instant it flips, the *average* queries over the batch will be far below the budget — easy
samples flip in a handful of moves and stop consuming queries, so the mean is pulled down toward the easy
tail. This is the opposite of SPSA, where every sample paid the full `768` regardless. So I expect Square
to win on *both* axes simultaneously, which is rare: higher `asr` (more, better-aligned steps on the hard
boundaries) *and* lower `avg_queries` (cheap steps plus early exit).

So the falsifiable expectations against SPSA's numbers. Square should beat SPSA on `asr` everywhere,
because corner-spending square moves aligned with the convolutional structure make real progress per
query where SPSA's three noisy steps ran out. I expect the mean to climb from 0.816 toward the mid-0.9s.
The two scenarios where SPSA was weakest — the VGG11-BN pairs at 0.605 and 0.620 — are the sharpest test:
if SPSA's residual failure there was step-starvation, hundreds of cheap structured steps should lift both
into the high-0.8s / low-0.9s. The ResNet20 pairs, already 0.955 and 0.910, should push toward
near-saturation, ~0.99. And the `avg_queries` should *invert* the SPSA regression: where SPSA read a flat
768 on every row, Square should read well below it — on the order of ~70-260 depending on how hard the
scenario is — because steps are one query each and flipped samples stop querying. If Square instead needs
*more* queries than SPSA, or fails to clear it on the VGG pairs, my diagnosis was wrong — the failure
would not be step-starvation but something the square proposal also cannot reach. But I expect the corner +
square + shared-sign + coarse-to-fine combination to dominate on both metrics, which is exactly why this is
the strongest rung. The distilled module — the literal scaffold edit calling `torchattacks.Square` — is in
the answer.
