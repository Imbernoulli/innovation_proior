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

Let me decompose SPSA's table the same way I decomposed the floor's, because the residual structure tells
me where the next method has to bite. By architecture: ResNet20 `(0.955 + 0.910)/2 = 0.933`, MobileNetV2
`(0.920 + 0.885)/2 = 0.903`, VGG11-BN `(0.605 + 0.620)/2 = 0.613`. VGG is *still* the outlier, now at
`0.613` against `~0.92` for the other two — a ratio `0.613/0.933 = 0.66`, better than the floor's `0.47`
but the same backbone is still the wall. And a sign flip worth noticing: by dataset, SPSA scores CIFAR-10
`(0.955 + 0.605 + 0.920)/3 = 0.827` against CIFAR-100 `(0.910 + 0.620 + 0.885)/3 = 0.805` — CIFAR-100 is now
slightly *lower*, whereas for the floor it was slightly higher. That inversion is mechanistically sensible:
the floor benefited from CIFAR-100's extra competitors because it was flailing for *any* flip, but once I
descend a real margin the "more targets" bonus evaporates (aiming already finds the nearest boundary) and
the more crowded 100-way logit geometry costs a hair. The gap is small (`0.022`), so dataset is second
order; architecture is first order, and the number to beat is the VGG pair near `0.61`. The improvement
SPSA bought there — `0.255 -> 0.605` (`2.37x`) and `0.300 -> 0.620` (`2.07x`) — proves the VGG failure was
directional, but the fact it *stalled* at `~0.61` while ResNet went to `~0.93` on the same three steps says
the VGG boundary needs *more corrections than three*, not a cleaner single direction. So the next rung is
not about a better direction; it is about buying an order of magnitude more steps out of the same budget.

So the question for this rung is sharp: can I get many cheap, *effective* moves instead of a few
expensive ones? SPSA pays `256` queries to estimate one gradient and then takes one step on it. What if a
single query bought me a whole candidate move that I either keep or discard? That is one query per *step*,
not `2n` queries per step — a 256x improvement in step count at fixed budget, *if* the per-step move is
good enough to make progress. The floor already used one-query accept-if-better — and failed, because its
moves were structureless. So the entire game is: design a one-query proposal whose moves are *structured*
enough that greedy accept-if-better converges in tens of steps. SPSA's strength was the direction; the
floor's strength was the cheap step. I want both: cheap steps *and* moves aligned with the model.

Before committing, let me price the one-query alternatives the background offers, because "one query per
step" is a family, not a single method. Option one: keep the floor's uniform proposal but fix its two
economic sins — move to `+/- eps` corners and run the full `~999` steps instead of `64`. But the floor
already proved the ceiling here: a uniform corner direction over `3072` coordinates is still `~1.4%`
aligned, and `999` of those is still a random walk; more cheap steps without structure does not clear the
VGG wall. Option two: SimBA-style orthonormal search — add or subtract a fixed small step along one basis
vector (a pixel or a DCT atom) per query. That is structured and cheap, but each move is a *small*
orthogonal step that can never be undone, so the perturbation is a monotone accumulation and I cannot
re-spend budget on a region I later realize was wrong; and single-basis moves touch one direction at a
time, throwing away the per-query leverage of moving a whole block. Option three: fixed-grid corner search
— restrict to `+/- eps` on a pre-defined discrete grid of blocks. Corners are right, but freezing the grid
throws away the freedom to *choose where* to spend the next move, which is exactly the freedom I most want
on a hard boundary. The synthesis these three point at is: corners (from option three), a block move that
re-spends freely (fixing option two's irreversibility), and a freely *sampled* location (fixing option
three's frozen grid) — greedy accept-if-better over that proposal. That is where I am headed, and the rest
of this reasoning is deriving the block's shape and sign from the model.

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
covered windows at once. Let me make this concrete with a `3 x 3` first-layer filter (`s = 3`). A `2 x 2`
block of changed pixels fully covers *zero* `3 x 3` windows — a `3 x 3` window does not fit inside a
`2 x 2` support at all — so it moves no first-layer activation to its full extent; a `3 x 3` block covers
exactly `1`. Now spend a real budget, `k = 64` pixels, three ways: an `8 x 8` square covers
`(8 - 3 + 1)^2 = 36` full windows; a `4 x 16` rectangle covers `(4 - 3 + 1)(16 - 3 + 1) = 2 * 14 = 28`; a
`2 x 32` strip covers `0`, because two rows can never contain a three-row window. Same `64`-pixel budget,
and the square saturates `36` first-layer units, the rectangle `28`, the strip *none*. Carrying this
through, the optimum for area `k` is the near-square rectangle; for `k = l^2` it is a literal `l x l`
square. So the convolutional structure forces the update's support to be
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
`E|<delta, v>| = Theta(||v_block||_1)`. Put the scaling on an `h x h` block with roughly unit-magnitude
`v` entries: `||v_block||_1 ~ h^2` while `||v_block||_2 ~ sqrt(h^2) = h`, so the shared sign aligns at the
`h^2` scale and the independent signs at the `h` scale — a factor of `h` better. For the `8 x 8` square
above, `h = 8`, so the shared sign is about `8x` better aligned with a locally-constant gradient than an
independent-sign fill of the exact same square; the two design choices compound, because the square is
what makes the local gradient approximately constant across the support in the first place. So I share one
sign across the whole square *within* each channel, but keep separate signs *per channel* (different
first-layer color filters can want different channel directions, and the implementation keeps that freedom
for free). The update is a square block, all spatial entries within a channel at `+/-2eps`, one sign per
channel, at a uniformly random location.

How big is the square, and does it stay fixed? Let `p in [0,1]` be the fraction of spatial pixels I touch
this step; the side is `s = round(sqrt(p * n_features / c))`, clamped to >= 1. Early in the search I am
far from a solution and want big, coarse moves that can change the prediction outright — large `p`. As I
close in, large squares overshoot and are more likely to be rejected, wasting a query. So `p` shrinks over
the budget, the direct analogue of step-size decay (the thing Adam gave SPSA, here built into the move
size). Put the schedule in pixels for CIFAR, where `n_features / c = H * W = 1024`, so `s =
round(sqrt(p * 1024))`. Starting at `p_init = 0.8` gives `s = round(sqrt(819.2)) = 29` — a `29 x 29` square
that covers almost the entire `32 x 32` image, a genuinely coarse "repaint most of it" opening move. Then
`p` halves: `0.4 -> sqrt(409.6) = 20`, `0.2 -> 14`, `0.1 -> 10`, `0.05 -> 7`, `0.025 -> 5`, so the side
walks `29 -> 20 -> 14 -> 10 -> 7 -> 5 -> ...` from near-global repaints down to tiny local touch-ups — the
step-size decay Adam gave SPSA, here realized as a shrinking support. The `resc_schedule = True` maps
`it -> int(it / n_queries * 10000)`, so the breakpoints that were tuned on a `10000`-query schedule are
compressed by exactly `10x` onto this `1000`-query budget — the same coarse-to-fine arc, ten times faster,
which matters because at one query per step I will take on the order of hundreds of steps and want the
whole arc to fit. And the
initialization: rather than start at the clean image and spend the first moves finding the boundary, start
*already* on the boundary at a structured high-frequency point CNNs are known to be sensitive to — random
width-1 vertical stripes at full `L_inf` radius. Two reasons make this the right free move. First,
economics: I earlier watched the floor spend `~32` accepted steps just accumulating to the `eps` radius; a
boundary init hands me that radius for free at query zero, so none of my precious one-query steps are
wasted merely reaching full magnitude. Second, direction: width-1 vertical stripes are a maximally
high-frequency pattern along the horizontal axis, and a convolutional first layer with small filters is
exactly a high-frequency-sensitive operator — its filters respond strongly to sharp local transitions — so
the stripe init already sits near a direction the model's early layers amplify, a better starting basin
than the clean image or uniform noise. It costs the same one initial query the floor spent seeding `best`,
but it spends it on the boundary in a model-relevant direction instead of at the timid clean point.

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
boundaries) *and* lower `avg_queries` (cheap steps plus early exit). Put the step arithmetic beside SPSA's:
SPSA got `1000 // 256 = 3` corrections; Square, at one query each, can make up to `~999`, a factor of
`~333x` more corrections on the same budget. The VGG rows stalled at `~0.61` on three steps precisely
because three was too few; if the boundary needs, say, `50` structured corrections, Square can afford it
and SPSA never could.

Let me verify the square construction against its endpoints rather than trust the derivation. At the coarse
end, `p -> 1` and a shared per-channel sign, the move is essentially a single global `+/-2eps` shift of an
entire channel that reprojects to a uniform `+/-eps` tint — a one-parameter, `2^3 = 8`-way corner probe of
the global colour direction, exactly the kind of blunt "is the whole image the wrong tint" test a coarse
opening should make. At the fine end, `s = 1`, the square collapses to a single pixel driven to `+/- eps`
— which is precisely a SimBA-style single-coordinate corner flip. So the coarse-to-fine schedule literally
interpolates between a global corner probe and single-pixel corner search, with the `l x l` square as the
general member: the two one-query methods I rejected as endpoints sit *inside* this family, and the square
schedule sweeps through them. That is the consistency check I wanted — the design is a strict
generalization, not a lateral move. The force-accept-on-flip clause cannot break feasibility because the
force-accepted candidate is still a projected `+/-eps` point, and the early-exit (stop querying once
`margin <= 0`) is what guarantees `avg_queries` falls below the budget rather than pinning at it: any
sample that flips at step `t` contributes `t + 1` queries, not `1000`, so easy samples drag the mean down.

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
