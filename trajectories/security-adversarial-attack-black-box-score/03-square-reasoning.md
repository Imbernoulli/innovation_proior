SPSA cleared the floor where I predicted and stalled where I feared, and reading the two failures together
tells me what the next rung has to be. The mean jumped from 0.474 to 0.816 — a real descent direction,
estimated from the queries the floor was wasting, beat blind noise everywhere. The VGG11-BN pairs, the
floor's collapse points, roughly doubled: 0.255 -> 0.605 on CIFAR-10, 0.300 -> 0.620 on CIFAR-100. So the
floor's VGG failure *was* directional. And the ResNet20 pairs saturated toward the low-0.9s, 0.955 and
0.910. But two things are wrong with SPSA, from one root. First, the regression I warned about: every row
reads `avg_queries = 768`, because one descent step costs `2 * nb_sample = 256` queries and SPSA gets only
three steps for the whole budget. Second, the more telling number: the VGG pairs only reached ~0.61 and
trail their ResNet siblings. Three noisy steps is simply not enough on the harder boundaries.

Decompose SPSA's table the way I did the floor's. By architecture: ResNet20 `0.933`, MobileNetV2 `0.903`,
VGG11-BN `(0.605 + 0.620)/2 = 0.613`. VGG is *still* the outlier, now at `0.613` against `~0.92` — ratio
`0.66`, better than the floor's `0.47` but the same backbone is still the wall. And a sign flip: by dataset,
CIFAR-10 `0.827` against CIFAR-100 `0.805` — CIFAR-100 is now slightly *lower*, whereas for the floor it was
higher. That inversion is mechanistically sensible: the floor benefited from CIFAR-100's extra competitors
because it was flailing for *any* flip, but once I descend a real margin the "more targets" bonus evaporates
(aiming already finds the nearest boundary) and the more crowded 100-way logit geometry costs a hair. The
gap is small (`0.022`), so dataset is second order; architecture is first order, and the number to beat is
the VGG pair near `0.61`. The improvement SPSA bought there (`2.37x` and `2.07x`) proves the failure was
directional, but the fact it *stalled* at `~0.61` while ResNet went to `~0.93` on the same three steps says
the VGG boundary needs *more corrections than three*, not a cleaner single direction. So the next rung is
not about a better direction; it is about buying an order of magnitude more steps out of the same budget.

The question is sharp: can I get many cheap, *effective* moves instead of a few expensive ones? SPSA pays
256 queries to estimate one gradient and takes one step on it. What if a single query bought me a whole
candidate move I either keep or discard? That is one query per *step*, not `2n` — a 256x improvement in step
count at fixed budget, *if* the per-step move is good enough. The floor already used one-query
accept-if-better and failed, because its moves were structureless. So the game is: design a one-query
proposal whose moves are *structured* enough that greedy accept-if-better converges in tens of steps. SPSA's
strength was the direction, the floor's was the cheap step; I want both.

Before committing, price the one-query alternatives the background offers. Keep the floor's uniform proposal
but move to `+/- eps` corners and run the full ~999 steps: the floor already proved the ceiling — a uniform
corner direction over 3072 coordinates is still `~1.4%` aligned, and 999 of those is still a random walk.
SimBA-style orthonormal search adds or subtracts a fixed step along one basis vector per query: structured
and cheap, but each move is a small orthogonal step that can never be undone, so the perturbation is a
monotone accumulation I cannot re-spend on a region I later realize was wrong, and single-basis moves throw
away the leverage of moving a whole block. Fixed-grid corner search restricts to `+/- eps` on a pre-defined
grid: corners are right, but freezing the grid throws away the freedom to *choose where* to spend the next
move — exactly the freedom I most want on a hard boundary. The synthesis is corners (from the grid method),
a block move that re-spends freely (fixing SimBA's irreversibility), and a freely *sampled* location (fixing
the frozen grid) — greedy accept-if-better over that proposal.

There is a deeper reason to abandon gradient estimation entirely, beyond the query tax. SPSA, like any
finite-difference method, follows the *local gradient*. A large class of defenses do not remove adversarial
examples — they wreck the gradient signal (shatter, randomize, flatten it), and a robustness probe fooled
into calling a masked model robust is worse than useless. So I want a method that never touches a gradient:
pure greedy hill-climbing on the score, one query per candidate, immune to masking by construction — but
with the proposal distribution carrying all the structure the floor lacked. The leverage is entirely in
*what distribution I draw the proposed update from*.

Start from the constraint. It is `L_inf`, a box, and successful `L_inf` perturbations almost always sit at a
*corner*, every component at `+/- eps` — if a component can move by up to `eps` and moving it helps, you
want it all the way out. This is exactly where the floor bled with its interior `eps/2` moves. So the first
rule: start on the boundary and stay there. If a component is at `x_i + eps` and I add `+2eps` then
re-project by clipping to the `L_inf` ball, it clips back to `x_i + eps`; adding `-2eps` clips to
`x_i - eps`. So updates of magnitude `2eps` followed by projection land every touched component on a corner
— full budget every step, and a later move can overwrite a region committed earlier.

Now *where* the nonzero entries go, and *how many*. A uniform scatter of `+/-2eps` is back to the floor's
hopeless high-dimensional random direction, so I have to use what the model actually is: a *convolutional*
network. Its first layer correlates small `s x s` patches against learned filters `w`. The change I induce
in a first-layer activation at `(u,v)` is `z_{u,v} = (delta * w)_{u,v}`, summed over the `s x s` window. I
do not know the weights, but I can bound the move: `|z_{u,v}| <= eps * sum |w| * 1[in support of delta]`,
maximal when the receptive window is *fully covered*. So for a fixed number `k` of changed pixels, I should
shape them to maximize the number of `s x s` windows fully covered. Make it concrete with a `3 x 3` filter
(`s = 3`) and `k = 64` pixels three ways: an `8 x 8` square covers `(8-3+1)^2 = 36` full windows, a `4 x 16`
rectangle covers `2 * 14 = 28`, a `2 x 32` strip covers `0` (two rows can never contain a three-row window).
Same budget, and the square saturates the most first-layer units. The optimum for area `k` is the
near-square rectangle; for `k = l^2` a literal `l x l` square. So the convolutional structure forces the
update's support to be a **square** — the shape that maximizes the worst-case change in first-layer
activations per pixel of budget. And I let the square's *position* be sampled freely each iteration, unlike
a frozen grid.

Now the sign pattern inside the square. The cheap default is an independent random sign per pixel and
channel, but a *spatially shared* sign should do better: greedy random search makes progress when the
proposed update correlates with the direction the loss responds to — call it `v` — and image gradients are
approximately piecewise constant, so neighboring pixels want to move the same way. Compare `E|<delta, v>|`
over one channel. Independent signs: by Khintchine, `E|<delta, v>| = Theta(||v_block||_2)` — the signs
partially cancel, random-walk style. One shared sign `rho`: `<delta, v> = rho * ||v_block||_1`, so
`E|<delta, v>| = Theta(||v_block||_1)`. On an `h x h` block with roughly unit-magnitude entries,
`||v_block||_1 ~ h^2` while `||v_block||_2 ~ h`, so the shared sign aligns at the `h^2` scale and
independent signs at the `h` scale — a factor of `h` better. For the `8 x 8` square, `h = 8`, so shared is
~`8x` better aligned; the two design choices compound, because the square is what makes the local gradient
approximately constant across the support in the first place. So I share one sign across the whole square
*within* each channel, but keep separate signs *per channel* (different first-layer color filters can want
different channel directions).

How big is the square, and does it stay fixed? Let `p in [0,1]` be the fraction of spatial pixels touched
this step; the side is `s = round(sqrt(p * n_features / c))`, clamped to `>= 1`. Early in the search I am
far from a solution and want big coarse moves that can change the prediction outright — large `p`. As I
close in, large squares overshoot and are more likely rejected, wasting a query. So `p` shrinks over the
budget, the direct analogue of step-size decay (what Adam gave SPSA, here built into the move size). In
pixels for CIFAR, `n_features / c = H * W = 1024`, so `s = round(sqrt(p * 1024))`. Starting at `p_init =
0.8` gives `s = round(sqrt(819.2)) = 29` — a `29 x 29` square covering almost the whole image, a coarse
"repaint most of it" opening. Then `p` halves: `0.4 -> 20`, `0.2 -> 14`, `0.1 -> 10`, `0.05 -> 7`,
`0.025 -> 5`, so the side walks from near-global repaints down to local touch-ups. `resc_schedule = True`
maps `it -> int(it / n_queries * 10000)`, compressing breakpoints tuned on a 10000-query schedule by `10x`
onto this 1000-query budget — the same coarse-to-fine arc ten times faster, which matters because at one
query per step I take hundreds of steps and want the whole arc to fit. And the initialization: rather than
start at the clean image and spend the first moves finding the boundary, start *already* on the boundary at
random width-1 vertical stripes at full `L_inf` radius. Two reasons. Economics: I earlier watched the floor
spend ~32 accepted steps just accumulating to the `eps` radius; a boundary init hands me that radius for
free at query zero. Direction: width-1 vertical stripes are a maximally high-frequency horizontal pattern,
and a convolutional first layer with small filters is a high-frequency-sensitive operator, so the stripe
init already sits near a direction the model's early layers amplify. It costs the same one initial query the
floor spent seeding `best`, but spends it on the boundary in a model-relevant direction.

The objective is the margin `J = f_y - max_{k!=y} f_k`, used both as the thing to minimize and the success
test — the same quantity SPSA descended, of which the floor's plain-`f_y` was a degenerate version.
`loss = "margin"` in the fill. The accept rule has one extra clause beyond accept-if-better: if a candidate
is already misclassified (`margin <= 0`) it is force-accepted regardless of the loss comparison, because
crossing the boundary is the actual goal and I do not want a marginally worse loss reading to throw away a
successful flip. Once a sample's margin hits zero I stop spending queries on it — the same early-exit the
floor and SPSA use, but now it matters far more because the steps are cheap and many samples flip in the
first dozen.

Now the economics, the axis on which SPSA bled. SPSA cost 256 queries per step and got 3. Square costs
**one** query per candidate. So with a 1000 budget it can take *hundreds* of structured steps: `1000 // 256
= 3` corrections for SPSA versus up to ~999 for Square, ~333x more on the same budget. The VGG rows stalled
at ~0.61 on three steps precisely because three was too few; if that boundary needs tens of structured
corrections, Square can afford them and SPSA never could. And because each step is one query and a sample
exits the instant it flips, the *average* queries over the batch is pulled far below the budget by the easy
tail — the opposite of SPSA, where every sample paid the full 768. The coarse-to-fine schedule also
generalizes the two one-query methods I rejected: at `p -> 1` with a shared channel sign the move is a
global `+/-eps` tint probe, and at `s = 1` the square collapses to a single-pixel SimBA-style corner flip,
so the schedule sweeps between them with the `l x l` square as the general member.

So the falsifiable expectations against SPSA's numbers. Square should beat SPSA on `asr` everywhere, because
corner-spending square moves aligned with the convolutional structure make real progress per query where
SPSA's three noisy steps ran out; the sharpest test is the VGG11-BN pairs at 0.605 and 0.620, where if the
residual failure was step-starvation, hundreds of cheap structured steps should lift them substantially. And
`avg_queries` should *invert* the SPSA regression: where SPSA read a flat 768 everywhere, Square should read
well below it, and varying by scenario difficulty rather than pinned flat, because steps are one query each
and flipped samples stop querying. If Square instead needs *more* queries than SPSA, or fails to clear it on
the VGG pairs, my diagnosis was wrong — the failure would be something the square proposal also cannot
reach. The distilled module calling `torchattacks.Square` is in the answer.
