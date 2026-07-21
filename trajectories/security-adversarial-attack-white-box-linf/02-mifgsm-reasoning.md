The single sign step landed where the linearization argument said it would, and the shape of the
result is the whole story. FGSM scored 0.825 / 0.839 / 0.825 on ResNet20-C10 / ResNet20-C100 /
MobileNetV2-C100, 0.878 on MobileNetV2-C10 — already a large majority flipped on the bottlenecked and
depthwise architectures — but only 0.617 and 0.621 on VGG11BN-C10 and VGG11BN-C100. That is the split
I expected, and it is diagnostic, not noise. One signed jump of size `eps` in every coordinate
maximizes the loss's *tangent plane* at the clean point, and over a full-budget jump the tangent plane
stops describing the true loss surface. Wherever the loss bends fast away from its linearization, the
corner FGSM lands on is not where the loss is actually high — and VGG11-BN, whose wider, shallower
low-resolution feature maps absorb small per-pixel perturbations more robustly, is exactly where that
bending costs the most. So roughly 38% of VGG samples survived not because they are safe but because
one linearized step could not reach them. The fix is forced by the diagnosis: stop trusting one
linearization across the whole `eps`-box, and re-read the slope as I move.

Let me read the survivor counts off the feedback exactly, because the *ratio* is what tells me where
the headroom lives. FGSM's ASR is the flipped fraction, so `1 − ASR` is the survivors: ResNet20 leaves
`0.175` (C10) and `0.161` (C100), MobileNetV2 leaves `0.122` (C10) and `0.175` (C100), while VGG11-BN
leaves `0.383` and `0.379`. Average the non-VGG survivors, `(0.175 + 0.161 + 0.122 + 0.175)/4 = 0.158`,
and compare to the VGG average `(0.383 + 0.379)/2 = 0.381`: VGG carries `0.381/0.158 = 2.4×` the
survivor mass of the other architectures. The mean ASR is `0.768`, and essentially the entire deficit
from `1.0` is those two VGG columns plus a thin residual on the easy ones. So the diagnosis is not a
vague "VGG is harder" — it is that a single linearized step under-reaches on a concentrated,
identifiable `~2.4×`-denser pool of samples, and any method that spends more gradient reads should cash
out overwhelmingly on VGG and barely move the already-saturated columns. That asymmetry is the ruler I
will hold the next method against.

I cannot trust a single tangent plane over a `2eps`-wide box, so I take a *small* step in the sign
direction, re-evaluate the gradient where I land, step again, and repeat. With step size `alpha`,
`x_{t+1} = x_t + alpha·sign(∇_x J(θ, x_t, y))`, projecting back into the budget each step. Each step is
still the `L_inf`-steepest move — sign of the gradient, for exactly the Hölder reason FGSM used — but
now read from the *current* point, so I follow the curved surface instead of betting everything on the
plane at `x`. This iterative sign-ascent is what I'd expect to recover the VGG headroom: the samples
that survived FGSM are the ones where the surface curves, and that is precisely what extra
re-linearized steps correct. But before I just run iterative FGSM, I want to ask whether iterating
*greedily* on the raw gradient is the best use of those steps, because there is a known pathology in
greedy gradient ascent that I can pre-empt cheaply, and the harness lets me.

Here is the pathology. The classifier's loss surface is curved and bumpy, so the raw input-gradient
jitters from step to step as the iterate moves across it. A greedy sign-ascent that chases whatever the
local slope says right now zig-zags, wastes steps oscillating instead of climbing, and can pin the
iterate in a shallow sharp maximum well short of the high-loss region a smoother trajectory would
reach. So I want the iterative scheme — it recovers FGSM's underfit — but with the per-step direction
stabilized so it settles along the component of the gradient that keeps pointing the same way and coasts
through the small humps rather than getting caught in them.

The cure is the oldest one for zig-zagging ascent: momentum. Step not on the raw gradient but on an
accumulated velocity `g_{t+1} = μ·g_t + (current gradient)`. This is a low-pass filter on the gradient
sequence — a coordinate whose sign oscillates `+,−,+,−` sums toward zero and contributes little, while a
coordinate that points the same way every step accumulates and dominates; the velocity keeps the DC
component and discards the alternating part, and coasts the iterate over small humps. One honest caveat
I carry forward: in the classic telling momentum's headline payoff is *transfer* — the averaged
direction keeps the component shared across models — and that is not on the table here, since I attack
and am graded by one architecture. But the *optimization* half is model-agnostic: even on a single bumpy
surface, averaging directions cancels the step-to-step sign flips. So I keep momentum for the
optimization reason, not the transfer one — a distinction that will matter when I set the step size and
the start, both tuned for a strong single-model climb.

Two details decide the method. The step is forced: the `L_inf`-optimal move for direction `g_{t+1}` is
the sign, so `x_{t+1} = x_t + alpha·sign(g_{t+1})`, same reason FGSM used it, with explicit per-step
budget control. The accumulation needs care, because the naive `g_{t+1} = μ·g_t + ∇_x J(x_t, y)` has a
problem: the input-gradient's magnitude is not stable across iterations — large far from a boundary,
small near one — so a single big-magnitude step dominates the running sum and the "average" becomes
whichever step had the largest gradient, the very magnitude-noise I am trying to smooth out. Momentum
must be an average of *directions*, every step getting a fair vote, so I normalize before accumulating:
`g_{t+1} = μ·g_t + ∇_x J(x_t, y) / ||∇_x J(x_t, y)||_1` (in code, dividing by the mean absolute value
over pixels, proportional to the per-sample `L_1` norm). A two-step trace shows the failure is real, not
cosmetic: with `g_1 = (100, −100)` (mass 200) and then `g_2 = (1, 3)` (mass 4, disagreeing about
coordinate two), raw accumulation gives `(101, −97)`, `sign = (+,−)` — the stale `g_1` drowns `g_2` and
the step still drives coordinate two down. Normalize first: `(0.5,−0.5) + (0.25,0.75) = (0.75, 0.25)`,
`sign = (+,+)` — the unit-mass vote lets the newer direction overturn it, which is what an average of
directions should do. Since I take `sign(g_{t+1})` downstream, only the relative-magnitude pattern
across coordinates survives, and the normalization keeps old and new votes comparable.

I set the decay `μ = 1`: `g_{t+1} = g_t + grad/||grad||_1` sums all past unit-normalized directions
with equal weight. The usual `μ = 0.9` discounts old gradients geometrically, which is right when early
iterates sit in a now-irrelevant region — but here the whole trajectory lives inside one `2·eps`-wide
box, every point within `0.016` of every other per coordinate, so the "old" gradients are votes from a
neighborhood as relevant as the current one and should count equally. Undiscounted summation cannot blow
up, since each addend is unit-mass and the downstream sign discards the running sum's scale. The family
contains both ancestors as limits: at `μ = 0` the normalization is annihilated by the sign
(`sign(grad/||grad||_1) = sign(grad)`), giving plain iterative sign-FGSM (BIM), and one such step of size
`eps` from the clean point recovers FGSM. So `μ = 1` interpolates a smoothed direction over BIM. BIM
already recovers the tangent-plane underfit by re-reading the gradient, so it should lift VGG well above
FGSM — but with nothing damping the sign flips it climbs greedily and settles in whatever sharp maximum
it hits first. The `μ = 1` velocity is a free upgrade over it: same 40 gradient reads, but the
accumulated direction coasts through the small humps.

The step count and step size are where I diverge from the bare momentum recipe. The natural default ties
`alpha = eps/T` so `T` aligned steps exactly fill the budget, but at `T = 40` that gives `alpha = eps/40`
and only `eps` of total travel — less than the `2·eps` box diameter, so a start near one wall cannot even
cross to the other, let alone refine. It is tuned for a handful of steps, not 40. I run `alpha = eps/10`:
total reach `40·eps/10 = 4·eps`, twice the diameter, so from any interior start the iterate crosses in
~20 steps and refines for ~20 more on the high-loss boundary, and the per-step clip keeps the larger step
safe. Between the tied `eps/40` and a coarse `eps/4`, `eps/10` balances reach against refinement grain:
once pinned to the boundary the smallest move is `alpha`, so `eps/10 = 0.000784` settles to within a
tenth of the budget of the true optimum — fine enough for the VGG curvature — while `eps/4` would only
refine in quarter-budget jumps and chatter around it. I also start from a *random* point in the
`eps`-box rather than the clean image: the gradient at `x` is corrupted by sharp curvature artifacts
localized there, and a uniform launch clipped to `[0,1]` jumps off that brittle point so the momentum's
first vote comes from a typical interior point. It costs one `uniform_` fill and a clamp. Neither choice
is transfer-motivated — the operative reasons are that iterating recovers FGSM's underfit and momentum
plus a random start gives a steadier, less-trapped single-model climb.

The cost stays trivial next to the L-BFGS method I rejected at the start: `T = 40` forward/backward pairs per
batch is `2400` network evaluations across the benchmark, versus L-BFGS's `~6·10^5` and only `40×`
FGSM's cost, with the momentum buffer and normalization adding negligible elementwise ops. I can afford
to re-read the gradient 40 times.

The falsifiable expectation the survivor ratio demands is asymmetric. Where FGSM already saturated
(ResNet20 and MobileNetV2 at `0.83–0.88`) iterating can only pin the easy samples harder and creep the
last points to the ceiling; the real gain has to land on the `2.4×`-denser VGG pool, where the extra
gradient reads have room to work. But VGG's `L_inf` robustness is partly architectural, so I do not
expect it to close to `1.0` — momentum recovers the curvature-underfit part of the survivors, not
whatever part is genuinely out of reach. So the signature is a clear VGG gain against near-flat
movement on the saturated columns; if the VGG columns barely move instead, the survivors are genuinely
robust rather than curvature-underfit. The contract fill — 40 iterations of
normalize-accumulate-sign-step from a random start, projecting to `[-eps,eps]` and `[0,1]` each step —
is in the answer.
