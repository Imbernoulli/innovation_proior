The coarse annealing gave me a twenty-piece profile at `0.884823`, just under the published `0.88922`, and
it left a specific diagnosis and a specific prediction. The diagnosis: the shape is right — a tall spike, a
shaped shoulder, several heights pinned at zero, an autoconvolution with a genuine flat cap about fourteen of
its forty-one nodes wide — but twenty pieces cannot render both a wide plateau and a sharp shoulder at once,
so the grid itself caps how high the μ-weighted mean level `⟨t⟩/T` can climb. The prediction that followed:
*lifting this same shape onto a finer grid*, with no new idea about the shape, should let the plateau widen
and the shoulder steepen and push `R` past `0.885` into the high `0.89`s. This rung tests that prediction. I
do not want to throw the coarse shape away and search a long vector from scratch — the previous rung showed
that searching a long vector blind is hopeless — I want to *keep* this shape and hand it more resolution.

The lift has to be free, or it is not worth doing, and it is: replacing each of the twenty heights by `k`
copies of itself produces a `20k`-piece step function that is *literally the same function*, only dilated.
Concretely, if `v' = repeat(v, k)`, then the cell containing position `x` in the fine profile has height
`v'_{⌊x⌋} = v_{⌊⌊x⌋/k⌋} = v_{⌊x/k⌋}`, which is exactly `f(x/k)` — the coarse function stretched horizontally
by a factor `k`. And from the very first rung I know `R` is invariant under dilation (the whole objective is
translation- and dilation-invariant), so `R(v') = R(v)` exactly. I check it rather than trust it: lifting the
coarse profile by `k = 2, 5, 10` returns `0.8848227841` every time, identical to ten digits. So upscaling
costs nothing and risks nothing; it just hands the optimizer a finer canvas. The catch is what the canvas
looks like at the lifted point. Every block of `k` identical copies is flat, so the lifted profile sits on a
*degenerate plateau*: along any direction that keeps each block internally flat, `R` does not change at all
(it is the same coarse function), and along the *within-block* directions — letting neighbouring copies
inside a block differ — the gradient at the exactly-uniform point is zero by symmetry. The reason is clean:
the lifted profile is invariant under reflecting each block internally (swapping the left and right copies),
`R` is invariant under reflection, and the "make the left copy larger, the right copy smaller" direction is
*odd* under that reflection — so its directional derivative equals its own negative and must vanish. The
first-order signal along exactly the directions I lifted the profile to exploit is zero; only the second
order carries information, and a gradient method started precisely at the symmetric point feels nothing. The new degrees of freedom the lift
created are real, but at the lifted point they are switched off; something has to switch them on.

Before I commit to lifting the coarse shape, I should rule out the obvious alternative: skip the annealing
entirely and run gradient ascent on five hundred heights from a smooth bump seed. The previous rung already
answers this without my having to run it. A gradient method is a hill-climber — a very good *coordinated*
one, moving all coordinates together, but still a method that only ever goes up — and I measured at the
coarse rung that hill-climbing from a bump seed stalls low, in the mid-`0.7`s, because the good support (the
particular arrangement of spike, shoulder, and genuine zeros) is separated from any smooth start by ridges
that descend before they climb. Adam from a bump seed at five hundred pieces would find that same poor basin,
only faster. What makes the lift work is not the gradient at all; it is that the object being lifted is the
*downhill-tolerantly searched support* from the coarse rung, already on the far side of those ridges. So the
hierarchy is not an optimization convenience but a necessity with a clean division of labor: the coarse
anneal supplies a support that gradient descent cannot reach on its own, and the lift carries that support to
a resolution where the gradient can finally refine it. Skipping either half fails — anneal alone is capped by
resolution, gradient alone is capped by its inability to cross the ridges that define the support.

So I need two things the coarse rung's machinery cannot supply at this scale: a way to break the within-block
symmetry, and an optimizer that moves all the new coordinates *together*. The second is the binding
constraint. At `N` in the hundreds, the single-coordinate annealing that worked at twenty is hopeless — the
previous rung's arithmetic already showed why: each evaluation is `O(N log N)`, dozens of times more
expensive than at twenty, and a coherent reshape of a shoulder now spans a hundred coordinates that a
one-at-a-time move can only align by luck before cooling freezes it. What I need is exactly the *coordinated*
move the very first rung identified as the only thing that raises `R`: a direction in the full height space,
followed by every coordinate at once. That means a gradient. So I compute the gradient of `R` with respect to
the height vector directly, and I make the whole thing differentiable the same way the coarse rung did — by
smoothing the hard `max`.

The gradient derivation is the heart of this rung, so I do it carefully. Write `A = ‖f*f‖_2^2`,
`C = ‖f*f‖_1`, and replace the peak by the softmax `B(β) = m + β^{-1} log Σ_j e^{β(L_j − m)}`, so the
surrogate objective is `Q = A / (B C)`. Everything depends on the node values `L_j`, so I first differentiate
`Q` with respect to each `L_j` by the quotient rule:

`∂Q/∂L_j = (1/(B C)) ∂A/∂L_j − (A/(B^2 C)) ∂B/∂L_j − (A/(B C^2)) ∂C/∂L_j.`

Each piece is local and cheap. The energy `A = ⅓ Σ_j (L_j^2 + L_j L_{j+1} + L_{j+1}^2)` touches node `j`
through the two cells that meet there, contributing `⅓(2L_j + L_{j+1})` from the cell to its right and
`⅓(2L_j + L_{j−1})` from the cell to its left, so `∂A/∂L_j = ⅓(4L_j + L_{j+1} + L_{j−1})`. The mass
`C = ½ Σ_j (L_j + L_{j+1})` contributes `½` from each adjacent cell, so `∂C/∂L_j = 1` at interior nodes. And
the softmax gives `∂B/∂L_j = e^{β(L_j − m)} / Z` — the softmax weight `w_j/Z`, a probability distribution
peaked at the argmax that becomes a hard indicator as `β → ∞`. It is worth reading the three terms of
`∂Q/∂L_j` together, because they *are* the flatten-the-cap instruction in disguise. The energy term
`∂A/∂L_j /(BC)` is positive at every node and largest where `L_j` is already large, so it pushes all the
high nodes up. The peak term `−(A/(B^2 C)) w_j/Z` is a penalty concentrated almost entirely on the current
peak node — that is where the softmax weight `w_j` lives — so it pushes the single tallest node *down*. The
mass term subtracts a uniform amount everywhere. Net effect: the gradient rewards raising the *sub-peak*
nodes toward the peak while penalizing the lone highest one, which is exactly "widen the plateau, do not let
a single spike run away." The instruction I read off the layer-cake by hand at the first rung falls straight
out of the calculus of the surrogate. Now the last link: the node values are the
self-convolution, `L_j = c_{j−1}` with `c = v * v`, so `c_k = Σ_n v_n v_{k−n}` and
`∂c_k/∂v_p = v_{k−p} + v_{k−p} = 2 v_{k−p}` (the two symmetric ways `p` appears in the product). Chaining,
`∂Q/∂v_p = Σ_k (∂Q/∂c_k)(∂c_k/∂v_p) = 2 Σ_k (∂Q/∂c_k) v_{k−p}`, which is `2` times the *correlation* of the
node-gradient with the height vector — computable as a single convolution of the node-gradient against the
reversed heights. So the whole gradient is a handful of FFTs, `O(N log N)`, exactly as cheap as one
evaluation. I verify the analytic gradient against a finite-difference gradient on a random vector and they
agree to about `10^{-10}`, so the chain rule is right and I can trust it to drive the optimizer. The budget
comparison against the coarse rung's tool is stark and settles the choice: one Adam step costs a few FFTs and
advances *all five hundred* coordinates along a coherent direction, so a few thousand such steps — a few
thousand coordinated reshapes — run in seconds; to make the same coordinated progress with single-coordinate
annealing would take on the order of five hundred accepted moves just to touch every coordinate once, times
however many sweeps a coherent reshape needs, i.e. millions of evaluations for what the gradient does in
thousands. At this resolution the gradient is not merely faster, it is the only affordable way to make
coordinated moves at all.

The optimizer riding this gradient is Adam, and the reason is the dynamic range I already see in the coarse
profile. Its heights span more than an order of magnitude — a spike at `1.0` down to shoulder values around
`0.05` — and after the lift the fine profile inherits that spread and will sharpen it further. A plain
fixed-step gradient ascent is wrong for such a vector: a step large enough to move the spike meaningfully
blows the thin shoulder values negative (then clipped to zero, destroying them), while a step small enough to
be safe for the shoulder crawls uselessly on the spike. Adam's per-coordinate adaptive scaling is exactly the
fix — it divides each coordinate's step by a running estimate of that coordinate's own gradient magnitude
(the bias-corrected second moment, with the standard `b1 = 0.9`, `b2 = 0.999`), so the spike and the thin
shoulder advance on comparable *relative* terms, the same scale-free invariance the coarse rung's
multiplicative perturbation gave the annealer. After every step I clip to non-negative to stay legal, and I
track the best *true* `R` ever seen — not the surrogate `Q` — for a precise reason. The softmax makes
`B ≥ ` the true peak always, so `Q = A/(BC) ≤ R` at every point, and the maximizer of the smooth surrogate
sits slightly off the maximizer of the real ratio; the vector Adam likes best under `Q` is not quite the
vector I want. Evaluating the exact `R` (with the hard `max`) at each step is cheap — one more reduction over
the same node values I already have — so I keep a running best-`R` vector and return that. This guarantees I
never hand back something worse than the best true ratio I actually passed through, regardless of how the
surrogate wobbles as `β` anneals.

The `β` schedule per level follows the same logic as the coarse ladder, but the ceiling has to grow with the
resolution, and there is a real reason rooted in what I am trying to build. The softmax overshoots the true
peak by at most `log(2N + 1)/β`, so at `N = 100` (with `201` nodes, `log ≈ 5.3`) a ceiling of `β ≈ 83N =
8300` keeps the overshoot under `6·10^{-4}`, and at `N = 500` (`1001` nodes, `log ≈ 6.9`) a ceiling of
`β ≈ 123N = 61500` keeps it under `10^{-4}`. But the sharper subtlety, the one I flagged at the coarse rung,
now bites: the flatter the cap I succeed in building, the *more* nodes cluster near the peak, and the softmax
sum is then dominated not by one term but by the whole tied plateau, so its overshoot is closer to
`log(#tied nodes)/β`. As the lift lets the plateau widen, the number of near-peak nodes grows, and `β` must
be pushed correspondingly sharper to keep `B` faithful to the true `max` — otherwise the optimizer chases a
blurred peak and the true `R` lags the surrogate. That is exactly why the ceiling climbs from level to level.
I anneal `β` geometrically from soft to sharp within each pass — soft early, where the surrogate is smooth
and the gradient points broadly toward better shapes, sharp late, where it is a faithful stand-in for the
`max` — and I run two passes per level.

The last ingredient switches the degenerate plateau back on: a small multiplicative perturbation
`v ↦ |v(1 + ε ξ)|`, `ξ` standard normal, applied right after each upscale. The lifted point has zero
within-block gradient, so Adam started exactly there can sit still; a tiny kick (`ε` around `0.03`–`0.06`)
breaks the block symmetry and gives the gradient something to grab, while being small enough not to damage
the good coarse shape it is perturbing. The kick shrinks as the ladder climbs — larger (`~0.06`) at the
first lift to `100`, gentler (`~0.03`) at the lift to `500` — for a real reason: at finer resolution the
carved structure is more delicate, so the perturbation needed to merely unstick the symmetric plateau is a
smaller fraction of a height that has already been shaped, and a kick as large as the coarse one would undo
more fine structure than it frees. So the rung is a ladder of free lifts and paid refinements: take the
optimized twenty-piece profile, upscale `×5` to `100`, kick, refine with two `β`-annealed Adam passes;
upscale `×5` again to `500`, kick, refine. The two passes at each level are not redundant — they have
different jobs. The first pass carries the kick and a moderate `β` ceiling: its job is to let the freshly
lifted, just-perturbed shape *reorganize*, moving broadly over a still-smooth surrogate while the plateau
finds its new width and the shoulder re-settles at the finer resolution. The second pass runs with no kick
and a sharper `β` ceiling: its job is to *polish* the reorganized shape against a nearly-faithful `max`,
tightening the plateau and steepening the shoulder without the disruption of a restart. Each lift is exactly
ratio-preserving; each refinement carves the within-block fine structure the previous resolution could not
represent.

The `×5`-then-`×5` factoring is itself a choice worth defending against the two extremes. I could lift once
by `×25`, straight from twenty to five hundred, or by `×2` a handful of times. The single `×25` jump is bad
because each block would be twenty-five copies wide, and the kick-plus-gradient would have to carve all that
within-block structure from a standing start on one enormous degenerate plateau — very close to just
optimizing five hundred heights from a crude blocky initialization, throwing away the graded refinement the
hierarchy exists to provide. Many tiny `×2` lifts are bad in the opposite way: each lift pays the fixed
overhead of a kick and two full Adam passes, while doubling barely adds new degrees of freedom, so most of the
budget goes to re-settling shapes that were already settled. A factor of `×5` sits in the middle — four new
within-block degrees of freedom per piece, enough for a kick to find real asymmetric structure and for the
gradient to exploit it, but not so many that the refinement is carving from scratch — and two such lifts span
the twenty-to-five-hundred range in exactly the graded way the hierarchy is meant to.

What I expect is the coarse rung's prediction coming true, and it does. The first lift to `100` clears the
twenty-piece value comfortably — `100` pieces can widen the plateau and steepen the shoulder further than
`20` — landing around `0.89`, into the band of the published fifty-step result; the second lift to `500` adds
more, and the construction settles at `R = 0.894706`, a clean `+0.0099` over the coarse rung and just under
the fifty-step `0.89628`. In the one variable that tracks the mechanism, the μ-weighted mean level `⟨t⟩/T`
has moved from `0.442` at twenty pieces to `0.4474` at five hundred (since `R = 2⟨t⟩/T`) — a small but real
step further toward the box's `1/2`, exactly the "hold higher, fall faster" the finer grid was supposed to
buy. I am honest that this does *not* reach the `~0.9016` of the best published `~575`-step constructions from
a few thousand Adam steps per level, and I want to be precise about why, because the reason is no longer
resolution. At five hundred pieces the autoconvolution has a thousand nodes; the plateau-versus-shoulder
tradeoff that strangled the twenty-piece profile is now mild — there is plenty of grid to hold a wide plateau
and a sharp shoulder both. The residual gap to `0.9016` is instead that pushing `⟨t⟩/T` the last sliver
toward `1/2` requires the shoulder to take a specific *irregular*, non-monotone profile, and a few thousand
smooth Adam steps descending from a smoothly lifted seed carve most of that structure but not the last of it.
Those `~575`-step records spent vastly more compute — on the order of a million gradient trajectories — and
on this problem the last fraction of a percent is dominated by exactly that fine irregular structure, not by
any idea I am missing. The per-lift gains tell the same story: `0.8848` to about `0.89` to `0.894706`, a
diminishing but still strictly positive sequence.

Two consistency checks reassure me the ladder refined the right thing rather than wandering off. First, the
support survived: lifting a zero produces `k` zeros, so the genuine gaps the coarse anneal opened
(`v_0, v_1, v_4…v_7` in the original) are preserved through every upscale, and the refined five-hundred-piece
profile is recognizably a finer rendering of the same spike-plus-shoulder-plus-zeros support — the method
sharpened the coarse shape, it did not jump to a different shape family. Second, the movement is monotone and
decelerating, which is what a resolution-limited refinement of a fixed shape should look like: each lift buys
less than the last as the grid stops being the bottleneck. If instead the value had *jumped* at some lift, or
kept climbing at a constant rate, I would suspect the shape itself was still reorganizing and the coarse
support was not actually near-optimal — but a smooth, saturating climb is the signature of polishing a shape
that is already essentially right. What I can see is that the
gradient is *still moving* when I stop at `500`: the improvement per lift has not flattened out. That is the
opening for the endpoint. If more resolution and a longer, more carefully annealed gradient run keep paying,
the move is to lift once more — to a few thousand pieces — and spend a long, kicked, sharpening Adam schedule
grinding toward the published step-function frontier.

I can make that next step falsifiable in advance. My diagnosis is that I am refining a fixed shape whose
gains saturate, and that the shape's own ceiling — the smooth spike-and-shoulder family this whole ladder
lives in — sits around `0.90`, at the published `~575`-step frontier. If that is right, then lifting to a few
thousand pieces and grinding much longer should push into the low `0.90`s and then *flatten*: the value
should approach roughly `0.90`, match the best published step-function results, and stop climbing no matter
how much more resolution or how many more Adam steps I add, because the bottleneck will no longer be the grid
or the optimizer but the shape family itself. The layer-cake variable should track this — `⟨t⟩/T` creeping
from today's `0.4474` toward but not past roughly `0.45`. If instead the endpoint blew through `0.91` and
kept rising, my "smooth basin with a shape ceiling near `0.90`" picture would be wrong, and I would have to
conclude the gradient can reach far more of the problem than I think. Either way the endpoint is a clean
test, and the honest expectation is the saturating one: a frontier this ladder reaches and cannot exceed
without a different kind of search entirely.
