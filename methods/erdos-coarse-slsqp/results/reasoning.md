The flat floor told me the only lever is the *shape* of the heights. Let me first make sure I understand
why the floor sits exactly at `0.5`, because that number controls how much room there is to move. With
every `v_i = 1/2`, the zero-shift overlap is `Σ_i v_i(1 − v_i) = 24 · (1/2)(1/2) = 24 · 1/4 = 6`, and the
evaluator rescales by `2/n = 2/24`, giving `6 · 2/24 = 0.5`. I ran the evaluator on the flat vector to be
sure I had the bookkeeping right, and it returns `0.5` with `Σ v = 12`, matching. So the floor is real and
it is the zero shift that pins it: at the flat profile every cell aligns perfectly with itself, and the
self-overlap is maximal. To get below `0.5` I have to break that perfect self-alignment — make the heights
*unequal* so that `Σ v_i(1 − v_i)` shrinks at shift zero without some other shift picking up the slack. So
now I actually have to optimize. The first decision is the piece count, and I deliberately keep it small —
a couple dozen cells — because I want to *understand* what an optimized profile looks like before I commit
to a long vector. With two dozen heights I can run many restarts cheaply, watch where the optimizer parks,
and read the shape off the result. The published `~600`-piece records exist out there, but I should not
start there; I should start where the search is fast and the structure is legible.

Now, what am I actually minimizing? The score is `max_k c_k` over integer shifts, rescaled — a *minimax*.
That is the crux and the difficulty. The objective is the maximum of many smooth functions of the heights
(one `c_k` per shift), so it is piecewise-smooth with kinks exactly where two shifts tie for the worst
overlap. A plain gradient method would chatter at those kinks, and a generic minimizer that only sees the
hard `max` gets no useful gradient information about the *other* near-worst shifts that are about to become
binding. I want to check whether that "many near-worst shifts" worry is real or just a textbook caution, so
I keep it in mind to test on the final vector — if only one shift is ever binding, a hard `max` would be
fine and I am overcomplicating. My instinct from optimizing `max` objectives before is that it is brittle,
so I lean toward replacing the hard `max` with a smooth surrogate — a log-sum-exp / softmax over the shifts
with a sharpness parameter `β`. For moderate `β` the surrogate is a smooth, differentiable stand-in that
*feels* all the near-worst shifts at once and pushes them all down together; as `β → ∞` it converges to the
true `max`. The plan, then: minimize the soft-max overlap, not the hard one, and anneal `β` upward so that
by the end the surrogate genuinely tracks the constant I am reporting.

The second issue is the constraints, and they are not optional decorations — they are what makes this the
Erdős problem rather than a trivial "set everything to zero" problem. The heights must stay in `[0,1]`
(box constraints) and must sum to exactly `n/2` (a linear equality). Without the sum constraint the
optimizer would just drive every height to `0`: then `Σ v_i(1 − v_i) = 0` at every shift and the bound
reads `0`, which is nonsense — it certifies nothing about `C5`. The `Σ v = n/2` rule is precisely the
balance between `A` and `B` that forbids that degenerate escape. So I need a constrained optimizer that
respects a box and a linear equality. The natural fit is **SLSQP** — sequential least-squares quadratic
programming — which handles exactly box bounds plus equality constraints. I will hand it the smooth
soft-max objective, the box, and the equality, and let it find a constrained stationary point.

There is a subtlety I want to get right: SLSQP minimizes the *surrogate*, but I will *report* the true
hard-max overlap of whatever vector it returns. Those two numbers diverge for finite `β` — the soft-max
sits *above* the hard max (log-sum-exp is an over-estimator of the max), and the gap closes as `β` grows.
I should quantify that gap rather than assume it is negligible, because if it stays large then optimizing
the surrogate is optimizing the wrong thing. I will check it on the returned vector once I have one. The
recipe is a short ladder of SLSQP solves at increasing `β`: start soft so the optimizer can move the whole
profile around without getting stuck on a single binding shift, then sharpen `β` so the surrogate hugs the
true max and the final vector is genuinely good under the real evaluator. After each solve I re-project onto
the constraint set — clip to `[0,1]` and shift the heights to restore `Σ = n/2` — so that the vector I
score is exactly feasible, not approximately. (SLSQP satisfies the equality only to its tolerance; the
explicit re-projection makes `Σ v = n/2` hold to machine precision before I trust the score.)

The third issue is local minima. This minimax landscape is non-convex — the overlap envelope can be
flattened in many qualitatively different ways, and SLSQP from one start finds one local basin. With only
two dozen heights the obvious remedy is *multi-start*: run the annealed-SLSQP ladder from many random
feasible initializations and keep the best true-overlap vector. Each random start is a different random
feasible profile (random heights, projected to sum `n/2`), and the basin SLSQP settles into depends on
where it begins. A dozen starts at this small `n` is cheap. Whether it actually *matters* — whether the
restarts disagree — is something I can only learn by running it, so I will look at the spread of the twelve
final scores rather than assume the restarts are redundant.

Now let me run the whole thing and look at what comes back, because most of my claims above are still
predictions. Twelve restarts at `n = 24`, seed `0`. The best vector scores `C = 0.381766`, with
`Σ v = 12.0000` exactly — so the projection is doing its job and the constraint holds. That is a drop of
`0.5 → 0.382`, i.e. essentially the whole gap from the flat floor down to the neighborhood of the known
landmarks, achieved with only 24 cells. Three things I wanted to verify, now that I have the vector:

*Are many shifts really co-binding?* I sorted the rescaled overlaps `c_k · 2/n` for the best vector. The
top five are `0.38177, 0.38176, 0.38176, 0.38176, 0.38176` — a cluster of five shifts within `~10⁻⁵` of
the max, not a lone spike. So the minimax envelope really is nearly flat across several shifts at the
optimum, which is exactly the regime where a hard `max` gives a misleading gradient and the soft-max earns
its keep. The worry was real, not textbook.

*Does the soft-max gap actually close?* On that same vector, the surrogate reads `0.385724` at `β = 60`
(a gap of `+0.00396` above the hard max `0.381766`) and `0.381927` at `β = 1200` (gap `+0.00016`). So the
over-estimate shrinks by more than an order of magnitude across the ladder — confirming both that the
soft-max sits above the true max as I expected, and that annealing is what makes the final surrogate a
faithful stand-in for the number I report. If I had stopped at `β = 60` I would have been steering by a
value `0.004` too high.

*Do the restarts disagree?* The twelve final scores are mostly `0.38177` but three of them park higher —
`0.38187`, `0.3819`, `0.38187`. So multi-start is not redundant: a single start has a real chance of
landing in a slightly worse basin `~10⁻⁴` above the best, and keeping the minimum over restarts is what
buys the last bit. Small effect at this resolution, but a genuine one.

One prediction of mine was *wrong*, and I want to record it honestly rather than paper over it. I had
guessed the optimized profile would come out symmetric — a clean palindrome about the center — since the
overlap score is shift-symmetric. The returned heights are
`[1, 1, 0.72, 0.38, 0.80, 0.32, 0.55, 0.45, 0.24, 0.47, 0, 0, 0.18, 0.21, 0.21, 0.24, 0.45, 0.55, 0.32,
0.80, 0.38, 0.72, 1, 1]`. Comparing against its reverse, the largest entry-wise discrepancy is `0.25` — so
it is decidedly *not* a palindrome. What *is* true, and what actually matters, is the part of my prediction
about polarization: the heights are pushed toward `0` and `1` (the profile is near-binary, with runs of
`1`s at the ends and a couple of hard `0`s in the middle) rather than hovering near `1/2`, arranged so that
no single shift accumulates a large product. The symmetry was a red herring; the polarization was the real
structural feature, and the data bears that one out.

So the picture that emerges, rather than the one I assumed at the start: a coarse near-binary profile, found
by annealed soft-max SLSQP with multi-start, drops the bound from the `0.5` floor to `0.381766` at `n = 24`.
That is a hair above the `0.380927` Haugland landmark — close, but not past it, which is the honest expected
outcome at this resolution: two dozen cells can only approximate the optimal step function coarsely, the
binding shifts are few and the steps are wide, so the worst overlap cannot be shaved as finely as a long
vector allows. The number I report is the true `max_k`-overlap of the best returned vector under the frozen
evaluator, `0.381766`.

The limitation this rung exposes is therefore resolution, and it points directly at the next move: lift the
optimized coarse profile to many more pieces and refine it there, where the extra degrees of freedom let the
optimizer carve the finer structure that brings the bound down toward the published step-function frontier.
