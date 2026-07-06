The coarse rung behaved exactly as the two-stage prediction said it would: annealing crossed from the ceiling
`2.0` down to `1.736925`, a large first drop that broke the flat symmetry, and the softmax-Adam polish then
carved that rough shape down to `1.537084`. The decomposition held — the anneal did the crossing, the polish did
the descent into the basin — and the polish alone was worth `0.199`, nearly half of the total `0.463` fall from
the ceiling, which tells me the basin floor really was well below where the random walk stopped. The returned
`50`-piece profile is asymmetric and partly sparse, with roughly `9` of the `50` heights driven to (near-)zero
and mass concentrated toward the ends — the end-weighted, peak-suppressing shape I predicted, and evidence that
the multiplicative-kick ratchet toward sparsity did what I expected. So the search idea is sound. What stopped
it is two compounding limits, and I have to name both because together they dictate the whole design of this
rung.

The first limit is resolution, and it is the obvious one: `1.537084` sits `~0.032` above the `600`-piece
AlphaEvolve construction at `1.5053` and `~0.034` above the record `1.5028628969`. Fifty pieces give an
autoconvolution of only `99` nodes, and there is a hard ceiling on how finely a `99`-node polyline can flatten
its own top. To go lower I need more nodes, which means more pieces. The second limit is the one I flagged last
rung and now have to confront head-on, because it will not go away just by adding pieces. When I push the peak
of the autoconvolution down, I am not moving a single node; by the conserved sum rule `Σ_k b_k = (Σa)^2`, mass I
take off the tallest node has to reappear on the others, so lowering the current maximum immediately promotes the
second-largest node to be the new maximum, and `R` does not actually fall. At the coarse optimum the
autoconvolution already has many nodes within a hair of its peak — a whole *plateau* of near-tight nodes — and a
gradient on the softmax surrogate, or a subgradient at the single argmax, can only press on one of them at a
time. Press one down, another pops up: pressing a balloon. This is precisely why the softmax polish tapered
where it did rather than continuing to the record band; it was never attacking the right object. The right
object is the *minimax* — lower the largest of the entire near-tight set simultaneously, subject to the mass
being fixed — and no single-node gradient, however smoothed, solves a minimax.

Naming it that way tells me what to do, because minimizing a maximum of smooth functions subject to linear
constraints is a classic problem with a classic solution: the epigraph form solved by linear programming. I
introduce an auxiliary variable `z` standing for the peak, and ask to `minimize z subject to b_k = (a*a)_k ≤ z
for every node k, the mass Σa held fixed, and a ≥ 0`. If the constraints were linear this would be one LP and I
would be done. They are not: each node `b_k = Σ_n a_n a_{k−n}` is *quadratic* in the heights. So I linearize
them around the current profile. Writing the step as `d`, `b_k(a + d) ≈ b_k(a) + Σ_j (∂b_k/∂a_j) d_j`, and the
gradient of a self-convolution node is itself a shifted copy of the heights: differentiating `b_k = Σ_n a_n
a_{k−n}` with respect to `a_j` picks up two equal terms (the `n = j` term and the `k − n = j` term), giving
`∂b_k/∂a_j = 2 a_{k−j}`. So each linearized constraint reads `b_k(a) + Σ_j 2 a_{k−j} d_j ≤ z`, i.e.
`Σ_j 2 a_{k−j} d_j − z ≤ −b_k(a)`, which is linear in the unknowns `(d, z)`. The objective picks out `z` alone;
one equality row `Σ_j d_j = 0` holds the mass fixed so the move is a pure reshape; and the bounds keep each
`a_j + d_j ≥ 0`. A single LP solve then returns the step `d` that lowers the modelled peak the most — and,
crucially, it lowers *all* the near-tight nodes at once, because they are all constraints in the same program.
That is exactly the thing the gradient could not do.

Before I commit to linearizing, I should be honest that there are more accurate ways to handle the quadratic
minimax and say why I am not using them. I could keep the exact quadratic and solve a sequential *quadratic*
program each round — an SQP — which models the curvature of the nodes faithfully instead of throwing it away.
But the curvature of `a*a` couples every pair of coordinates, so the per-node Hessian is dense, and a QP over
`600` variables with hundreds of active quadratic constraints is markedly heavier to assemble and solve than an
LP; the trust region already controls the error the linear model makes, so I would be paying for curvature I can
cheaply fence off instead. I could reach for a bundle method, the textbook tool for non-smooth minimax, which
accumulates subgradients into a stabilized model — but that is a good deal more machinery, and a trust-region
SLP is essentially the pragmatic, sparsity-friendly version of the same idea: a polyhedral model of the max,
stabilized by a trust region, refreshed each round. A general nonlinear solver thrown at the epigraph problem
would not exploit the two structural gifts I have — that the constraint gradients are just shifted copies of the
heights (an FFT away) and that only a few nodes are ever active — whereas the LP-with-top-K formulation uses
both. So SLP is not a compromise I settle for; it is the formulation that matches the structure of this specific
objective, and it runs on an off-the-shelf LP solver. That is why I reach for it over the more elaborate
alternatives.

I want to see, on a toy, why the LP genuinely beats the gradient rather than just asserting it, because the
distinction is the whole justification for switching engines. Suppose two nodes are tied at the peak, with
linearized responses `g_1·d` and `g_2·d` to a step `d`, and take the simplest non-trivial case `g_1 = (1, 0)`,
`g_2 = (0, 1)` with the mass constraint forcing `d = (t, −t)`. If I followed the negative gradient of node `1`
alone, I would move in `d ∝ (−1, +1)`, which lowers node `1` but *raises* node `2` by the same amount — the new
maximum is unchanged or worse. The minimax LP instead minimizes `max(g_1·d, g_2·d) = max(t, −t) = |t|` over the
feasible line, whose minimum is at `t = 0`: it correctly reports that no descent direction exists here, that the
two constraints are in genuine conflict and this point is minimax-stationary. Following either single gradient
would have moved me and made things no better; only the program that sees both constraints together knows to
stop, or — when the geometry permits — finds the compromise direction that lowers both. That is the behaviour I
need at a plateau of hundreds of near-tight nodes, and it is unavailable to any method that looks at one node at
a time.

The linearization is only trustworthy locally, so the load-bearing safeguard is a trust region. The term I
dropped in `b_k(a + d) ≈ b_k(a) + (∇b_k)·d` is the exact quadratic `Σ a-cross-d`, second order in `d`, so if I
bound each `|d_j| ≤ tr` with `tr` small the modelled peak and the true peak stay close and the LP step is
faithful. I accept the step only if the *true* `R` actually drops when I recompute it with a full FFT; on
success I grow the trust region (`×1.05`, capped) to take bolder steps next round, and on rejection I shrink it
(`×0.6`) and, when it collapses, restart-kick out with a small multiplicative perturbation and a trust reset.
Too large a trust region and the neglected quadratic moves the true peak onto a node the linear model did not
watch, so the step is rejected and the trust shrinks; too small and progress crawls. Adapting it each round is
what lets a linearized model converge on a genuinely non-convex objective — the accept-true-`R`, grow-on-success,
shrink-on-failure discipline is doing the same job the Metropolis rule did last rung, keeping the method honest
about a surrogate that is only locally valid. This whole construction — epigraph, linearized node constraints,
trust region, accept-if-true-`R`-drops — is sequential linear programming: focus the optimization on the constraints that
are close to tight, the positions where the autoconvolution is largest, and drive them down together.

I can put a number on how small the trust region has to be, which is worth doing so `tr` is a reasoned quantity
rather than a knob I twiddle. I normalize the heights to `Σa = 1` inside the solver, so the peak node is about
`R/(2N) ≈ 1.517/1200 ≈ 1.3·10^{−3}` and typical nodes are of that order. The term the linear model drops is
exactly `(d*d)_k = Σ_n d_n d_{k−n}`, the self-convolution of the step, which with `|d_j| ≤ tr` and a few hundred
active coordinates is bounded by roughly `600·tr^2`. At `tr = 10^{−4}` that is `600·10^{−8} = 6·10^{−6}`, about
half a percent of the peak node — small enough that the linear model is a good guide but large enough that a
step can occasionally move the true peak onto an unmodelled node, which is precisely the case the accept-if-
true-`R`-drops check catches and the trust-region shrink then corrects. If I tried `tr = 10^{−3}` the dropped
term would be `~10×` larger, `~5%` of the peak, and the model would mislead often; if I tried `10^{−5}` the
steps would be so timid that thousands of rounds would barely move. So `~10^{−4}` with an adaptive cap around
`2.5·10^{−4}` is the band where the quadratic error is a fraction of a percent — small enough to trust, large
enough to progress — and the arithmetic, not a hunch, is what places it there.

There is a subtlety in the constraints that I have to get right or the LP will find a way to cheat. `R` is
scale-invariant — multiplying every height by a constant leaves it unchanged — so if I only asked the LP to
lower the peak node without pinning the scale, it could "descend" by simply shrinking all the heights, driving
`z = max_k b_k` toward zero while `R` stayed exactly where it was. That would be a spurious improvement in the
linear model that evaporates the moment I recompute the true `R`. The equality row `Σ_j d_j = 0` is what
forecloses it: holding the total mass fixed makes every feasible step a *pure reshape* rather than a rescale, so
the only way the LP can lower the modelled peak is by genuinely redistributing overlap among the nodes — the
move I actually want. I reinforce this by renormalizing `a` to `Σa = 1` at the start of each round, so the scale
never drifts and the peak-node magnitude `~R/(2N)` stays a fixed reference the trust region is calibrated
against. Mass-fixed plus scale-normalized is what makes the epigraph LP measure real progress instead of an
artifact of amplitude.

The LP itself is well-posed, which is worth checking so I am not handing the solver something degenerate. The
variables are the `n` step components and the epigraph scalar, `n + 1 = 601` of them; the inequality rows are
the (top-K) linearized node constraints, at most `1199`; there is one equality row `Σ_j d_j = 0`; and each `d_j`
is boxed into `[max(−a_j, −tr), tr]` with `z` free. The all-zero step `d = 0` with `z = max_k b_k` is always
feasible, so the program never comes back empty, and the feasible set is a bounded polytope, so the minimum is
attained. When `linprog` does report failure it is a numerical edge, not an infeasibility, and I treat it by
shrinking the trust and retrying rather than trusting a spurious step.

Efficiency forces the next choice, and it is what makes SLP affordable at `N = 600` rather than a good idea I
cannot run. The autoconvolution at `600` pieces has `2·600 − 1 = 1199` nodes, and writing a constraint for every
one each round is wasteful because the vast majority are slack — deep below the peak and playing no role in the
minimax. So I restrict the LP to the *top-K* largest nodes, the active near-tight set. With a small trust region
the true peak cannot jump to a node far outside that set in one step, so the restricted program is a faithful
local model, and it is an order of magnitude cheaper to solve; I interleave occasional full passes over all
`1199` nodes as a guard so nothing sneaks above the peak unseen. The cost that actually matters here is the LP
solve, not the FFTs — an FFT at `N = 600` is still well under a millisecond, while a `linprog` over `~600`
variables and hundreds of constraints is tens of milliseconds — so the top-K focusing, by cutting the constraint
count, is what buys me the hundreds of rounds a real descent needs inside a `~10`-minute budget. A back-of-the-
envelope: at, say, `30` ms per solve, `10` minutes is roughly `2·10^4` solves, comfortably enough for many
restart blocks of tens of rounds each.

The constraint matrix does not need to be built entry by entry, which matters because assembling it naively each
round would swamp the savings from top-K. The Jacobian `J[k, j] = 2 a_{k−j}` is Toeplitz — each row is the same
sequence of doubled heights shifted by one — so I form it as a strided sliding-window view over the array
`[0,…,0, 2a_0,…,2a_{n−1}, 0,…,0]` padded with `n−1` zeros on each side, reading off successive length-`n`
windows in reverse. No per-entry loop, no dense copy until the LP needs it: the `2 a_{k−j}` structure falls out
of the shift automatically. Then I keep only the rows indexed by the top-K near-tight nodes, so the matrix
actually handed to `linprog` is small. This is the concrete reason the "cheap FFT, focused LP" accounting holds
in practice rather than just in principle.

That budget arithmetic is also what fixes the scale at `N = 600` rather than higher or lower. Lower — staying at
`50` — cannot render a fine enough autoconvolution, as the coarse stall proved. Much higher — jumping toward the
`30000` pieces the record uses — would put `~60000` constraints and `30000` variables into every LP, pushing a
single solve from tens of milliseconds to seconds or worse, so I could afford only a handful of rounds and the
SLP would never grind. `N = 600` is the scale where the LP is still cheap enough to iterate thousands of times
and where the grid is fine enough to hold real structure — and, not coincidentally, it is exactly the resolution
at which AlphaEvolve reached `1.5053`, so it is the right target to aim this engine at. I keep the warm start
from the softmax-Adam pass, because SLP is a local method and the basin it lands in depends on where it begins:
the `β`-annealed Adam gets from the flat ceiling down to around `1.52` cheaply and lands in a sensible
asymmetric basin — I lift the coarse shape's lesson to `600` pieces and let Adam re-break the symmetry at the
finer resolution — and from there the SLP takes over and keeps descending where the gradient stalled. The warm
start needs a sharper final `β` at this resolution than it did at `N = 50` — I anneal `β` from `300` up to
`2·10^5` over `12000` steps rather than topping out near `5000` — because with `1199` nodes instead of `99` the
plateau of near-tight nodes is finer, and a softmax that is too soft averages over nodes that are meaningfully
below the peak, blurring the target; the sharper endpoint is what lets the surrogate resolve the true peak
before the SLP takes over. That the warm start itself only reaches `~1.52` and no lower is expected and, in
fact, is the whole reason the SLP exists — it is the point where the single-node gradient hits the balloon wall.

It is worth reading the target through the peak/mean lens, because at `N = 600` the prefactor `2N/(2N−1) =
1.00083` is within a thousandth of `1`, so the reported `ratio` is essentially `peak/mean` itself. The warm-start
value `~1.52` therefore means the autoconvolution's tallest node is about `1.52×` its average node; the flat
ceiling was `~2×`; and the AlphaEvolve band `1.5053` is `~1.505×`. So the SLP's job, stated in these terms, is
to shave `peak/mean` from `1.52` toward `1.505` — to flatten the top of the node profile by roughly one part in
a hundred while holding the mass fixed. That is a tiny, coupled adjustment across hundreds of near-tight nodes,
exactly the kind of move the minimax LP makes and the single-node gradient cannot, which is why I expect the
descent to be slow-but-real rather than either fast or stalled: each round flattens the plateau a hair, and the
`ratio` should drift down through the `1.51`s as the rounds accumulate. When a round finds no improving step I
kick out with a small multiplicative perturbation and reset the trust, and I checkpoint the best true-`R` vector
continuously, so a restart that lands worse never costs me the ground I have gained. So the
rung is a two-stage pipeline: a short `β`-annealed Adam pass to reach `~1.52`, then a long run of trust-region
top-K sequential LP to grind the whole near-tight plateau down, checkpointing the best true-`R` vector
throughout so nothing is lost to a bad restart. I run the SLP as several blocks of tens of rounds each rather
than one undivided sweep, because a block boundary is a natural place to recompute the active near-tight set
from the current best and to reset a trust region that may have collapsed mid-block; a single monolithic run
tends to strand itself once the trust has shrunk to nothing, whereas re-seeding each block from the checkpointed
best, with a fresh trust, gives the search repeated chances to find a new improving direction on the plateau it
has reached.

What do I expect the harness to report? The Adam warm start should land near `1.52`, as it did the analogous job
last rung. The sequential LP is the new engine, and I expect it to keep finding small, real improvements round
after round — each solve lowers the whole near-tight plateau a little — pushing from `1.52` down into the `1.51`
range and, with enough rounds, toward the `1.5053` band of the `600`-piece AlphaEvolve construction. On the
metric that reads as `pieces_N = 600` and a `ratio` that falls out of the `1.52`s into the `1.51`s, with a
returned profile that should show the peak-suppressing structure sharpening: a tall spike at a boundary, mass
heavier toward the ends, the middle third thinned well below the mean, and a large fraction — I would guess a
few hundred — of the `600` heights driven near zero. If instead the SLP stalled immediately at the warm-start
value, that would falsify the minimax diagnosis — it would mean the plateau was not the obstruction — so the
round-by-round descent is the thing to watch. I am honest that I will *not* reach the record `1.5028628969`
here: that is a `30000`-piece construction found by an agentic search spending tens of hours, and the gain from
`1.5053` to `1.50286` lives in the fourth decimal and demands both far finer resolution and far more compute than
a single `600`-piece run commands. So I expect this rung to prove the SLP principle — that attacking the whole
near-tight plateau with a trust-region LP buys the descent the gradient could not, into the AlphaEvolve
`600`-piece band — and then to taper, limited by resolution and by how many LP rounds I can afford. That taper is
the opening for the endpoint: if more pieces and a longer SLP grind keep paying, the move is to lift further and
grind the SLP longer, pushing toward the record.
