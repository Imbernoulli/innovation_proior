The coarse rung behaved as the two-stage prediction said: annealing crossed from the ceiling `2.0` down to
`1.736925`, and the softmax-Adam polish carved that rough shape to `1.537084`. The decomposition held — the
anneal did the crossing, the polish the descent into the basin — and the polish alone was worth `0.199`, nearly
half the total `0.463` fall, so the basin floor really was well below where the random walk stopped. The
returned `50`-piece profile is asymmetric and partly sparse, roughly `9` of `50` heights driven to near-zero
with mass toward the ends — the end-weighted, peak-suppressing shape I predicted, and evidence the
multiplicative-kick ratchet did what I expected. The search idea is sound. What stopped it is two compounding
limits.

The first is resolution: `1.537084` sits `~0.032` above the `600`-piece AlphaEvolve construction at `1.5053`.
Fifty pieces give an autoconvolution of only `99` nodes, and there is a hard ceiling on how finely a `99`-node
polyline can flatten its own top. To go lower I need more nodes, hence more pieces. The second limit is the one
I flagged last rung and now have to confront head-on, because adding pieces will not dissolve it. By the
conserved sum rule `Σ_k b_k = (Σa)^2`, mass taken off the tallest node reappears on the others, so lowering the
current maximum immediately promotes the second-largest to be the new maximum and `R` does not fall. At the
coarse optimum the autoconvolution already has a whole *plateau* of near-tight nodes, and a gradient on the
softmax surrogate — like a subgradient at the single argmax — can only press one of them at a time: press one
down, another pops up. That is precisely why the softmax polish tapered where it did. The right object is the
*minimax* — lower the largest of the entire near-tight set simultaneously, subject to fixed mass — and no
single-node gradient, however smoothed, solves a minimax.

Naming it that way tells me what to do: minimizing a maximum of smooth functions subject to linear constraints
is the epigraph form solved by linear programming. Introduce an auxiliary `z` for the peak and
`minimize z subject to b_k = (a*a)_k ≤ z for every node, Σa fixed, a ≥ 0`. The constraints are *quadratic* in
the heights, so I linearize around the current profile: writing the step as `d`,
`b_k(a + d) ≈ b_k(a) + Σ_j (∂b_k/∂a_j) d_j`, and `∂b_k/∂a_j = 2 a_{k−j}` (differentiating `b_k = Σ_n a_n
a_{k−n}` picks up the `n=j` and `k−n=j` terms). So each linearized constraint reads `Σ_j 2 a_{k−j} d_j − z ≤
−b_k(a)`, linear in `(d, z)`. The objective picks out `z`; one equality row `Σ_j d_j = 0` holds the mass fixed
so the move is a pure reshape; the bounds keep each `a_j + d_j ≥ 0`. A single LP returns the step that lowers
the modelled peak the most and — crucially — lowers *all* the near-tight nodes at once, because they are all
constraints in the same program. That is the thing the gradient could not do.

There are more accurate ways to handle the quadratic minimax, and I should say why I am not using them. A
sequential *quadratic* program would model the node curvature faithfully, but the curvature of `a*a` couples
every pair of coordinates, so the per-node Hessian is dense and a QP over `600` variables with hundreds of
active quadratic constraints is markedly heavier than an LP; the trust region already fences off the error the
linear model makes, so I would be paying for curvature I can cheaply drop. A bundle method is the textbook tool
for non-smooth minimax but a good deal more machinery, and a trust-region SLP is essentially its pragmatic,
sparsity-friendly version. A general nonlinear solver would exploit neither of my two structural gifts — that
the constraint gradients are shifted copies of the heights (an FFT away) and that only a few nodes are ever
active. So SLP is not a compromise; it is the formulation that matches this objective and runs on an off-the-
shelf LP solver.

It is worth seeing on a toy why the LP genuinely beats the gradient. Suppose two nodes are tied at the peak with
linearized responses `g_1·d`, `g_2·d`, and take `g_1 = (1, 0)`, `g_2 = (0, 1)` with the mass constraint forcing
`d = (t, −t)`. Following the negative gradient of node `1` alone moves `d ∝ (−1, +1)`, which lowers node `1` but
*raises* node `2` by the same amount — the maximum is no better. The minimax LP instead minimizes
`max(t, −t) = |t|` over the feasible line, whose minimum is at `t = 0`: it correctly reports that the two
constraints are in genuine conflict and this point is minimax-stationary. That is the behaviour I need at a
plateau of hundreds of near-tight nodes, unavailable to anything that looks at one node at a time.

The linearization is only trustworthy locally, so the load-bearing safeguard is a trust region. The term I
dropped, `(d*d)_k`, is second order in `d`, so bounding each `|d_j| ≤ tr` with `tr` small keeps the modelled and
true peaks close. I accept the step only if the *true* `R` drops on a full FFT recompute; on success I grow the
trust (`×1.05`, capped) for bolder steps, on rejection I shrink it (`×0.6`) and, when it collapses, restart-kick
out with a small multiplicative perturbation and a trust reset. This accept-true-`R`, grow-on-success,
shrink-on-failure discipline is doing the same job the Metropolis rule did last rung — keeping the method honest
about a surrogate that is only locally valid.

I can put a number on how small the trust region has to be. I normalize to `Σa = 1` inside the solver, so the
peak node is about `R/(2N) ≈ 1.52/1200 ≈ 1.3·10^{−3}` (using the warm-start value I expect to land at), and
typical nodes are of that order. The dropped term `(d*d)_k` with `|d_j| ≤ tr` and a few hundred active
coordinates is bounded by roughly `600·tr^2`. At `tr = 10^{−4}` that is `6·10^{−6}`, about half a percent of the
peak node — small enough that the linear model is a good guide, large enough that a step can occasionally move
the true peak onto an unmodelled node, which is exactly the case the accept-if-true-`R`-drops check catches. At
`tr = 10^{−3}` the dropped term is `~5%` of the peak and the model misleads often; at `10^{−5}` the steps are so
timid that thousands of rounds barely move. So `~10^{−4}` with an adaptive cap around `2.5·10^{−4}` is the band
where the quadratic error is a fraction of a percent, placed by arithmetic rather than a hunch.

One subtlety I have to get right or the LP will cheat. `R` is scale-invariant, so if I asked it to lower the
peak node without pinning the scale it could "descend" by shrinking all the heights, driving `z = max_k b_k`
toward zero while `R` stayed put — a spurious improvement that evaporates when I recompute the true `R`. The
equality row `Σ_j d_j = 0` forecloses it: holding total mass fixed makes every feasible step a pure reshape, so
the only way the LP can lower the modelled peak is by genuinely redistributing overlap. I reinforce this by
renormalizing to `Σa = 1` at the start of each round so the scale never drifts and the peak-node magnitude
`~R/(2N)` stays the fixed reference the trust region is calibrated against. The program is well-posed: the
all-zero step with `z = max_k b_k` is always feasible, so it never comes back empty, and when `linprog` reports
failure it is a numerical edge, which I treat by shrinking the trust and retrying rather than trusting a
spurious step.

Efficiency is what makes SLP affordable at `N = 600` rather than merely a good idea. The autoconvolution has
`1199` nodes, and writing a constraint for every one each round is wasteful because the vast majority are slack.
So I restrict the LP to the *top-K* largest nodes; with a small trust region the true peak cannot jump to a node
far outside that set in one step, so the restricted program is a faithful local model and an order of magnitude
cheaper, with occasional full passes as a guard. The cost that matters is the LP solve, not the FFTs — an FFT at
`600` is well under a millisecond while a `linprog` over `~600` variables is tens of milliseconds — so the
top-K focusing is what buys the hundreds of rounds a real descent needs inside a `~10`-minute budget. The
constraint matrix need not be built entry by entry: the Jacobian `J[k,j] = 2 a_{k−j}` is Toeplitz, so I form it
as a strided sliding-window view over `[0,…,0, 2a_0,…,2a_{n−1}, 0,…,0]`, reading successive length-`n` windows
in reverse, then keep only the top-K rows.

That budget arithmetic fixes the scale at `N = 600`. Lower — staying at `50` — cannot render a fine enough
autoconvolution, as the coarse stall proved. Much higher — toward the `30000` the record uses — would put
`~60000` constraints and `30000` variables into every LP, pushing a solve from tens of milliseconds to seconds
or worse, so I could afford only a handful of rounds. `N = 600` is where the LP is still cheap enough to iterate
thousands of times and the grid holds real structure — and it is exactly the resolution at which AlphaEvolve
reached `1.5053`, so it is the right target. I keep the warm start from the softmax-Adam pass, because SLP is
local and the basin depends on where it begins: `β`-annealed Adam gets from the flat ceiling down to around
`1.52` cheaply and lands in a sensible asymmetric basin, and from there the SLP takes over. The warm start needs
a sharper final `β` here than at `N = 50` — I anneal `β` from `300` up to `2·10^5` over `12000` steps rather
than topping near `5000` — because with `1199` nodes the plateau is finer and a softmax too soft averages over
nodes meaningfully below the peak. That the warm start itself only reaches `~1.52` is expected and is the whole
reason the SLP exists: it is the point where the single-node gradient hits the balloon wall.

Through the peak/mean lens the target is legible: at `N = 600` the prefactor is `1.00083`, so `ratio ≈
peak/mean`. The warm-start `~1.52` means the tallest node is `~1.52×` the average; the flat ceiling was `~2×`;
AlphaEvolve's `1.5053` is `~1.505×`. So the SLP's job is to shave `peak/mean` from `1.52` toward `1.505` — a
tiny, coupled adjustment across hundreds of near-tight nodes, exactly the move the minimax LP makes and the
single-node gradient cannot. I expect the descent to be slow-but-real: each round flattens the plateau a hair
and the `ratio` should drift down through the `1.51`s. When a round finds no improving step I kick out and reset
the trust, checkpointing the best true-`R` vector continuously so a bad restart never costs ground. I run the
SLP as several blocks of tens of rounds rather than one undivided sweep, because a block boundary is a natural
place to recompute the active near-tight set from the current best and reset a collapsed trust region.

So I expect `pieces_N = 600` and a `ratio` falling out of the `1.52`s into the `1.51`s, on a profile sharpening
the peak-suppressing structure — a tall spike at a boundary, mass heavier toward the ends, the middle third
thinned well below the mean, and a large fraction of heights driven near zero. If the SLP stalled immediately at
the warm-start value, that would falsify the minimax diagnosis — it would mean the plateau was not the
obstruction. I will *not* reach the record `1.5028628969` here: that is a `30000`-piece construction found by an
agentic search over tens of hours, and the gain from `1.5053` lives in the fourth decimal, demanding far finer
resolution and far more compute than a single `600`-piece run commands. If more pieces and a longer grind keep
paying, the move at the endpoint is to lift further and grind the SLP longer.
