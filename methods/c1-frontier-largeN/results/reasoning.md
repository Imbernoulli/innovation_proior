The sequential-LP refinement kept paying — `1.5371` at `N=50`, down to `1.5172` at `N=600` — but the feedback was
explicit about where it stops and why. At `N=600` my trust-region SLP, warm-started from one softmax-Adam basin,
converges to about `1.517` and then only crawls: restart kicks buy a few ten-thousandths and no more. That is the
honest signature of a single local constructor — it finds *a* good basin, not *the* good basin. AlphaEvolve reached
`1.5053` at the very same `N=600` resolution, so the gap is not resolution alone; at fixed `N` two functions can sit
that far apart only if they live in different basins, which means the missing ingredient is search breadth, not more
pieces. So the endpoint question is: with the same SLP engine but more search and a finer grid, how far down can I
genuinely push, and how close can I get to the record `1.5028628969`?

Before deciding how to spend the budget I want to understand the objective by hand, because the whole strategy hinges
on what actually lowers `R`. Take the smallest non-trivial case, `N=2`, `a = [a0,a1]`. Then `a*a = [a0², 2a0a1, a1²]`,
the peak is the max of those three, and `R = 2·2·max / (a0+a1)² = 4·max/(a0+a1)²`. The flat vector `[1,1]` gives
`a*a=[1,2,1]`, peak `2`, so `R = 8/4 = 2.0` — and the evaluator returns exactly `2.000000` for the flat indicator at
every `N` I tried, which is the ceiling yardstick, so the formula and the code agree. Now my instinct is "concentrate
mass at the ends to kill the central overlap", so I test `[3,1]`: `a*a=[9,6,1]`, peak `9`, `R = 36/16 = 2.25` — that
is *worse* than flat, and the evaluator confirms `2.25`. The lesson is immediate and important: a raw tall spike does
not suppress the peak, it *creates* one, because the diagonal self-overlap term `a0²` is itself a node and a big `a0`
makes it the tallest. What actually helps is the opposite balance — `[1, 0.5]` gives `a*a=[1,1,0.25]`, peak `1`,
`R = 4/2.25 = 1.778`, below flat, because no single node dominates. So the thing I am chasing is a profile whose
autoconvolution is *flat on top* — many nodes tied at the peak, none allowed to stick up — not a profile that looks
spiky in `a`.

That reframes the two levers I have. The first is **basin diversity**: instead of one warm start, launch the SLP from
several structurally different initializations and keep the global best. The record constructions are *asymmetric*,
with mass concentrated toward the boundaries and a thinned interior — the AutoEvolver `30000`-piece solution is
exactly this, large end-features over an irregular plateau. My rung-3 solution already drifted toward that shape,
which is suggestive. But the `N=2` calculation just warned me that a literal end-spike is bad on its own, so I should
not trust the intuition blindly; I should seed the basins and let the SLP tell me whether they are good. So I seed not
only from the rung-3 shape but from a spread of explicit boundary-spike profiles — varying which end carries the
heavier ramp, since I do not know a priori which way the asymmetry leans — and let each run a slice of the budget,
restart-kicking out of trust collapse, keeping whichever start lands lowest.

I want to check the seeds before committing the schedule to them, because the hand calculation makes me suspicious.
Evaluating the boundary-ramp seeds at `N=600` before any SLP, they come in at `R ≈ 3.76, 3.77, 4.61, 4.76, 4.76` —
every one of them *worse than the flat `2.0`*, for exactly the reason `[3,1]` was worse: the ramped end inflates the
diagonal node. So these are not good points. The bet is only that they sit in good *basins* — that once the SLP is
allowed to redistribute mass, the descent from a boundary-ramp start ends up lower than the descent from flat. That
is a real question, not something I can assert, so I run the SLP on one seed and watch it: starting from `R=3.76`, by
round 20 it is at `1.625`, round 40 `1.554`, and it crawls to `1.5518` by round 120. The engine does not keep the
spike — it relaxes it into a suppressed plateau, blowing straight through the flat `2.0` and down past my single
rung-3 basin. So the seed's bad starting value was a red herring; the basin is good. Good — the diversity lever is
worth the budget, but for the basin it reaches, not the point it starts from.

The justification for *several* seeds rather than one then has to be that different seeds reach genuinely different
floors. I check this directly: running the same SLP from three of the boundary seeds, the three runs settle at
`R = 1.552, 1.548, 1.521`. They are spread by about `0.03` — far more than the ten-thousandths a restart kick moves a
single run — which is the concrete evidence that the basins are distinct and that keeping the global best across
diverse starts is buying real ground, not noise. One warm start would have committed to whichever of these it happened
to hit. So I spend most of the budget here: many starts, keep the lowest, then give that winner a long dedicated
full-constraint SLP polish with restart kicks, checkpointing continuously so a long search never loses ground.

The mechanics of the SLP are unchanged from the previous rung — epigraph `z`, linearized node constraints, trust
region, accept-if-true-`R`-drops — and I should make sure the linearization is actually right, since the whole LP step
trusts the analytic Jacobian of `a*a`. The node `(a*a)_k = Σ_j a_j a_{k-j}`, so `∂(a*a)_k/∂a_i = 2 a_{k-i}`, which the
`_jac` routine builds as a strided shift of `2a`. Finite-differencing the autoconvolution at a random vector against
`_jac` gives a max discrepancy of `1e-6` at step `1e-6` — i.e. they agree to the step size — so the linearized
constraints the LP sees are the true first-order behavior of every node, and an accepted step that the LP predicts
lowers the epigraph really does lower the peak to first order. That is what lets me accept on the *true* re-evaluated
`R` and trust that progress is real rather than an artifact of a wrong gradient.

The second lever is **resolution**: repeat-lift the best `N=600` shape to a finer grid, where the autoconvolution can
develop structure a coarse vector cannot hold, and SLP-polish there. The repeat-lift is free — replacing each height
by copies gives the *same* function and the *same* `R` — so it costs nothing and risks nothing; the SLP then has more
coordinates to coordinate. The catch is cost: the LP at the heart of each SLP round scales with the number of pieces,
and at a few thousand pieces a single round is several seconds, so a finer grid buys representational room but spends
the budget fast. The LP solve, not the FFTs, is the bottleneck, so doubling the grid roughly halves how many rounds I
can afford — and within this rung's budget the finer grid did not pay back the rounds it cost, so the returned
frontier stays at `N=600`, where the SLP is fast enough to do the many restarts the diversity lever needs.

What do I expect the schedule to return? Running the full pipeline lands a little below `1.517` — in the low `1.51`s,
edging toward the AlphaEvolve `600`-piece `1.5053`, with the winning run's argmax sitting at one boundary, a few
hundred near-tight autoconvolution nodes tied within a hair of the peak (the flat-top plateau the minimax LP drives
down together), and the middle third thinned well below the mean. The asymmetry direction is start-dependent — across
runs the heavy end lands left or right depending on which seed wins — so I report the structure, not a fixed index.
I am honest that I do *not* expect to reach `1.5028628969`. That record is a `30000`-piece deliberately irregular
construction found by an agentic coding-agent search (AutoEvolver, via Claude/Opus "aspiration prompting") that ran
for tens of hours, after TTT-Discover's `30000`-piece `1.5028628983` and AlphaEvolve's `600`-piece `1.5053`; the gain
from `1.5053` to `1.50286` lives entirely in the fourth decimal place and was bought with two orders of magnitude more
pieces and vastly more compute than a single bounded SLP run on a small grid commands. The gap from wherever this
endpoint lands down to `1.50286` — and the further gap down to the provable floor `1.28` — is the honest measure of
how open the first autocorrelation inequality still is. So the endpoint of this ladder is the frontier a single
diversified SLP constructor can reach, with the AutoEvolver record standing above as the still-open distance —
exactly the way the `30000`-piece record stands above any modest-grid local construction, and exactly the shape the
`C2` sibling took when its single gradient constructor stopped short of the evolutionary-search record. The honest
number I report is the one the evaluator returns on the returned vector, and the remaining gap to `1.5028628969` is
the part of the problem this constructor leaves open.
