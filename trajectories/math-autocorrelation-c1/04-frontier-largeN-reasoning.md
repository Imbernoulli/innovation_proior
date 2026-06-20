The sequential-LP refinement kept paying — `1.5371` at `N=50`, down to `1.5172` at `N=600` — but the feedback was
explicit about where it stops and why. At `N=600` my trust-region SLP, warm-started from one softmax-Adam basin,
converges to about `1.517` and then only crawls: restart kicks buy a few ten-thousandths and no more. That is the
honest signature of a single local constructor — it finds *a* good basin, not *the* good basin. AlphaEvolve reached
`1.5053` at the very same `N=600` resolution, so the gap is not resolution alone; it is that AlphaEvolve's agentic
search explored far more of the basin landscape than one warm start can. So the endpoint question is: with the same
SLP engine but more search and a finer grid, how far down can I genuinely push, and how close can I get to the
record `1.5028628969`?

Two levers are available, and I should be clear-eyed about what each can and cannot buy. The first is **basin
diversity**: instead of one warm start, launch the SLP from several structurally different initializations and keep
the global best. The record constructions are *asymmetric*, with mass concentrated into tall spikes at the
boundaries and a thinned interior — the AutoEvolver `30000`-piece solution is exactly this, huge spikes at the two
ends over an irregular plateau. My rung-3 solution already drifted toward that shape (a spike at one boundary), which
tells me the good basins live in the boundary-spike family. So I will seed the SLP not only from the rung-3 shape but
from a spread of explicit boundary-spike profiles — varying which end carries the heavier spike, since the optimum is
asymmetric and I do not know a priori which way it leans — and let each run a slice of the budget, restart-kicking out
of trust collapse, keeping whichever start lands lowest. This is the cheap, honest way to imitate what an agentic
search does: try several promising shapes and keep the best, rather than trusting one.

The second lever is **resolution**: repeat-lift the best `N=600` shape to a finer grid, where the autoconvolution can
develop structure a coarse vector cannot hold, and SLP-polish there. The repeat-lift is free — replacing each height
by copies gives the *same* function and the *same* `R` — so it costs nothing and risks nothing; the SLP then has more
coordinates to coordinate. The catch is cost: the LP at the heart of each SLP round scales with the number of pieces,
and at a few thousand pieces a single round is several seconds, so a finer grid buys representational room but spends
the budget fast. I have to weigh more pieces against fewer rounds, and at this scale the trade is delicate — the LP
solve, not the FFTs, is the bottleneck, so doubling the grid roughly halves how many rounds I can afford.

The mechanics of the SLP are unchanged from the previous rung — epigraph `z`, linearized node constraints, trust
region, accept-if-true-`R`-drops, top-K focus on the near-tight nodes — but the *schedule* is where the endpoint
differs. I spend most of the budget on the multi-start search at the resolution where the SLP is fast enough to do
many restarts, because that is where I expect the real gain: finding a lower basin than the single rung-3 one. I keep
the FFT evaluator and the top-K focusing so the runs stay affordable, and I checkpoint the global best continuously so
a long search never loses ground.

What do I expect? The multi-start search should find a basin a little below `1.517` — the boundary-spike seeds are in
the right family, and several of them exploring in parallel should beat one warm start — landing somewhere in the low
`1.51`s, edging toward the AlphaEvolve `600`-piece `1.5053`. I am honest, though, that I do *not* expect to reach
`1.5028628969`, and I want to say why precisely. That record is a `30000`-piece deliberately irregular construction
found by an agentic coding-agent search (AutoEvolver, via Claude/Opus "aspiration prompting") that ran for tens of
hours, after TTT-Discover's `30000`-piece `1.5028628983` and AlphaEvolve's `600`-piece `1.5053`; the gain from
`1.5053` to `1.50286` lives entirely in the fourth decimal place and was bought with two orders of magnitude more
pieces and vastly more compute than a single bounded SLP run on a small grid commands. The gap from wherever this
endpoint lands down to `1.50286` — and the further gap down to the provable floor `1.28` — is the honest measure of
how open the first autocorrelation inequality still is. So the endpoint of this ladder is the frontier a single
diversified SLP constructor can reach, with the AutoEvolver record standing above as the still-open distance —
exactly the way the `30000`-piece record stands above any modest-grid local construction, and exactly the shape the
`C2` sibling took when its single gradient constructor stopped short of the evolutionary-search record. The honest
number I report is the one the evaluator returns on the returned vector, and the remaining gap to `1.5028628969` is
the part of the problem this constructor leaves open.
