The annealing works, and it plateaus, and I diagnosed why before I even ran it: with the boundary
pressed against its perimeter limit, almost every random flip lands on boundary that is already
correct, and the rare flip that would fix a genuinely misclassified fish — a mackerel sitting just
outside the net, or a sardine the net is still swallowing — is one proposal in thousands. The search
is undirected. This is precisely the wall the benchmark hit too: ALE-Agent's SA reached performance
`2880`, fifth place, and then a program-evolution system, ShinkaEvolve, evolved that same SA to
`3140`, second place, with two changes and only two. I do not have to guess what to try next; I have
to reproduce those two levers in my representation and see them move the number.

The first lever ShinkaEvolve found is **caching the validation process**. In their kd-tree solution
they augmented each node to cache subtree statistics — bounding boxes and fish counts — so that
checking and scoring a candidate net no longer walked the tree from scratch. The principle, stripped
of the kd-tree, is: never recompute from the whole structure what you can maintain incrementally on
the small patch a move touches. In my grid representation I already cache the per-cell fish counts
(that is what makes scoring O(1)) and the running boundary-edge count (that makes the perimeter
check O(1)). What I have *not* been caching is which cells are currently *on the boundary* — and
that is the set my proposals should be drawn from and the set my next operator needs to consult. So
I add a boundary-flag cache: a per-cell bit saying "this region cell touches the outside." Crucially,
when I accept a flip, the boundary status of cells can only change in the 3×3 window around the
flipped cell, so I refresh exactly those nine flags and nothing else. Now every candidate move can
ask "is this a boundary cell?" and "what are its outside neighbors?" in O(1), with no scan of the
grid. That is the same idea as caching subtree statistics: the validation and the proposal both read
a cache that an accepted move updates only locally.

The second lever is the one that actually redirects the search: the **targeted edge move**.
ShinkaEvolve described it exactly — heuristically identify a misclassified fish (for instance a
mackerel outside the polygon) and greedily move the nearest edge to correct its state. This is the
cure for undirectedness. Instead of proposing a uniformly random flip and hoping it helps, I aim the
proposal at a fish the current net gets wrong. In my grid that becomes concrete and cheap. I sample a
boundary cell of the region — those are exactly the cells the cache flags — and I look at its outside
neighbors. If one of them is mackerel-rich (a cell whose `#mackerel − #sardine` is high, sitting just
outside the net), I propose *adding* it: that is moving the nearest edge outward to capture a
misclassified mackerel. Conversely, if the boundary cell I sampled is itself sardine-heavy — the net
is catching sardine there — I propose *removing* it: moving the edge inward to release a misclassified
sardine. Either way the proposal is *directed* at a fish the net misclassifies and at the nearest
piece of boundary that can fix it, exactly as ShinkaEvolve framed it. I still wrap these directed
proposals in the same Metropolis acceptance and cooling, so the search can still decline a fix that
costs too much elsewhere; what changes is that a large fraction of proposals are now aimed at real
errors instead of at already-correct boundary.

I mix the two move types rather than replacing the random flip entirely. A pure targeted search
would over-commit to fixing local errors and could stop exploring the larger reshapes that the
high-temperature random flips still discover. So with some probability I propose a targeted edge
move and otherwise fall back to the baseline uniform flip. The temperature schedule, the
perimeter and topology checks (simple-point plus the diagonal-pinch guard), and the warm start from
the best rectangle all carry over unchanged — this rung is the previous rung's SA with the boundary
cache feeding a directed operator on top.

Two things I want to verify rather than assume, because it is easy to fool myself here. First, the
cache must actually be *faithful*: the boundary flags it maintains incrementally have to match what a
full recompute would say, or the targeted operator aims at phantom boundary. The 3×3 refresh after
each accepted flip is the exact footprint over which boundary status can change, so the cache is
correct by construction — but I confirm it by watching that the internal `a − b` the search tracks
still matches the frozen evaluator's exact count to the unit on the emitted net. Second, the
targeted move must not quietly break validity: it goes through the same simple-point, pinch, and
perimeter gates as any flip, so a directed proposal that would self-intersect or bust the budget is
rejected like any other. With both confirmed, the directed operator only changes *which* legal moves
get proposed, never whether illegal ones slip through.

What I expect, and what I will read in the feedback: the targeted edge move should lift exactly the
instances where rung 3 plateaued against the perimeter ceiling — the overlapping-shoal layouts where
the win comes from a coordinated "release this sardine cell, capture that mackerel cell" trade that
undirected flips almost never propose. On the easy seeds where the net is already near-optimal there
is little misclassification to target, so the rung should hold rung 3's result, not regress. The gain
should be modest but real and reproducible across runs — the same character as the benchmark's
`2880 → 3140`: not a new algorithm, but the same SA made to *search in the right direction*, with a
cache that makes the directed proposals affordable. This is the endpoint: it reproduces the two
specific levers ShinkaEvolve evolved on top of ALE-Agent's SA, and there is no further rung here
because the next gains on the real benchmark come from a different kind of search (continuous-polygon
representations, multi-net restarts) rather than from sharpening this local-move annealer.
