The annealing works, it plateaus, and the third-rung table lets me locate the plateau precisely rather
than wave at it. The reversibility thesis was vindicated where I predicted: seed 2 went from the
greedy's `3188` to `3967`, and I can read the mechanism straight off the `a` and `b` columns — the
region moved from `a = 3618, b = 431` to `a = 4073, b = 107`, recovering `455` of the mackerel the
greedy had sacrificed while shedding another `324` sardine down to near the floor. That is the
two-directional trade doing exactly what I built it for. But two seeds tell the other half of the story.
Seed 1 actually came in *below* the greedy — `2330` against the greedy's `2362` — precisely the risk I
flagged: the search warm-started from the box (`a ≈ 2643`) and carved sardine down to `b = 312`, but it
stuck at `a = 2641` and never extended to grab the outlying mackerel the greedy had reached at
`a = 2770`, and its perimeter sat at `360000`, exactly the safe cap. It was pinned against the length
limit and could only shed, not extend. And seed 3 held the box entirely (`m = 4`, `b = 500` untouched):
the undirected search never stumbled onto the carve at all, even starting right next to it. So the
plateau is not diffuse. It is two specific pathologies: on a perimeter-saturated boundary the search
can shed but cannot make the coordinated extend-move (seed 1), and on a boundary that needs one
particular carve the random walk may never propose it (seed 3). Both are the same disease I diagnosed
before running: the proposals are undirected, so the rare flip that would fix a genuinely misclassified
fish is one in thousands and gets drowned in flips that touch already-correct boundary.

This is precisely the wall the benchmark hit too. ALE-Agent's SA reached performance `2880`, fifth
place; then a program-evolution system, ShinkaEvolve, evolved that same SA to `3140`, second place,
with two changes and only two. Those AtCoder numbers live on a relative scale and are not comparable to
my raw `a − b + 1` means — I cannot expect my mean to land near `3140` — but the *character* of that
jump is exactly what I am after: not a new algorithm, the same SA made to search in the right direction.
So I do not have to guess what to try next. I have to reproduce those two levers inside my grid
representation and see whether they move my number the way they moved the benchmark's, on the very
seeds my third rung left stranded.

The first lever ShinkaEvolve found is caching the validation process. In their kd-tree solution they
augmented each node to cache subtree statistics — bounding boxes and fish counts — so that checking and
scoring a candidate net no longer walked the tree from scratch. Stripped of the kd-tree, the principle
is: never recompute from the whole structure what you can maintain incrementally on the small patch a
move touches. In my grid I already cache the per-cell fish counts `Av`, `Bv` (that is what makes scoring
`O(1)`) and the running boundary-edge count (that makes the perimeter check `O(1)`). What I have *not*
been caching is which cells are currently *on the boundary* — and in the third rung that cost me: to
propose a remove I sampled a random region cell and threw it away if it turned out to be interior, and
as the region grew, most remove proposals were wasted on interior cells. More to the point, the operator
I am about to add needs to sample boundary cells and inspect their outside neighbors constantly, and
doing that by scanning the region on every proposal would be `O(region)` per step — with directed moves
at more than half of tens of millions of proposals, that scan would dominate the whole run. So I add a
boundary-flag cache: a per-cell bit saying "this region cell touches the outside or the grid border."

The cache is only worth anything if I can keep it current cheaply, so I need to know the exact footprint
over which boundary status can change when I flip one cell. Let me derive it rather than assume nine
cells. A cell `d`'s boundary status depends only on `d` itself being inside and on whether each of its
four neighbors is inside. So flipping cell `(i, j)` can change the boundary status of a cell `d` only if
`d = (i, j)` or if `(i, j)` is one of `d`'s four neighbors — that is, `d` is `(i, j)` or one of its four
orthogonal neighbors: five cells in all. A cell two steps away, like `(i+2, j)`, has neighbor set
`{(i+1,j), (i+3,j), (i+2,j±1)}`, none of which is `(i, j)`, so its status cannot change — confirmed. The
exact footprint is those five cells; refreshing the full `3 × 3` window of nine is a clean safe superset
that also touches the four diagonals harmlessly. So after each accepted flip I recompute the boundary
flag on the `3 × 3` neighborhood and nothing else, and the cache is correct by construction — not by
hope, but by the footprint argument I just checked. Now every candidate move can ask "is this a boundary
cell?" and "what are its outside neighbors?" in `O(1)`, with no scan of the grid. That is the same idea
as caching subtree statistics: the proposal and the validity both read a cache that an accepted move
updates only locally.

The second lever is the one that actually redirects the search: the targeted edge move. ShinkaEvolve
described it exactly — heuristically identify a misclassified fish, for instance a mackerel outside the
polygon, and greedily move the nearest edge to correct its state. This is the cure for undirectedness,
and in my grid it becomes concrete and cheap because of the cache I just built. With some probability,
instead of a uniform flip, I sample a boundary cell — those are exactly the cells the cache flags — and
I look at its outside neighbors. If one of them is mackerel-rich, a cell whose `Av − Bv` is high sitting
just outside the net, I propose *adding* it: that is moving the nearest edge outward to capture a
misclassified mackerel. Conversely, if the boundary cell I sampled is itself sardine-heavy, the net is
catching sardine there, so I propose *removing* it: moving the edge inward to release a misclassified
sardine. Either way the proposal is aimed at a fish the net gets wrong and at the nearest piece of
boundary that can fix it, exactly as ShinkaEvolve framed it, and it still passes through the same
Metropolis acceptance and cooling so the search can still decline a fix that costs too much elsewhere.

I want to ground this in seed 1, because that is where the third rung most visibly failed and where I
can check the operator is actually pointed at the right thing. Seed 1 was left at `a = 2641`, about `129`
mackerel short of what the greedy captured, with a perimeter pinned at the `360000` cap and `b` still at
`312`. Undirected SA could not help it: the extend-moves that would grab those mackerel were refused by
the perimeter gate, and the shed-moves it *could* make did not point anywhere useful. The targeted
operator attacks exactly this. A targeted remove samples a sardine-heavy boundary cell and drops it —
which both sheds sardine (seed 1 still has `312` to work on) and *shortens the boundary*, freeing
perimeter; a targeted add then samples a boundary cell whose outside neighbor is mackerel-rich and
spends that freed perimeter reaching outward to capture it. The coordinated "shave a sardine cell here,
extend toward a mackerel cell there" that a blind random walk almost never proposes as a pair is now
two directed proposals, each aimed at a real error, that the acceptance rule can chain through the cool.
That is the trade seed 1 needed and could not get. On seed 3, symmetrically, the directed remove aims
straight at the `500`-sardine boundary the undirected search never carved. So I can predict where the
lift should land: the seeds the third rung left stranded against the plateau — seed 1, held back by the
perimeter gate, and seed 3, held at the untouched box — are exactly where directing the proposals should
buy the most.

Let me put the seed-1 trade on a ledger with the perimeter arithmetic, because the whole point is that
it fits under a budget the undirected search could not work with. At `G = 50` each cell side is `2000`,
so each boundary unit-edge is `2000` of length. A targeted remove of a sardine-heavy boundary cell with,
say, one inside neighbor shortens the boundary by `2` unit-edges — it hands back `4000` of perimeter —
while raising `a − b` by the sardine it releases. A targeted add of a mackerel-rich outside neighbor with
one inside neighbor spends `2` unit-edges — `4000` of perimeter — while raising `a − b` by the mackerel
it captures. So the paired move is roughly perimeter-neutral: `−4000` then `+4000`, landing right back at
the `360000` cap seed 1 was pinned to, but with a sardine cell traded out and a mackerel cell traded in.
That is why the coordination matters: neither half is affordable alone against a saturated boundary — the
add alone busts the cap, the remove alone just shrinks the net — but *as a sequence* they net to zero
perimeter and a positive score swing. An undirected walk would have to propose those two specific flips
close together by luck; the directed operator proposes each of them on purpose, at the sardine cell and
the mackerel cell it can see. This is also exactly the sense in which the grid move *is* ShinkaEvolve's
"move the nearest edge": adding or removing one boundary cell shifts the net's local edge inward or
outward by exactly one cell-width, so a cell flip aimed at a misclassified fish is literally the nearest
axis-parallel edge stepping over to reclassify it.

I do not replace the uniform flip entirely, and the reason is in the third rung's one clear success. A
pure targeted search would over-commit to fixing local errors and could stop exploring the larger
reshapes that the high-temperature random flips still discover — and it was exactly such a large reshape,
warm-started and then annealed freely, that won seed 2. So with probability `P_TARGET = 0.55` I propose a
targeted edge move and otherwise fall back to the baseline uniform flip: a little over half the proposals
directed at real errors, a little under half kept for undirected exploration. I am not pretending `0.55`
is derived from anything but the judgment that I want the directed operator to dominate without silencing
the exploration that found seed 2; it is a mix, not a replacement. Everything else carries over
unchanged — the temperature schedule `T0 = 8 → T1 = 0.05`, the simple-point and diagonal-pinch topology
checks, the exact running perimeter, and the warm start from the best rectangle. This rung is the
previous rung's SA with the boundary cache feeding a directed operator on top.

It helps to estimate how much the mix actually changes the *density* of useful proposals, because that
density is the whole lever. In the third rung essentially none of the proposals were error-directed:
every flip was uniform, so the chance a given proposal landed on the one boundary spot where a
misclassified fish could be fixed fell as the boundary grew, and near the plateau it was a small
fraction of a percent per flip. Now `55%` of proposals sample a boundary cell and immediately inspect
its four neighbors for the most mackerel-rich outside cell or check whether the cell itself is
sardine-heavy — so `55%` of proposals are *pointed at boundary*, and among those the ones sitting next
to a genuine misclassification are a real fraction rather than a needle in the whole grid. Even if only,
say, one boundary cell in ten borders a fixable error, the directed stream proposes a plausible fix on
the order of `5%` of all flips instead of the third rung's fraction of a percent — an order-of-magnitude
increase in the rate at which the search even *looks* at the errors it needs to fix. The gain per fix is
small, a handful of fish, but the number of fixes attempted per second is what was starving before, and
that is the quantity the directed operator multiplies. The acceptance rule and cooling are unchanged, so
a fix that costs too much elsewhere is still declined; what changes is only how often a real fix is put
on the table.

I should check the degenerate case, because an operator that helps the stranded seeds must at least not
harm the ones already near-optimal, and seed 2 at `a = 4073, b = 107` is nearly there. On a boundary with
no misclassified fish, the targeted add samples a boundary cell and finds its best outside neighbor has
`Av − Bv ≤ 0` — there is no mackerel out there to grab — so the proposed add is a downhill move that the
cooled acceptance rejects, and the targeted remove only fires when the sampled boundary cell is itself
sardine-heavy, which on a clean boundary it is not. So on an already-good net the directed operator
mostly proposes moves that get rejected, and the search behaves like the third rung's undirected anneal
with a slightly reduced effective step count. That is the worst it can do: it cannot *degrade* a net that
has nothing to fix, it can only spend some proposals discovering there is nothing to fix. This is why I
expect seed 2 to hold or edge up rather than regress, and it is the reassurance that adding a directed
operator on top of a working anneal is safe — the direction only changes which legal moves get proposed,
never whether a bad one is forced through.

There is a consequence of the operator's shape that I can turn into a second, sharper prediction. The
targeted moves are, by design, usually *uphill*: a targeted add is chosen precisely because its outside
neighbor is mackerel-rich, so its `Δ = Av − Bv` is typically positive, and a targeted remove fires
precisely on a sardine-heavy cell, so releasing it also typically raises `a − b`. A positive-`Δ` move is
accepted deterministically at any temperature — the Metropolis rule takes every improving flip
regardless of `T`. So unlike the third rung's payoff, which leaned on lucky high-temperature reshapes
that differ run to run, the directed gains here are mostly improving moves that the search will take
whenever it happens to sample the right boundary cell, early or late in the cool. That predicts the
lift should be *reproducible* across runs in a way the plateau-crossing reshapes were not: if the
targeted operator is really front-loading the improving fixes the undirected search found only by
accident, then repeated runs should cluster tightly above the third rung's band rather than scatter. If
instead the gain swings wildly between runs, then it is coming from chance reshapes and not from the
directed fixes I am positing, and my mechanism story is wrong. The reproducibility of the gap, not just
its sign, is therefore a test of *why* the operator works — and it is the same signature the benchmark's
`2880 → 3140` jump carried: a small, steady lift from directed local repair, not a volatile leap from a
new global search.

One more accounting note keeps this honest about scale. Every seed still ends with the exact same final
guard as the third rung — a true-perimeter check that falls back to the best rectangle if the traced
polygon is ever degenerate — so a directed proposal that would self-intersect, pinch, or bust the budget
is rejected by the simple-point, diagonal-pinch, and perimeter gates just like any other flip, and if
the annealed net somehow came out illegal the rung still emits a legal box. The targeted operator
therefore has exactly one job and one risk surface: it changes which legal moves the search spends its
proposals on, and it inherits every validity guarantee the third rung already earned. That containment
is what lets me add a heuristic, hand-tuned operator on top of the anneal without reopening the question
of whether the output is a legal net — a question I want settled by construction so that the only thing
the feedback is measuring is whether directing the search actually catches more fish.

There are cheaper and more elaborate versions of each lever that I considered and set aside. For the
directed operator I could maintain a global priority queue of the most-misclassified fish and always
attack the worst one — strictly more directed, but it needs a heap updated on every accepted flip at
`O(log n)`, and since my resolution is a cell and not a fish, the queue would collapse back to ranking
cells anyway; sampling a boundary cell and looking at its four neighbors is `O(1)` and captures the same
"aim at the nearest fixable error" without the bookkeeping. For the cache I could recompute the whole
boundary set every few thousand accepted flips instead of incrementally — amortized cheaper per line of
code, but staler between refreshes, and the operator reads the cache far more often than the region
changes, so a stale boundary set would mis-aim proposals. The `3 × 3` incremental refresh is both exact
and `O(1)`, which dominates the periodic-recompute alternative outright. So the cheap local versions are
not a compromise; they are the better engineering given that a cell is my finest unit and the cache is
read far more than it is written.

Two things I want to verify rather than assume, and I want to be careful not to fool myself about which
check proves which claim. First, is the cache *faithful*? Its correctness rests on the footprint
argument — the `3 × 3` refresh covers the exact five cells whose boundary status a flip can change — so
it is right by construction, and I should note that even if it somehow drifted it could not corrupt the
score: the score is driven entirely by the exact `Av`, `Bv` counts and is independent of the boundary
flags, which only steer *which* cells get proposed and gate the remove shortcut. A stale flag can at
worst skip a legal move or propose an interior remove, and the latter is caught downstream — removing an
interior cell fails the simple-point test because it would open a hole. So a wrong cache costs efficiency,
never legality. Second, does the accounting still match the truth? I confirm the internal `a − b` the
search tracks equals the frozen evaluator's exact count on the emitted net; agreement to the unit is my
evidence that the `Av`/`Bv` bucketing and the boundary trace are exact — the same guarantee the third
rung had, now carried through the directed operator. I keep these two verifications separate on purpose:
the footprint argument certifies the cache, the count-match certifies the accounting, and conflating them
would let a bug in one hide behind the other.

What I expect, and what I will read in the feedback, follows seed by seed from the third-rung table. Seed
1, held back by the perimeter gate, should recover the mackerel it left uncaptured once the targeted
remove frees the length the targeted add needs — its `a` should climb back up toward and perhaps past the
greedy's `2770`, lifting it above its stalled `2330`. Seed 3, held at the untouched box, is where a
directed remove finally has a clear target, so it should carve past the box for the first time. Seed 2 is
already near its frontier at `a = 4073, b = 107`, so there is little misclassification left to aim at and I
expect only a small directed gain on top of its `3967`. Seed 4, already well carved at `b = 68`, should
edge up a little. Seed 5 is the one I am least hopeful about: its shoals are so heavily overlapping that
`b` is still `1222` and its perimeter is already near the limit at `392000`, so there may simply be no
profitable carve to target, and it could hold or even fall back to the box — a wobble, not a regression.
The overall gain should be modest but real and reproducible across runs, the same character as the
benchmark's `2880 → 3140`: not a new algorithm, but the SA made to search in the right direction with a
cache that makes the directed proposals affordable. This is the endpoint. The next gains on the real
benchmark come from a different kind of search — continuous-polygon representations, multi-net restarts —
rather than from sharpening this local-move annealer, so there is no further rung to add here; the job of
this one is to show the two levers move the number, and seeds 1 and 3 are where I expect to watch them
do it.
