The annealing works, and it plateaus, and I think I know why before I even look at a run: with the
boundary pressed against its perimeter limit, almost every random flip lands on boundary that is
already correct, and the rare flip that would fix a genuinely misclassified fish — a mackerel sitting
just outside the net, or a sardine the net is still swallowing — is one proposal in thousands. The
search is undirected. That diagnosis is the thing to act on. It also matches where the benchmark got
stuck: ALE-Agent's SA reached performance `2880`, fifth place, and a program-evolution system,
ShinkaEvolve, took that same SA to `3140`, second place, with a small number of targeted edits
described as "strengthening the directionality of the search." So I have two directions worth
pursuing — make the search affordable enough to be directed, and then actually direct it — and a
number to check myself against.

Start with the affordability. ShinkaEvolve's kd-tree change augmented each node to cache subtree
statistics — bounding boxes and fish counts — so that scoring a candidate net no longer walked the
tree. Stripped of the kd-tree, the principle is: never recompute from the whole structure what you can
maintain incrementally on the small patch a move touches. In my grid representation I already cache
per-cell fish counts (that is what makes scoring O(1)) and a running boundary-edge count (that makes
the perimeter check O(1)). What I am *not* caching is which cells are currently on the boundary — and
that is exactly the set a directed operator would want to draw proposals from. So I add a per-cell
boundary-flag cache: a bit saying "this region cell touches the outside."

The whole value of that cache rests on a claim I should not just assert: that an accepted flip only
disturbs boundary flags in a small local window, so I can refresh that window and nothing else. A
cell `(p,q)`'s flag is a function of its own membership and its four orthogonal neighbours'
membership. A flip changes membership at exactly one cell `(i,j)`. So the only flags that can change
are the cell `(i,j)` itself and the cells that have `(i,j)` as an orthogonal neighbour — i.e. `(i,j)`
plus its four orthogonal neighbours, a plus-shape. That argument says the diagonal cells of the 3×3
window should *never* change, and the orthogonal-plus is the true footprint. Let me actually test
that rather than trust the argument. I generate 300k random grids, snapshot every cell's true
boundary flag, flip one random cell, recompute, and look for any flag that changed outside the 3×3
window — and separately, for any flag at a strict diagonal of the flipped cell that changed:

```
trials=300000  outside_3x3_violations=0
trials=300000  strict_diagonal_changes=0
```

Zero on both. So the footprint is genuinely the orthogonal plus, and refreshing the full 3×3
neighbourhood after each accepted flip is a safe superset of it — cheap and correct. That is the
local-maintenance discipline the kd-tree caching used, made concrete here.

Now the cache wants to feed something. The undirectedness is the real problem, so the operator I add
should aim a proposal at a fish the net gets wrong instead of at uniformly random boundary. In the
grid this becomes cheap and concrete: sample a boundary cell — those are exactly the cells the cache
flags — and look at its outside neighbours. If one of them is mackerel-rich (high `Av − Bv`, sitting
just outside the net), propose adding it: that is moving the nearest edge outward to capture a
misclassified mackerel. Conversely, if the sampled boundary cell is itself sardine-heavy (`Av − Bv <
0`, the net is catching sardine there), propose removing it: moving the edge inward to release a
misclassified sardine. Either way the proposal is aimed at a fish the net misclassifies and at the
nearest boundary that can fix it. I keep the same Metropolis acceptance and cooling around it, so the
search can still decline a fix that costs too much elsewhere; what changes is that a large fraction of
proposals now land on real errors rather than on already-correct boundary.

I want to mix this with the baseline uniform flip rather than replace it. A pure targeted search would
over-commit to local error-fixing and stop discovering the larger reshapes that high-temperature
random flips still find. So with probability `P_TARGET` I propose a targeted edge move and otherwise
fall back to the uniform flip. The cooling schedule, the warm start from the best rectangle, and the
validity gates all carry over.

Before I trust any of this I have to check the two pieces of bookkeeping the operator leans on,
because both are easy to get subtly wrong.

First, the perimeter bookkeeping. Toggling a cell changes the running boundary-edge count, and the
code claims the change is `4 − 2·sameNbr`, where `sameNbr` is the number of orthogonal in-region
neighbours (negated when removing). The reasoning: an added cell contributes four unit edges, but each
in-region neighbour was already exposing a shared edge that now becomes interior — that edge vanishes
from the neighbour's count and is not added to the new cell's, so each such neighbour removes two from
the global count. Net `4 − 2·sameNbr`. That is a derivation, not a verification, so I check it against
a recomputed-from-scratch boundary count: 200k random grids, toggle a random cell, compare the
predicted delta to (after − before) of a full O(grid) boundary-edge recount.

```
checks=200000  fails=0
```

Exact every time, including the grid-border cases where a cell has fewer than four neighbours (the
border edges are part of the four and are handled by the same formula). So the O(1) perimeter update
is trustworthy.

Second, the topology gate — the part I was most likely to fool myself on. Every flip, targeted or
uniform, goes through `simple_point`, a crossing-number test on the 8-neighbourhood plus an explicit
diagonal-pinch rejection. I want to trace it on concrete configurations rather than assume it does
what I think. A diagonal-only touch — in-cells at `(0,0)` and `(1,1)` meeting only at a corner — should
be rejected, because the boundary there is a figure-8, not one cycle; testing `(1,1)` returns false,
good. An orthogonal bar `(1,1),(1,2)` should be fine; testing `(1,1)` returns true, good. Then I tried
a case I expected to pass and it didn't: a vertical bar `(0,1),(1,1),(2,1)`, testing the centre
`(1,1)`, returned **false**. My first instinct was that the guard was over-firing. So I traced the
crossing number by hand. The clockwise 8-neighbourhood sequence from the top is

```
seq (p2..p9): [1, 0, 0, 0, 1, 0, 0, 0]
trans (0->1 count): 2   -> simple iff trans==1
```

Foreground above (`p2`) and below (`p6`), nothing else, gives two separate `0→1` transitions, so
`trans = 2`. That is not the guard misfiring — it is the test correctly reporting that the centre cell
touches *two distinct* foreground arcs, so removing it would split the region into two disconnected
arms. The crossing number is literally counting connected foreground arcs around the cell, and `trans
= 1` is the condition that toggling keeps the region one piece. My expectation was wrong and the trace
corrected it; the gate is doing exactly its job. Since the targeted operator routes every proposal
through this same gate, a directed move that would self-intersect or disconnect is rejected like any
other — the operator only changes *which* legal moves get proposed, never whether illegal ones slip
through.

There is one more consistency I can pin down here without the full harness: the warm start. It picks
the best perimeter-feasible rectangle by 2D prefix sums of the per-cell weight `Av − Bv`, and the
running `curScore` is then `Σ(Av − Bv)` over in-region cells. For those to agree, the prefix-sum
rectangle weight has to equal a direct cell sum. On a small random grid I take a rectangle `[1,4)×[0,3)`
and compare the inclusion–exclusion `PS[i2][j2] − PS[i1][j2] − PS[i2][j1] + PS[i1][j1]` against the
literal double-loop sum:

```
prefix-sum rect weight: 1   direct sum: 1   match: True
```
