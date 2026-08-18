The cost I minimize over orderings has two layers, and the second one is the whole difficulty. Besides
the adjacency cleaning cost `c[prev][cur]` on every consecutive pair, each batch also pays
`e[prevprev][cur]` to the batch sitting **two** positions before it. So the ordering interacts not just
with its neighbor but with its neighbor's neighbor, and any DP that only remembers the neighbor is going
to be blind to half the cost.

Before any algorithm, the scale, because it fixes the data type. With `n <= 16` and both `c` and `e` up
to `10^9`, an order has `n - 1` adjacency terms and `n - 2` carry-over terms, so the total can reach
`15*10^9 + 14*10^9 = 2.9*10^10` — more than ten times the signed 32-bit ceiling `~2.147*10^9`. Every
cost accumulator must be `long long`; an `int` is a silent wrong answer on the large tests, and the
constraints are clearly drawn to trip exactly that.

`16!` is about `2*10^13`, so enumeration is dead. Minimizing a sum of transition-like costs over an
ordering of a small set is bitmask-DP territory: build the order one batch at a time, keyed by a mask of
which batches are placed, with `popcount(mask)` tracking progress. The only real question is what *else*
the state must carry to charge each cost term exactly once.

The reflex is the Held-Karp open-path DP: `dp[mask][last]` = cheapest partial order over `mask` ending at
`last`, extended by appending an unused `nxt` for `c[last][nxt]`. That is `O(2^n n^2)` and provably right
when the cost is a sum of pairwise terms between *consecutive* elements, because then the only thing the
future needs from the past is the last placed element. But my carry-over term reaches two positions back,
and a `(mask, last)` state has already thrown the two-back batch away. The tempting patch is to fold the
penalty into the transition using the only previous-ish thing in hand — add `e[last][nxt]` when appending
`nxt`. That charges `e` from *one* back rather than two, and it charges it on the second batch, which by
the contract owes no carry-over at all yet.

**First trace — and the baseline is already suspect on paper.** Walk a length-3 order `x, y, z`. When I
append `y` after `x` the code adds `e[last][nxt] = e[x][y]`. But by the problem, `y` is only *one*
position after `x`, so `y` should pay **no** carry-over at all yet — carry-over starts at the third
batch. And when I append `z` after `y` the code adds `e[y][z]`, charging the carry-over from `y` (one
back) onto `z`, whereas the problem charges `e[x][z]` from `x` (two back). So the hacked baseline pays
the wrong `e` on the second batch (it should pay none) and the wrong `e` on the third (it uses `y`
instead of `x`). On paper this looks broken; let me confirm it numerically against brute force rather
than trust the paper argument.

**Numeric self-check of the baseline against brute force.** Take the documented sample, `n = 3`.

Brute force over all 6 orders, charging exactly the problem's definition (adjacency on consecutive
pairs, carry-over `e[order[t-2]][order[t]]` for `t >= 2`):

The patch fails on the sample. With `n = 3`, brute force over the six
orders puts the optimum at `12`, order `2,1,0`: adjacency `c[2][1] + c[1][0] = 4 + 5 = 9` plus the
two-back carry-over `e[2][0] = 3`; every other order costs at least `13`, and nothing costs less than
`12`. Now walk the folded baseline along that same order: appending `1` after `2` it adds
`c[2][1] + e[2][1] = 4 + 0`, then appending `0` after `1` it adds `c[1][0] + e[1][0] = 5 + 0`, total
`9`. It omits the real `e[2][0] = 3` because the two-back batch of `0` is `2`, which the state no longer
holds — and it would report `9` as a global minimum that no order actually achieves. The failure is
state insufficiency, not an arithmetic offset: two histories with the same `(mask, last)` can have
different two-back batches and owe different carry-overs, yet the baseline collapses them into one cell.
No transition-time trick recovers information the state doesn't carry. The state has to widen to the last
two batches.

So let `dp[mask][last][prev]` be the cheapest partial order over exactly `mask`, ending `..., prev,
last`. Appending an unused `nxt` pays `c[last][nxt]` for the new adjacency and `e[prev][nxt]` for the
carry-over from the batch now two positions back, landing in `dp[mask | (1<<nxt)][nxt][last]`. The
boundary is the first two batches: the first pays nothing, the second pays adjacency only, since no
two-back batch exists yet. I encode "one batch placed so far" as `prev == last`, seeding
`dp[1<<s][s][s] = 0`; extending such a state places the *second* batch, so I add only `c[last][nxt]` and
skip `e`. Once `prev != last` a genuine two-back batch exists and `e[prev][nxt]` applies. The answer is
the minimum full-mask cell. Tracing `2,1,0`: from `dp[{2}][2][2] = 0`, append `1` (single marker, add
`c[2][1] = 4`) to `dp[{1,2}][1][2] = 4`, then append `0` (distinct now, add `c[1][0] + e[2][0] = 8`) to
`dp[{0,1,2}][0][1] = 12` — matching the brute-force optimum and charging the carry-over from the correct
two-back `2`.

The width is affordable. States number `2^n * n^2 = 65536 * 256 ~ 1.68*10^7` for `n = 16`, one
`long long` each, about `134` MB — inside the `256` MB budget but tight enough that I won't duplicate the
table. Transitions are `2^n * n^3 = 2.68*10^8` relaxations, each a couple of adds and a compare, well
under two seconds. I flatten `(last, prev)` into a single `n*n` dimension so the table is a clean `2^n`
by `n^2` array rather than a jagged three-level structure.

I trace the smallest input that exercises a real two-back charge, `n = 3` with the sample matrices,
following order `2,1,0`. Step one: from `dp[{2}][2*3+2] = 0`, append `1`. New most-recent batch is `1`,
and the batch before it is `2`, so the correct new index is `last_new=1, prev_new=2`, i.e.
`1*3 + 2 = 5`. But my code wrote `dp[nmask][prev*n + last] = dp[nmask][2*3 + 1] = dp[nmask][7]`, which
decodes as `last_new=2, prev_new=1` — the two batches **swapped**. So the next time I read this cell I
would think the order ends `..., 2, 1` when it actually ends `..., 1, 2`, and I would charge `c[2][...]`
where I should charge `c[1][...]`. The result is a corrupted chain; on the sample this produced a wrong
total (I got a value that did not match the brute-force `12`).

One encoding hazard in this flattened layout is worth stating precisely, because it is easy to get
backwards. After appending `nxt` the order ends `..., last, nxt`, so the destination cell must index
`(nxt, last)` — new most-recent is `nxt`, new two-back is `last`, i.e. `dp[nmask][nxt*n + last]`.
Mechanically reusing the old `prev` and writing `dp[nmask][prev*n + last]` swaps the two slots: every
later read of that cell would believe the order ends `..., last, prev` and charge `c[last][...]` where it
should charge `c[nxt][...]`, corrupting the chain. The source reads `(last, prev)`; the destination
writes `(nxt, last)`. Keeping those two encodings distinct is the one place this DP genuinely differs
from single-slot Held-Karp bookkeeping.

**A second debug episode: the carry-over fires one step too early.** While re-reading the code I worried
about the `single` flag, so I traced a different order, `0,1,2`, whose true cost is `19` (adjacency
`5+4=9`, carry `e[0][2]=10`). Start `dp[{0}][0*3+0]=0` (single marker `prev==last==0`). Append `1`:
because `prev==last` I must treat this as placing the *second* batch and add **no** carry-over, only
`c[0][1]=5`, reaching `dp[{0,1}][1*3+0]=5`. I checked the condition: `single = (prev == last) = (0 ==
0) = true`, so `add = c[0][1]` only — good, no `e` charged. (An earlier mental version of mine had used
`if (popcount(mask) >= 2) add += e[...]`, which is equivalent, but the `prev==last` marker is cleaner
because it lives in the state and survives flattening.) Append `2`: now `prev=0, last=1`, distinct, so
`single=false`, `add = c[1][2] + e[0][2] = 4 + 10 = 14`, reaching `dp[{0,1,2}][2*3+1] = 5 + 14 = 19`.
That matches the brute-force `19` for this order, and crucially the carry-over `e[0][2]` fired exactly
when the third batch was placed, charged from the correct two-back batch `0`. Had I forgotten the
`single` guard, the second batch would have wrongly paid `e[0][1]`, inflating every order. The guard is
load-bearing and the trace confirms it triggers at the right step.

The corners fall out. `n = 0` and `n = 1` short-circuit to `0` — no transitions, no carry-over — via the
empty-input guard and explicit branches. `n = 2` reaches the full mask with `single == true` throughout,
adding one adjacency term and no `e`, so the answer is `min(c[0][1], c[1][0])`. When `e` is all zero
every `e[prev][nxt]` vanishes and the DP degenerates to plain Held-Karp open-path TSP, so a correct
pure-adjacency answer is a special case. Diagonal entries `c[i][i]`, `e[i][i]` are read but never used,
since `nxt` is always distinct from `last` and `prev`. Every accumulator is `long long`, and the `INF`
sentinel (`4e18`) is skipped by `if (cur >= INF) continue;` before any addition, so it never enters
arithmetic and cannot overflow.

So the program is a single self-contained C++17 file running the `O(2^n n^3)` two-back bitmask DP over
the widened `(last, prev)` state. The full module is in the answer.
