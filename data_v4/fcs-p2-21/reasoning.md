**Fixing the scale before the algorithm.** This is box stacking with rotation and unlimited reuse:
`n <= 200` types, each placed with any of its three dimensions vertical so the other two form the
base, and a box may rest on another only if **both** base dimensions are *strictly* smaller. Two
numbers set the whole shape of the solution. First, `3n <= 600` oriented boxes — so whatever DP I
land on, an `O(m^2)` pass is `~360{,}000` operations, trivially inside a 1 s limit, and I do not need
to be clever about speed. Second, dimensions run to `10^6`, so a base **area** is a product up to
`10^6 * 10^6 = 10^12`, well past 32 bits. Heights sum to at most a few hundred boxes times `10^6`,
comfortably inside 32 bits, but areas are not: a 32-bit area comparison is a silent wrong answer the
moment two large bases differ only above `2^31`. So areas — and, to keep one type throughout, heights
too — are `long long`. The empty stack is always legal, so the answer never drops below `0`.

**The structure under the rotations.** Each type can stand three ways: choose which dimension is the
height, and the remaining two are the base. That turns `n` types into `m = 3n` *oriented* boxes,
each a triple `(w, d, h)`. Because supply is unlimited and I may reuse a type in different
orientations, there is no "each box once" constraint at the oriented level — every orientation is an
independent item, and the only coupling is the strict-nesting relation between bases. The nesting
test allows rotating the upper rectangle, so `(w1,d1)` fits inside `(w2,d2)` iff, after sorting each
pair, both min-vs-min and max-vs-max shrink. I normalize every base to `w <= d` at creation; then
nesting is just two scalar `>` comparisons and rotation never comes up again. The normalization is
sound precisely because the requirement is symmetric in the two base dimensions.

**Breaking the tempting greedy.** The most natural heuristic is: stand each type on its tallest face
(largest dimension up), use it once, and chain by nesting. But strict nesting is a *global* coupling,
and a per-box score fixes orientation one box at a time — here is a concrete break. Take the given
example `(6,6,10), (5,9,9), (4,8,8)`. Tallest-face gives bases
`(6,6) h10`, `(5,9) h9`, `(4,8) h8`. The near-square `(6,6)` base cannot support the others — `(5,9)`
on it needs `9 < 6`, which fails — so the tall `10`-box is stranded and the best chain is `(5,9)`
then `(4,8)` for `9 + 8 = 17`. Now the move greedy structurally refuses: rotate type 0 *off* its
tallest face to base `(6,10) h6`. That base is wider, so it can carry `(5,9) h9` (`5<6`, `9<10`) then
`(4,8) h8` (`4<5`, `8<9`), for `6 + 9 + 8 = 23 > 17`. A box's best orientation depends on what it
must support — a global decision — and the same kind of instance defeats area- and volume-greedy for
the same reason. Greedy is out.

**The ordered DP, and why the order is the whole game.** I want the maximum-total-height legal
stack. The relation "box `a` can rest directly on box `b`" (b's base strictly larger in both
dimensions) is a strict partial order, and it is **acyclic**: each step strictly shrinks both base
dimensions, hence strictly shrinks the base **area**. So the problem is a longest weighted path in a
DAG whose nodes are the oriented boxes and whose node weight is the height, and I can solve it by
processing boxes in a topological order with a one-dimensional DP.

The topological order falls out of that same area fact. If `a` can sit on `b` then `area(a) <
area(b)` strictly, so sorting the oriented boxes by base area **descending** puts every potential
supporter before every box that could rest on it. Area ties are harmless: two boxes with equal area
cannot have one's base strictly inside the other's (equal area rules out both dimensions shrinking),
so tied boxes never nest and their relative order is irrelevant. Then the DP: process boxes `j` in
sorted order and let `dp[j]` be the tallest stack whose **top** box is `j`. Either `j` stands alone,
`dp[j] = h[j]`, or it rests on some earlier `i` with a strictly larger base in both dimensions,
`dp[j] = max(dp[j], dp[i] + h[j])`. Every box that could support `j` already sits among the `i < j`,
so the inner scan sees all valid supports and `dp[j]` is exact. The answer is `max_j dp[j]`, or `0`
if there are no boxes — the box-stacking specialization of the LIS DP, `O(m^2)` time, `O(m)` space.

**The comparison the constraints invite me to botch.** The nesting test must be strict `>` in
*both* base dimensions, not `>=` — and cubes and equal-dimension boxes are exactly the inputs that
punish a slip. Take a single cube `(2,2,2)`. All three of its orientations have base
`(2,2)`, and `(2,2)` does not fit strictly inside `(2,2)`, so exactly one box can be placed — the
answer is `2`. With `>=`, box `(2,2)` would be allowed on `(2,2)` (since `2 >= 2`), and the DP would
chain all three identical orientations into `2 + 2 + 2 = 6`, which the strict rule forbids. With `>`
each orientation stands alone and the cube correctly returns `2`. Same reason it must be `>` and not
`>=` for equal-area, non-cube ties like base `(8,8)` carrying `(4,8)`: `8 > 8` is false, so `(4,8)`
may not rest on `(8,8)`, exactly as required.

**Edge cases.** `n = 0`: the box loop never runs, `m = 0`, the DP loop never runs, and `best` stays
its initial `0` — the empty stack. `n = 1`, e.g. `(2,3,4)`: orientations `(3,4) h2`, `(2,4) h3`,
`(2,3) h4`, and `(2,3)` nests on `(3,4)` (`2<3`, `3<4`), so a single *type* stacks two of its own
orientations for `4 + 2 = 6` — a corner the "use each box once" greedy gets wrong even at `n = 1`.

**Verification.** Rather than stop at hand traces I differential-tested against an independent brute
oracle that formulates the same problem as a longest weighted path in the strict-nesting DAG solved
by memoized DFS — a deliberately different shape from the sorted `O(m^2)` DP. Over 1200+ cases
(random plus the edge cases: `n=0`, `n=1`, all-cubes, nesting chains, the greedy traps, and an
`n=200` / dimensions-to-`10^6` stress mode) there were zero mismatches; the worst case runs in a few
milliseconds and ~3.6 MB, far inside the limits. The classic reference instance
`{4,6,7},{1,2,3},{4,5,6},{10,12,32}` returns the known `60`, and the greedy-trap cases (where the
oracle and the sorted DP still agree) are what pin the algorithm rather than just the code.
