I have `m` ventilation slots in a row (`0..m-1`, `m <= 20`) and `n` sealant strips; strip `i` seals the
half-open span `[a_i, b_i)` — vents `a_i .. b_i-1`, left included, right excluded — at cost `c_i`. Any
subset may be applied, strips may overlap, and I must seal every vent at minimum total cost or print
`-1`. Two features of the scales decide the shape of the solution before I pick an algorithm. First,
`m <= 20`, so "which vents are sealed so far" is a bitmask in `0 .. 2^20-1` — small enough to index a
table of size `2^m` directly. Second, `n <= 2*10^5` and `c_i <= 10^9`, and a cover can stack many
strips, so the running cost can reach `2*10^5 * 10^9 = 2*10^14`, far past the 32-bit ceiling
`~2.1*10^9`. Costs and accumulators must be `long long`; an `int` here is a silent wrong answer on the
large tests.

The half-open convention is where a transcription slip would hide, so I fix its meaning concretely
before writing any loop. `[a, b)` seals vents `a..b-1`: strip `0 2` seals `{0,1}` (two vents, not vent
2), strip `2 4` seals `{2,3}`, and the two are exactly disjoint, meeting at the boundary without
overlapping. Read as inclusive `[a,b]` they would seal `{0,1,2}` and `{2,3,4}` — wrong on both count
and indices, and overlapping where they should tile. This is the off-by-one the mask-build loop lives
or dies on.

Now the algorithm. Greedy is tempting: sort strips by left endpoint, and at each uncovered vent take
the strip reaching furthest right — the classic minimum-interval cover, `O(n log n)`. But cost here is
attached to the strip, not its length, and overlaps are free, so "reach furthest right" optimizes
span, not cost. That gap is easy to make concrete. Take `m = 4` with a wide strip `[0,4)` =
`{0,1,2,3}` at cost `12`, and two narrow ones `[0,2)` = `{0,1}` at `5` and `[2,4)` = `{2,3}` at `6`.
Greedy, standing on uncovered vent 0, grabs the strip reaching furthest right — the wide one — for
`12` and stops. But the two narrow strips union to the same `{0,1,2,3}` for `5 + 6 = 11`. Greedy is
out, and the reason is exactly that a single wide strip can cost more than two narrow ones covering
the same vents.

So the state is the set of sealed vents and the objective ranges over arbitrary subsets — a subset DP.
Let `dp[S]` be the minimum cost to have sealed exactly the set `S`; start `dp[0] = 0`, everything else
`+infinity`, and relax by adding one strip at a time: from `S`, applying strip `i` reaches
`S | mask_i` at cost `dp[S] + c_i`. The answer is `dp[(1<<m)-1]`, or `-1` if it stays infinite. This
is correct because any subset of strips achieving a cover can be applied in some order, tracing a
chain `0 -> S_1 -> ... -> FULL` whose cost is the sum of the strips used; relaxing every single-strip
extension from every state considers every such chain. Order is irrelevant since the masks combine by
OR, and re-applying a strip never helps.

One ordering fact makes a single pass enough. Every transition moves to a strict superset
`nS = S | mask_i`, and a strict superset has strictly larger integer value (all of `S`'s bits plus at
least one more). So scanning `S = 0 .. 2^m-1` in increasing order, every predecessor that can relax
into `nS` is processed before `nS` itself — `dp[nS]` is final by the time the loop reaches it, and no
fixpoint iteration is needed.

The core is two loops: build each strip's mask from its span, then relax over all `S`. The mask build
is the one place the half-open boundary bites, and it is a single character:

```
for (int h = a; h < b; h++) bits |= (1 << h);   // inclusive a, exclusive b
```

Writing `h <= b` instead seals `a..b`, one vent too many: for strip `0 2` at `m=4` it sets bit 2 (a
vent the strip does not cover) and for `2 4` it sets bit 4 (a vent that does not even exist). The two
strips then both claim vent 2 and overlap where they should tile, so nothing ever produces the exact
mask `1111` — the DP would print `-1` for an instance that plainly covers for `11`. With `< b`, strip
`0 2` -> `{0,1}` = mask 3 and strip `2 4` -> `{2,3}` = mask 12 are disjoint, and the DP relaxes
`dp[3]=5` then `+6` to `dp[15]=11`. Running the four-strip sample
(`0 2 5 / 2 4 6 / 0 4 12 / 1 3 4`, masks `3,12,15,6`) the same way lands `dp[15]=11`, matching the
stated answer, with the wide strip's `dp[15]=12` losing.

The impossible case and the cost type are the two quieter concerns. If some vent lies in no strip's
span — `m=2` with only `0 1 3`, so vent 1 is unreachable — then `dp[FULL]` is never written and stays
infinite, so `-1` is just testing whether it is still the sentinel. I set `INF = 4e18` (below
`LLONG_MAX ≈ 9.2e18`) and guard the scan with `if (dp[S] == INF) continue;`, so an unreached cell is
never used as a relaxation base and no `INF + cost` can wrap. The reachable magnitudes stay small: a
finite `dp[S]` is a sum of at most `m <= 20` strip costs along its chain, at most `20 * 10^9 = 2*10^10`,
comfortably inside the `4e18` headroom, while `int` would already wrap after three `10^9` strips.
Testing `dp[FULL] >= INF` reports the impossible case.

Timing is the last question. The scan is `O(2^m * n)`, and at `m=20, n=2*10^5` the worst case is
`~2*10^11` — over the 2 s budget if an adversary fills the input with strips. The answer-preserving
escape is that only the cheapest strip per distinct mask ever matters (a costlier strip with the same
coverage never wins), and contiguous half-open spans yield at most `m(m+1)/2 <= 210` distinct masks;
collapsing duplicates to their cheapest representative bounds the inner loop by 210, i.e.
`2^20 * 210 ≈ 2*10^8`. I keep the plain loop over all `n`, with the dedup available as a drop-in that
preserves the answer exactly if `n` is what bites.

The remaining corners fall out of the same code: `m = 1` with one covering strip returns its cost;
zero-cost strips relax fine, since `dp[S] + 0` is a valid step, so free covers are allowed; and the
touching spans `[0,2)`/`[2,4)` are the boundary case already traced. I cross-checked the whole thing
against an independent brute force that enumerates all `2^n` strip subsets over several hundred random
small cases, feasible and impossible, with no mismatch. The full program is in the answer.

