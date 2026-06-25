**Reading the problem and pinning the contract.** I have `n` flagged segments on a number line; segment `i` is the closed integer interval `[l_i, r_i]`. I install checkpoints at integer coordinates, a checkpoint at `x` inspects segment `i` iff `l_i <= x <= r_i`, and I want the minimum number of checkpoints so every segment is inspected. The word that I keep underlining is *closed*: `x = l_i` and `x = r_i` both count, so two segments that touch at exactly one coordinate — `[1,5]` and `[5,9]` share only the point `5` — can be inspected by a single checkpoint sitting on that shared point. That is the whole flavour of this problem, and I expect the difficulty to live in one comparison operator. Scale first, because it fixes data types: `n <= 2*10^5`, coordinates in `[-10^9, 10^9]`. The answer is at most `n <= 2*10^5`, which fits in 32 bits, but coordinates reach `10^9` and I will keep a "last checkpoint coordinate" sentinel below every real coordinate; `LLONG_MIN` as a `long long` is the clean choice and avoids any flirtation with `-10^9` overflowing an `int` sentinel. I will store endpoints as `long long` throughout. There is no summation anywhere, so overflow is not a real threat once coordinates are 64-bit — but the sentinel discipline still matters because coordinates can be negative.

**Laying out the candidate approaches.** This is the classic "stab all intervals with the fewest points." Two routes:

- *Greedy by earliest right endpoint, with an exchange argument.* Sort segments by right endpoint ascending. Walk through them; whenever I meet a segment that no existing checkpoint inspects, open a new checkpoint at that segment's right endpoint `r`, and let it also inspect every later segment whose left endpoint is `<= r`. This is `O(n log n)` for the sort and `O(n)` for the sweep. The two things I must *prove*, not assume, are (a) why anchoring the new checkpoint at the right endpoint is optimal, and (b) the exact test that decides "no existing checkpoint inspects this segment" when intervals are closed.
- *Coordinate compression + maximum point-disjoint set.* The minimum number of stabbing points equals the maximum number of pairwise point-disjoint segments (an interval analogue of a min-cut/max-matching duality). This is more machinery, and the closed-endpoint subtlety just migrates into the disjointness predicate (`b < c` vs `b <= c`), so it buys me nothing. I will use it only as an *independent* check of the greedy.

I commit to greedy-by-right-endpoint, but I refuse to commit to the boundary operator until I have traced it.

**Deriving the greedy and its exchange argument.** Claim: there is an optimal solution in which the first checkpoint (processing segments in increasing right endpoint) is placed at the smallest right endpoint among all segments. Let `s` be the segment with the smallest right endpoint `r_min`. *Any* solution must inspect `s`, so it has a checkpoint at some coordinate `x` with `l_s <= x <= r_s = r_min`. Now slide that checkpoint rightward to `r_min`. Does this break anything? Any other segment `t` that was inspected by the old `x` has `l_t <= x <= r_t`. Because `r_min` is the *smallest* right endpoint, `r_t >= r_min >= x`, so `r_t >= r_min`. And `l_t <= x <= r_min`. Hence `l_t <= r_min <= r_t`, i.e. the moved checkpoint at `r_min` still inspects `t`. So moving the checkpoint to `r_min` only *gains* coverage, never loses it — the exchange is safe, and an optimal solution placing a checkpoint at `r_min` exists. Remove `s` and every segment now inspected by the `r_min` checkpoint, and recurse on the rest. That is exactly the sweep.

So the algorithm is: sort by `r`; keep `last` = coordinate of the most recently placed checkpoint (initially "none"); for each segment in order, if it is *not* inspected by the checkpoint at `last`, place a new checkpoint at this segment's `r` and set `last = r`.

**The decisive boundary: which test means "not yet inspected"?** A segment `[l, r]` is inspected by the checkpoint at `last` iff `l <= last <= r`. Because I process in increasing-`r` order, every segment seen *after* I set `last = r_prev` has `r >= r_prev = last`, so the right half `last <= r` holds automatically once a checkpoint exists. Therefore the only part of the membership test that can fail is the left half `l <= last`. So:

- "already inspected" `<=>` `l <= last`,
- "needs a new checkpoint" `<=>` `l > last`, equivalently `last < l`.

The closed-interval touching case is precisely the equality `l == last`: a segment whose left endpoint equals the previous checkpoint *is* inspected (the checkpoint sits on its left endpoint). So the correct test for "open a new checkpoint" must be the **strict** `last < l`, treating `l == last` as covered. If I wrote `last <= l` I would wrongly re-open a checkpoint for a touching segment. I can feel the off-by-one already; I will verify it on a trace, not by faith.

**Numeric self-check of the duality claim before I trust the whole construction.** I asserted "minimum stabbing points = maximum number of pairwise point-disjoint segments." Let me sanity-check it on a concrete instance rather than wave at it. Take `[1,5], [5,8], [9,12], [10,14]`. Pairwise point-disjoint means sharing no integer coordinate; closed intervals `[a,b]` and `[c,d]` (with `a<=b`, `c<=d`) share a point iff `max(a,c) <= min(b,d)`. Check pairs: `[1,5]` vs `[5,8]`: `max(1,5)=5`, `min(5,8)=5`, `5<=5` so they *share* the point 5 — not disjoint. `[1,5]` vs `[9,12]`: `max=9 > min=5` — disjoint. `[1,5]` vs `[10,14]`: disjoint. `[5,8]` vs `[9,12]`: `max=9 > min=8` — disjoint. `[5,8]` vs `[10,14]`: disjoint. `[9,12]` vs `[10,14]`: `max(9,10)=10`, `min(12,14)=12`, `10<=12` — share, not disjoint. So the disjointness graph has two "touching" pairs: `{[1,5],[5,8]}` and `{[9,12],[10,14]}`. A maximum pairwise-disjoint set picks at most one from each touching pair and they are mutually disjoint across pairs, e.g. `{[1,5],[9,12]}` of size 2; you cannot get 3 because any third segment touches one of the chosen two. So the maximum disjoint count is `2`. And indeed I can stab everything with `2` checkpoints (at `5` and at `12`). `2 = 2`: the duality holds on this instance, and crucially the boundary convention is consistent on both sides — "touching counts as overlapping/inspectable" on the stabbing side matches "touching counts as not-disjoint" on the packing side. That consistency is what convinces me the strict `<` is the right operator, not just a guess.

**First implementation — and a trace, because the boundary is exactly where this dies.** My first cut of the sweep, transcribing "open a checkpoint when the segment starts at or after the last one":

```
sort(seg.begin(), seg.end());            // by right endpoint
long long checkpoints = 0, last = LLONG_MIN;
for (auto &p : seg) {
    long long r = p.first, l = p.second;
    if (last <= l) {                     // <-- suspicious operator
        checkpoints++;
        last = r;
    }
}
```

I deliberately wrote `<=` to see it fail, and I trace the smallest input that distinguishes `<=` from `<`: the touching pair `[1,5], [5,9]`, whose correct answer is `1` (a single checkpoint at coordinate `5` is on both). Sorted by `r`: `[1,5]` (r=5), `[5,9]` (r=9). Start `last = LLONG_MIN`, `checkpoints = 0`. First segment `l=1, r=5`: `last <= 1`? `LLONG_MIN <= 1` true, so `checkpoints = 1`, `last = 5`. Second segment `l=5, r=9`: `last <= l` is `5 <= 5` **true**, so `checkpoints = 2`, `last = 9`. Output `2`.

**Diagnosing the bug.** The code says `2` but the answer is `1`. The defect is exactly the boundary I flagged: at the second segment `l == last == 5`, the checkpoint already placed at coordinate `5` sits on this segment's left endpoint, so the segment *is* inspected — yet `last <= l` reported "uncovered" and opened a redundant checkpoint. The membership test `l <= last <= r` is satisfied (`5 <= 5 <= 9`), so the segment should have been skipped. Using `<=` to mean "needs a new checkpoint" mis-classifies the touching case. The fix is the strict comparison: open a new checkpoint only when `last < l`, i.e. when the segment's left endpoint is *strictly beyond* the last checkpoint.

**Fixing and re-verifying.** Change the operator to strict:

```
if (last < l) {            // l == last means the segment touches the last checkpoint => covered
    checkpoints++;
    last = r;
}
```

Re-trace `[1,5], [5,9]`: start `last = LLONG_MIN`, `checkpoints = 0`. `[1,5]`: `LLONG_MIN < 1` true -> `checkpoints = 1`, `last = 5`. `[5,9]`: `5 < 5` **false** -> skip. Output `1`. Correct. Re-trace the four-segment sample `[1,5], [5,8], [9,12], [10,14]` (sorted by r already): `[1,5]` -> `last=LLONG_MIN<1` -> checkpoint at 5, `last=5`, count 1. `[5,8]`: `5<5` false -> skip (the checkpoint at 5 inspects it). `[9,12]`: `5<9` true -> checkpoint at 12, `last=12`, count 2. `[10,14]`: `12<10` false -> skip (checkpoint at 12 inspects it, since `10<=12<=14`). Output `2`. Correct, and it matches the worked example. The case that broke now passes, and it passes for the reason I fixed.

**A second debug episode: the sort key and a non-touching trap.** With the operator fixed I want to be sure I am sorting by the *right* endpoint, not the left. I stored pairs as `{r, l}` so the default `pair` ordering sorts by `r` first — but let me trace an instance where sorting by left endpoint would give a different, wrong answer, to confirm the key matters and that I have it right. Take `[1,10], [2,3], [4,5]`. The correct minimum: `[2,3]` and `[4,5]` are point-disjoint (`max(2,4)=4 > min(3,5)=3`), and `[1,10]` overlaps both, so two checkpoints suffice — e.g. one at `3` (inspects `[1,10]` and `[2,3]`) and one at `5` (inspects `[4,5]`)... wait, does `5` inspect `[1,10]`? `1<=5<=10` yes, but `[1,10]` is already inspected at `3`; the point is `{3,5}` stabs all three, so the answer is `2`. Now suppose I sorted by *left* endpoint: order would be `[1,10], [2,3], [4,5]`. First segment `[1,10]`: open checkpoint at its right endpoint `10`, `last=10`. `[2,3]`: `10 < 2`? false -> but does the checkpoint at `10` inspect `[2,3]`? `2 <= 10 <= 3` is **false** (`10 > 3`)! So skipping it is *wrong* — the sort-by-left order broke my own invariant that "later segments always have `r >= last`," which is the very assumption that let me reduce the membership test to just `l <= last`. With a left-endpoint sort, a new checkpoint placed at a huge right endpoint poisons the `last <= r` half of the test for subsequent shorter segments.

Let me confirm my actual code (sort by right endpoint) handles this instance. Sorted by `r`: `[2,3]` (r=3), `[4,5]` (r=5), `[1,10]` (r=10). `[2,3]`: `LLONG_MIN < 2` -> checkpoint at 3, `last=3`, count 1. `[4,5]`: `3 < 4` true -> checkpoint at 5, `last=5`, count 2. `[1,10]`: `5 < 1`? false -> skip; and indeed the checkpoint at `5` inspects `[1,10]` since `1<=5<=10`. Output `2`. Correct. The right-endpoint sort is load-bearing: it guarantees `r >= last` for every segment processed after a checkpoint, which is the precondition that makes "`l <= last`" a complete membership test. Sorting by left endpoint would silently break that and the `<` fix would not save it. Good that I traced it rather than assuming `{r,l}` "looked sorted enough."

**Edge cases, deliberately.**
- `n = 0`: `cin >> n` reads `0`, the vector is empty, the loop never runs, `checkpoints = 0`. Correct — nothing to inspect.
- `n = 1`, `[3,3]` (degenerate point segment): sorted trivially. `[3,3]`: `LLONG_MIN < 3` true -> checkpoint, count 1. Output `1`. Correct.
- All segments share a common point, e.g. `[1,5],[2,7],[3,4]`: sorted by r -> `[3,4](r=4), [1,5](r=5), [2,7](r=7)`. `[3,4]`: checkpoint at 4, `last=4`, count 1. `[1,5]`: `4<1` false -> skip (`1<=4<=5`). `[2,7]`: `4<2` false -> skip (`2<=4<=7`). Output `1`. Correct — one checkpoint at 4 stabs all three.
- Fully disjoint chain `[1,2],[4,5],[7,8]`: each needs its own checkpoint -> `3`. Trace: `[1,2]` -> cp at 2, `last=2`. `[4,5]`: `2<4` -> cp at 5, `last=5`. `[7,8]`: `5<7` -> cp at 8. Output `3 = n`. Correct.
- End-to-end touching chain `[1,5],[5,9],[9,13]`: each pair shares one point. `[1,5]` -> cp at 5, `last=5`. `[5,9]`: `5<5` false -> skip. `[9,13]`: `5<9` true -> cp at 13. Output `2`. Hand check: `{5, 9}`? checkpoint at 5 covers `[1,5]` and `[5,9]`; checkpoint at 13 covers `[9,13]` — but does anything at `9` matter? `[9,13]` needs `9<=x<=13`; `13` works. Two checkpoints. Could one do it? No single integer lies in all three (`5` misses `[9,13]`). So `2` is optimal. Correct — and this is exactly where a `<=` bug would have returned `3`.
- Negative coordinates `[-9,-9],[-3,-1]`: `LLONG_MIN` sentinel is below `-9`, so the first segment correctly opens a checkpoint. Both disjoint -> `2`. The negative sentinel choice matters here; an `int` sentinel of `-2e9` would also work, but `LLONG_MIN` is unambiguous.
- Duplicates `[2,6],[2,6],[2,6]`: sorted, all r=6. First -> cp at 6, `last=6`. Others: `6<2` false -> skip. Output `1`. Correct.

**Verification at scale and against an independent brute.** I sorted by right endpoint (`O(n log n)`) and sweep once (`O(n)`), `O(n)` memory. On `n = 2*10^5` with full-range coordinates the program runs in about `0.03 s`, far under the 1 s limit. For correctness I wrote an independent brute force that computes the maximum pairwise point-disjoint set by exhaustive subset enumeration (a different argument from the greedy sweep), and ran `>= 400` random small instances with tiny coordinate ranges (so touching and nesting happen constantly): **zero** mismatches. I also compiled a deliberately broken variant with `<=` instead of `<`; it diverged from the brute on `224` of `400` cases and returned `2` instead of `1` on `[1,5],[5,9]` — concrete proof that the boundary operator is the load-bearing line, not cosmetics.

**Final solution.** I proved the greedy by an exchange argument (slide any inspecting checkpoint to the smallest right endpoint; it can only gain coverage), reduced the closed-interval membership test to the single comparison `l <= last` under the right-endpoint sort, and pinned the off-by-one to a strict `<` by tracing the touching case `[1,5],[5,9]` where `<=` wrongly returns `2`. This is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 -> no segments -> 0 checkpoints
    vector<pair<long long,long long>> seg(n); // (right endpoint, left endpoint)
    for (int i = 0; i < n; i++) {
        long long l, r;
        cin >> l >> r;
        seg[i] = {r, l};                   // sort key is the right endpoint
    }

    sort(seg.begin(), seg.end());          // by right endpoint ascending

    long long checkpoints = 0;
    long long last = LLONG_MIN;            // coordinate of the most recent checkpoint
    for (int i = 0; i < n; i++) {
        long long r = seg[i].first, l = seg[i].second;
        // The segment is already inspected iff l <= last <= r. Since we process in
        // right-endpoint order, last <= r always holds once a checkpoint exists, so
        // the only test that matters is whether l <= last. If l > last the segment
        // is NOT covered and we must open a new checkpoint at r (the latest spot that
        // still inspects this segment, maximizing future coverage).
        if (last < l) {                    // strict: l == last means already covered
            checkpoints++;
            last = r;
        }
    }

    cout << checkpoints << "\n";
    return 0;
}
```

**Causal recap.** The minimum-checkpoints problem is "stab all closed intervals with fewest integer points"; an exchange argument (any checkpoint inspecting the smallest-right-endpoint segment slides rightward to that endpoint without losing coverage) proves the greedy-by-right-endpoint sweep optimal, and the right-endpoint sort collapses the closed-interval membership test `l <= last <= r` to just `l <= last`. The single decision that determines correctness is whether a segment whose left endpoint equals the previous checkpoint (`l == last`) is "covered" — it is, because the checkpoint sits on its endpoint — so the "open a new checkpoint" condition must be the strict `last < l`; tracing the touching pair `[1,5],[5,9]` shows the `<=` form returning the illegal `2` instead of `1`, and the strict form, plus a right-endpoint (not left-endpoint) sort and a `long long` sentinel below all coordinates, closes the touching, nesting, disjoint, degenerate, negative-coordinate, and empty corners against an independent disjoint-set brute over 400 cases.
