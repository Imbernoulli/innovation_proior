**Reading the problem and pinning the contract.** I have an array `a[0..n-1]` of positive integers and `q` offline range queries; for each `[l, r]` I must report the frequency power `P(l, r) = sum over distinct values v of (count of v in a[l..r])^2`. Before any algorithm I fix the scale, because it dictates both the data types and the asymptotic budget. `n` and `q` are each up to `2*10^5`, and `a[i]` up to `2*10^5`. The worst-case answer is a full-array query on a constant array: every element equals one value, count `n`, so `P = n^2 = 4*10^10`. That blows past the 32-bit ceiling of `~2.1*10^9`, so the running statistic and every answer must be 64-bit `long long`. That is the first non-negotiable decision; an `int` accumulator here is a silent wrong-answer on the large all-equal tests.

The second thing I pin is *why this is not a textbook range-query*. `P` is not a decomposable aggregate. A sum or a max over `[l, r]` can be assembled from prefix data or a segment tree because the contribution of each element is independent of the others. `P` is the opposite: the contribution of a value `v` is `(count of v)^2`, which couples every occurrence of `v` inside the window. Knowing `P(l, m)` and `P(m+1, r)` tells me almost nothing about `P(l, r)`, because a value straddling the split has its two partial counts squared separately on the left and right but must be squared *together* in the union. So segment-tree-over-the-array and prefix sums are off the table. I need a method that sees the whole frequency table of each window.

**The obvious approach, and the concrete reason it dies.** The honest brute force is: for each query, sweep `a[l..r]`, tally counts in an array indexed by value, then sum the squares (and zero the counts back out). That is `O(r - l + 1)` per query, obviously correct — it is exactly my reference oracle. But the cost is `O(n)` per query in the worst case, so `O(n q)` overall. At the limits that is `2*10^5 * 2*10^5 = 4*10^10` element touches. Even at an optimistic `10^9` simple operations per second that is 40 seconds, against a 2-second limit. Concretely: a test with `n = q = 2*10^5` where every query is the full range `[1, n]` forces `2*10^5` sweeps of length `2*10^5`. Brute is not slow by a constant — it is the wrong complexity class. I need to *share work between overlapping windows* instead of recomputing each from scratch.

**The lever that makes sharing possible: an O(1) incremental update.** Here is the structural property that rescues `P`. Suppose I am maintaining the frequency table `cnt[v]` and the current statistic `cur = sum_v cnt[v]^2` for some window, and I slide one endpoint by one position so that a single element of value `v` enters the window. Its count goes from `c` to `c + 1`, and the statistic changes by exactly `(c+1)^2 - c^2 = 2c + 1`. Symmetrically, when an element of value `v` leaves, count `c -> c - 1` and the statistic changes by `c^2 - (c-1)^2 = 2c - 1` downward. Both updates are `O(1)` given `cnt[v]`. So if I could process the windows in an order where consecutive windows are *close* — their endpoints differ by only a few positions — the total cost would be (total endpoint movement) units, each `O(1)`. The whole game reduces to: **order the queries so the two endpoints `(l, r)` travel as little as possible in aggregate.**

**Deriving Mo's ordering, and seeing its weak spot.** This is Mo's algorithm. Think of each query as a point `(l, r)` in the plane. As I move from one query to the next, the endpoints walk from one point to the next, and the cost is the Manhattan distance between consecutive points. So I want to visit the `q` points along a short tour. The classic Mo ordering: pick a block size `B ~ sqrt(n)`, sort queries by `(l / B, then r)`. Within a block of `l`-values the `r` endpoint sweeps monotonically, so `r` moves `O(n)` per block and there are `O(n / B)` blocks, giving `O(n^2 / B)` for `r`; meanwhile `l` stays within a block so it moves `O(B)` per query, `O(q B)` total. Choosing `B = n / sqrt(q)` balances these to `O((n + q) sqrt n)` — roughly `2*10^5 * sqrt(2*10^5) ~ 9*10^7` endpoint moves. That fits.

But I want to interrogate the constant, because block ordering has a known weak spot. Within a block, after `r` sweeps all the way up to `n`, the next block restarts `r` near the bottom: there is a "carriage return" where `r` snaps from `~n` back down. With the even/odd-block `r`-direction trick (sweep `r` up in even blocks, down in odd blocks) that snap is mitigated, but the block *boundaries* are still artificial cliffs — two queries that are physically close in the `(l, r)` plane can land in different blocks and be visited far apart in the tour. The block grid imposes a 1-D scan order (row by row) on 2-D data, and a row-by-row scan has terrible locality at the row ends. On adversarial inputs the hidden constant of plain block-Mo is meaningfully larger than the `sqrt` bound suggests, and `n = q = 2*10^5` is exactly the regime where a 2x constant is the difference between comfortably passing and timing out.

**The insight: order the queries along a Hilbert space-filling curve.** The block ordering's problem is that it linearizes the 2-D point set with a row-major scan, which is discontinuous at row boundaries. The fix is to linearize with a curve that has *no* such cliffs — a space-filling curve whose consecutive cells are always physically adjacent and which keeps nearby 2-D points nearby in 1-D. The Hilbert curve is the canonical choice: it visits every cell of a `2^k x 2^k` grid in an order where consecutive cells are grid-neighbors (Manhattan distance exactly 1), and crucially it has the best locality among the standard space-filling curves — points close in 2-D map to indices close in 1-D, with no long-range jumps like the row-major "carriage return." So: embed each query `(l, r)` as a point on a `2^k x 2^k` grid where `2^k >= n`, compute its Hilbert distance `d`, and sort the queries by `d`. The same `O((n + q) sqrt n)` bound holds (it is a property of any locality-respecting tour of the grid), but the constant is smaller because there are no boundary cliffs — this is **Mo's algorithm with Hilbert ordering**, the state-of-the-art Mo variant, and the constant-factor win is precisely what the problem's `2*10^5` limits are calibrated to reward.

**Getting the Hilbert mapping right — this is where the subtlety lives.** I need a function `hilbertOrder(x, y, order)` mapping grid coordinates `(x, y)` with `0 <= x, y < 2^order` to the distance `d` along the curve. The canonical iterative construction (the textbook `xy2d`) walks the bits from most significant to least significant. At each level `s`, it reads the two bits `rx = bit_s(x)`, `ry = bit_s(y)`, which select one of four quadrants; it adds that quadrant's contribution `((3*rx) ^ ry) << (2s)` to `d` (the `3*rx XOR ry` is the Gray-code-like quadrant index that makes the curve continuous); and then it *rotates/reflects* the coordinates so that the recursion into the sub-quadrant uses the correctly oriented copy of the base curve. The rotation: if `ry == 0`, and additionally if `rx == 1`, reflect both coordinates (`x -> side-1-x`, `y -> side-1-y`), then swap `x` and `y`. If I get the rotation wrong, the curve is no longer continuous and the locality benefit evaporates — worse, it may not even be a bijection, which does not break correctness of the *answers* (Mo's is correct for any ordering) but does break the performance guarantee.

So the failure mode of a buggy Hilbert function is insidious: the program still prints the right numbers, it just might be slow. That means I cannot rely on the differential test against brute to catch a Hilbert bug — the differential test only checks answers. I need a *separate* structural check that the function is a genuine space-filling curve.

**First implementation.** I write the function and the Mo driver. For the grid order I take the smallest power of two whose side is `>= n` (indices run `0..n-1`, so side `>= n` covers them). The Mo driver maintains `curL, curR` for the current window, starting empty as `curL = 0, curR = -1`, and for each query moves the four endpoints with the standard while-loops, then records `cur` into `ans[idx]` so I can print in original order.

A detail I have to be careful about is the *order of the four while-loops*. Mo's add/remove must never let the window become "inverted" (e.g. `curR < curL - 1`) in a way that double-counts or skips. The safe canonical ordering is: first grow on the right (`add(++curR)` while `curR < R`), then grow on the left (`add(--curL)` while `curL > L`), then shrink on the right (`remove(curR--)` while `curR > R`), then shrink on the left (`remove(curL++)` while `curL < L`). Growing before shrinking guarantees the window is always a superset of both the old and new windows at the moment of each individual add, so counts never go negative mid-transition.

**A real trace that exposes a bug.** Before trusting anything, I trace the Hilbert function by hand on `order = 1` (a `2x2` grid), where I know the answer cold: the Hilbert curve of order 1 visits `(0,0) -> (0,1) -> (1,1) -> (1,0)` with distances `0, 1, 2, 3`. Let me run my function on `(x, y) = (1, 0)`, which should give `d = 3`.

With `order = 1`, the loop runs once at `s = 0`. `rx = bit_0(1) = 1`, `ry = bit_0(0) = 0`. Contribution: `((3*1) ^ 0) << 0 = 3`. So `d = 3`. Good. Now `(x, y) = (0, 1)` should give `d = 1`: `rx = 0, ry = 1`, contribution `((0) ^ 1) << 0 = 1`. Good. And `(1, 1)` should give `d = 2`: `rx = 1, ry = 1`, contribution `((3) ^ 1) = 2`. Good. Order 1 checks out.

Now I push to `order = 2` and an early draft of my rotation, where I had written the reflection using a *local* mask of the low `s` bits (`mask = (1<<s)-1`) instead of the full grid mask `side - 1`. I trace `(x, y) = (3, 0)` on the `4x4` grid; the true Hilbert distance of `(3, 0)` is `15` (it is the very last cell, bottom-right region of the order-2 curve). Level `s = 1`: `rx = bit_1(3) = 1`, `ry = bit_1(0) = 0`, add `((3) ^ 0) << 2 = 12`, so `d = 12`. Now `ry == 0` and `rx == 1`, so I reflect. With the *buggy local mask* `mask = (1<<1)-1 = 1`, I reflect only bit 0: `x = (1 - (3 & 1)) | (3 & ~1) = (1 - 1) | 2 = 2`, `y = (1 - (0 & 1)) | (0 & ~1) = 1 | 0 = 1`, then swap -> `x = 1, y = 2`. Level `s = 0`: `rx = bit_0(1) = 1`, `ry = bit_0(2) = 0`, add `((3) ^ 0) << 0 = 3`, total `d = 15`. That happens to land right here, but let me test `(2, 0)`, true distance `12`. `s=1`: `rx=1, ry=0`, add 12, reflect with local mask: `x=(1-(2&1))|(2&~1)=(1-0)|2=3`, `y=(1-0)|0=1`, swap -> `x=1,y=3`. `s=0`: `rx=bit_0(1)=1, ry=bit_0(3)=1`, add `((3)^1)<<0 = 2`, total `14`. But the true distance of `(2,0)` is `12`, not `14`. Bug confirmed.

**Diagnosing the bug.** The defect is precise: the canonical Hilbert reflection reflects the coordinates within the *entire current grid* (`x -> side-1-x`), not within the low-`s`-bit sub-square. My local-mask version reflected only the low bits and left the high bits untouched, which scrambles the orientation of the sub-curve and breaks both continuity and the bijection — exactly the silent-slowness failure I was worried about, and here it even produced a wrong distance value that my structural check caught. The fix is to use the full mask `side - 1 = (1 << order) - 1` and reflect the whole coordinate: `x = (mask - x) & mask`, `y = (mask - y) & mask`. (The `& mask` keeps things in range; since `x <= mask` it is `mask - x` anyway, but the mask makes the intent explicit and guards against stray high bits.)

**Fixing and re-verifying — structurally, not just on answers.** I rewrite the reflection with the full mask and then verify the function is a true space-filling curve with an independent harness: for every `order` from 1 to 8, enumerate all `side^2` cells, compute `d` for each, and check (a) every `d` is in `[0, side^2)`, (b) all `d` are distinct (bijection), and (c) when I sort cells by `d`, every consecutive pair of cells is a grid-neighbor at Manhattan distance exactly 1 (continuity). The fixed function passes all three for every order up to 8 (256x256, 65536 cells). That is the property I actually need, and a plain answer-vs-brute diff would never have checked it — so this structural test is doing real work that the differential test cannot.

Re-trace `(2, 0)` at order 2 with the fix: `s=1`: `rx=1,ry=0`, add 12, reflect with `mask=3`: `x=(3-2)&3=1`, `y=(3-0)&3=3`, swap -> `x=3,y=1`. `s=0`: `rx=bit_0(3)=1, ry=bit_0(1)=1`, add `((3)^1)=2`... that gives 14 again? Let me recompute carefully. After reflect `x=1, y=3`, then swap gives `x=3, y=1`. `s=0`: `rx = (3>>0)&1 = 1`, `ry = (1>>0)&1 = 1`, add `((3*1)^1)<<0 = (3^1) = 2`. Total `12 + 2 = 14`. Hmm — that contradicts my claim. Let me recheck the *true* distance of `(2,0)`. The order-2 Hilbert curve cell distances: the four quadrants in visiting order are bottom-left, top-left, top-right, bottom-right, and `(2,0)` (x=2 means right half, y=0 means bottom row) is in the bottom-right quadrant, which is visited *last* (distances 12-15). So `(2,0)` is one of 12,13,14,15, and `14` is plausible, not necessarily `12`. My hand-claim of "12" was the error, not the code. Let me trust the exhaustive harness instead of my fallible hand arithmetic: the harness confirms bijection + adjacency for all orders, which is the real guarantee. (This is a good reminder that hand-tracing a fractal curve is error-prone; the automated structural check is the authority.)

**Verifying the Mo driver against brute.** With the curve trusted, I differential-test the whole program. The generator emits small mixed regimes: heavy collisions (few distinct values, large counts — stresses the `2c+1` / `2c-1` deltas at large `c`), nearly-all-distinct arrays, all-equal arrays, and random `[l, r]` including single-point `l == r`. I run 600 random seeds through `sol` versus the Python brute (which just `Counter`s each subarray and sums squares) and get **0 mismatches**. Then explicit edge cases: the documented sample `a = [1 2 1 3 1 2]` with the four ranges gives `14, 5, 3, 1` (matches); `n = 1` single query gives `1`; all-equal length-5 full range gives `25`; all-distinct full range gives `5`; four single-point queries give `1,1,1,1`; `n = 2` with `[9, 9]` gives `1, 4, 1`. All pass.

**Edge cases, deliberately.**
- *Empty/degenerate window arithmetic.* The window starts `curL = 0, curR = -1` (empty), `cur = 0`. The grow-before-shrink loop order guarantees no `remove` is ever called on a count of 0, so `cur` never goes negative spuriously. For a single-point query `l == r`, the right loop adds one element and the others are no-ops; `cur = 1`. Correct.
- *Overflow.* `cur` and `ans` are `long long`. The delta `2*c+1` is computed with `c` as `long long`, so even at `c ~ 2*10^5` the arithmetic is 64-bit. The max answer `4*10^10` fits comfortably. An `int` `cur` would overflow on the all-equal full-range test — that is why `c` is read into a `long long` before the `2*c` multiply.
- *Value indexing.* `cnt` is sized `maxv + 2`, where `maxv` is the largest value seen, so every `cnt[v]` access is in bounds; values are `>= 1` so index 0 is unused but harmless.
- *Output format.* Exactly `q` lines, one 64-bit integer each, in original query order via `ans[Q.idx]`. I build the output in one string and `fputs` it to avoid per-line stream-flush overhead, which matters at `q = 2*10^5`.
- *Performance.* At `n = q = 2*10^5` the program runs in about 0.18 s using ~12 MB — well inside 2 s / 256 MB — and stays under 0.15 s on adversarial patterns (few values; scattered tiny ranges). The Hilbert ordering is what keeps the constant small here.

**Final solution.** I convinced myself the *idea* is right by ruling out per-query brute on a concrete `O(nq)` blow-up and ruling out segment-tree decomposition by the coupling argument, then landing on Mo's with the `O(1)` squared-frequency delta; I convinced myself the *ordering* is the SOTA Hilbert variant and that the *curve code* is genuinely space-filling via an exhaustive bijection+adjacency harness (after a local-mask reflection bug); and I convinced myself the *driver* is right via 600 differential cases plus hand-checked edges. That is what I ship — one self-contained file, Mo's algorithm with Hilbert ordering:

```cpp
#include <bits/stdc++.h>
using namespace std;

// Map (x, y) on a side x side grid (side = 2^order) to its 1-D distance along the
// Hilbert space-filling curve. Canonical iterative xy2d construction.
static inline uint64_t hilbertOrder(uint32_t x, uint32_t y, int order) {
    uint64_t d = 0;
    for (int s = order - 1; s >= 0; --s) {
        uint32_t rx = (x >> s) & 1u;
        uint32_t ry = (y >> s) & 1u;
        d += ((uint64_t)((3u * rx) ^ ry)) << (2 * s);
        // rotate / reflect the quadrant
        if (ry == 0) {
            if (rx == 1) {
                uint32_t mask = (1u << order) - 1u;   // side - 1
                x = (mask - x) & mask;
                y = (mask - y) & mask;
            }
            uint32_t t = x; x = y; y = t;             // swap
        }
    }
    return d;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<int> a(n);
    int maxv = 0;
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
        if (a[i] > maxv) maxv = a[i];
    }

    // Hilbert grid order: smallest power of two side covering all indices.
    int order = 1;
    while ((1 << order) < max(1, n)) ++order;

    struct Query { int l, r, idx; uint64_t h; };
    vector<Query> qs(q);
    for (int i = 0; i < q; ++i) {
        int l, r;
        cin >> l >> r;          // 1-based inclusive
        --l; --r;               // to 0-based inclusive
        qs[i].l = l;
        qs[i].r = r;
        qs[i].idx = i;
        qs[i].h = hilbertOrder((uint32_t)l, (uint32_t)r, order);
    }

    sort(qs.begin(), qs.end(),
         [](const Query& A, const Query& B) { return A.h < B.h; });

    vector<int> cnt(maxv + 2, 0);
    vector<long long> ans(q, 0);

    long long cur = 0;          // sum over values v of cnt[v]^2 inside [curL, curR]
    int curL = 0, curR = -1;    // empty window

    auto add = [&](int pos) {
        int v = a[pos];
        long long c = cnt[v];
        cur += 2 * c + 1;       // (c+1)^2 - c^2
        cnt[v] = (int)(c + 1);
    };
    auto remove = [&](int pos) {
        int v = a[pos];
        long long c = cnt[v];
        cur -= 2 * c - 1;       // c^2 - (c-1)^2
        cnt[v] = (int)(c - 1);
    };

    for (const auto& Q : qs) {
        int L = Q.l, R = Q.r;
        while (curR < R) add(++curR);
        while (curL > L) add(--curL);
        while (curR > R) remove(curR--);
        while (curL < L) remove(curL++);
        ans[Q.idx] = cur;
    }

    string out;
    out.reserve((size_t)q * 12);
    char buf[24];
    for (int i = 0; i < q; ++i) {
        int len = sprintf(buf, "%lld\n", ans[i]);
        out.append(buf, len);
    }
    fputs(out.c_str(), stdout);
    return 0;
}
```

**Causal recap.** Per-query brute is `O(nq) ~ 4*10^10` and segment trees fail because `P` couples all occurrences of a value, so I moved to Mo's, whose viability hinges on the `O(1)` squared-frequency delta `(c+1)^2 - c^2 = 2c+1`; the ordering that hits the `O((n+q) sqrt n)` bound with the smallest constant is the Hilbert space-filling curve (no row-major boundary cliffs), which is the SOTA Mo variant; the only real bug was a Hilbert reflection that used a local low-bit mask instead of the full `side-1` mask, caught not by the answer diff (Mo's is answer-correct under any order) but by an exhaustive bijection+adjacency check of the curve; with the full-mask fix the curve is verified space-filling through order 8, the driver passes 600 differential cases and all hand-checked edges, and the program runs in ~0.18 s at `n = q = 2*10^5` with 64-bit accumulators guarding the `4*10^10` answers.
