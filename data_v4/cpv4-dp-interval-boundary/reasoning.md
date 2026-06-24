**Reading the problem and pinning the contract.** I have `n` slabs in a row with widths `w[0..n-1]`,
and a robot fuses the whole row into one piece by a sequence of welds. A weld joins two *currently
adjacent* pieces and costs the combined width of those two pieces — i.e. the sum of the original
widths of every slab inside the merged span. I want the minimum total cost over all valid weld
orders. Input is `n` then the `n` widths; I print one integer. Before any algorithm I fix the scale,
because it dictates the data types and the asymptotics. `n <= 400` and `1 <= w[i] <= 10^6`. The total
of all widths is at most `400 * 10^6 = 4*10^8`. The final weld over the whole row already costs that
much, and earlier welds re-pay overlapping sub-widths, so the grand total can be on the order of
`(n-1)` times the total width, i.e. roughly `400 * 4*10^8 = 1.6*10^11`. That is far past the 32-bit
ceiling of `~2.1*10^9`, so every accumulator and the entire DP table must be 64-bit `long long`. An
`int` here is a silent wrong-answer on the large tests. That is decision one, non-negotiable.

`n <= 400` is also a deliberate signal about the intended complexity: `400^3 = 6.4*10^7`, which is
comfortable in 2 seconds, but `400^4` is not. So I am being pointed at an `O(n^3)` interval DP, not
anything heavier.

**Laying out the candidate approaches.** Two routes, and I want the one I can *prove*.

- *Greedy by smallest adjacent pair.* The cost structure smells like Huffman coding: there, you
  always merge the two smallest items, and it is optimal. The instinct here is to always weld the
  cheapest currently-adjacent pair first. It is short and `O(n log n)`-ish. But Huffman gets to
  combine *any* two items; here I may only fuse *neighbours*. That adjacency restriction is exactly
  the kind of thing that breaks a global greedy, so I will not trust it without trying to break it.
- *Interval DP.* Let `dp[i][j]` be the minimum heat to fuse the closed range of slabs `[i, j]` into a
  single piece. The structural fact I derived in the background is the lever: however I fuse `[i, j]`,
  the *last* weld in that range joins two adjacent sub-pieces that together span all of `[i, j]`, so
  it pays `sum(w[i..j])` no matter how the two halves were built, and the two halves are themselves
  fully-fused sub-ranges `[i, k]` and `[k+1, j]`. That gives a recurrence. `O(n^3)`. The only risk
  is the *boundaries*: which slabs `[i, j]` covers, where `k` may sit, and which prefix index gives
  the width.

**Stress-testing greedy before committing.** "Smallest-first feels Huffman-optimal" is precisely the
kind of belief that ships wrong code, so let me actually attack it. Take `w = [1, 100, 1, 100]`,
indices `0..3`. Smallest-first looks for the cheapest *adjacent* pair. The adjacent pairs are
`(1,100)=101`, `(100,1)=101`, `(1,100)=101` — all equal, so this instance does not separate the
orders; let me pick a sharper one. Take `w = [3, 1, 1, 3]`. Adjacent sums: `(3,1)=4`, `(1,1)=2`,
`(1,3)=4`. Greedy welds the middle `(1,1)` first, cost `2`, leaving pieces `[3, 2, 3]` (the middle is
now a width-2 piece). Now adjacent sums are `(3,2)=5` and `(2,3)=5`; weld either, say left, cost `5`,
leaving `[5, 3]`; final weld cost `8`. Greedy total `2 + 5 + 8 = 15`.

Is `15` optimal? Let me hunt for a cheaper order. Weld the left pair `(3,1)` first: cost `4`, pieces
`[4, 1, 3]`. Then weld `(1,3)`: cost `4`, pieces `[4, 4]`. Final weld: cost `8`. Total `4 + 4 + 8 =
16` — worse. Try welding `(1,1)` then the *right* `(2,3)`: same as the left-symmetric `15`. Hmm, on
this instance greedy's `15` does seem best, so this is not a counterexample. Let me push harder with
asymmetry: `w = [1, 9, 1, 9, 1]`. I will not enumerate by hand under time pressure; instead I will
trust the brute force I am about to write to settle whether greedy can ever lose. The *reason* I
distrust greedy is concrete: by fusing two cheap neighbours, greedy may bury a slab in the middle of
a piece that then gets re-paid in every later weld, whereas a different order could keep an expensive
boundary slab on the outside and weld it last. That is a global trade-off greedy cannot see. So I am
going to *build* the interval DP and verify it against an independent brute force, and only ship the
DP. (When I run the comparison later, greedy is irrelevant — I am comparing the DP against an
exhaustive order-enumeration, which is the real ground truth.)

**Deriving the DP and checking the recurrence on paper.** I want `dp[i][j]` = minimum heat to fuse
the closed range `[i, j]`. Base case: a single slab needs no welds, so `dp[i][i] = 0`. For `j > i`,
the last weld splits the range into `[i, k]` and `[k+1, j]` for some boundary, pays the full range
width, and the two halves are solved recursively:

```
dp[i][j] = min over k in [i, j-1] of ( dp[i][k] + dp[k+1][j] ) + W(i, j)
```

where `W(i, j) = w[i] + w[i+1] + ... + w[j]` is the **inclusive** sum over the closed range. The
split index `k` is the index of the *last slab of the left half*, so the left half is `[i, k]` and
the right half is `[k+1, j]`. For both halves to be non-empty I need `k >= i` (left non-empty) and
`k+1 <= j`, i.e. `k <= j-1` (right non-empty). So `k` ranges over `i .. j-1` inclusive. Getting this
range wrong by one — allowing `k = j` — would index `dp[j+1][j]`, an empty/garbage range; allowing
`k = i-1` would make the left half empty. The half-open feeling of "split point" versus the inclusive
closed range is exactly where I expect to slip.

I will store widths via a prefix-sum array. I will use the **half-open** convention `prefix[t] =
w[0] + ... + w[t-1]`, so `prefix[0] = 0` and `prefix[n] = total`. Then the sum over the *closed*
range `[i, j]` is `prefix[j+1] - prefix[i]` — note the `j+1`, because `prefix` is half-open but my
range is inclusive. This mismatch between a half-open prefix array and an inclusive range index is a
classic off-by-one trap, and I am flagging it now so I check it under a trace.

Let me confirm the recurrence by hand on the sample `w = [3, 1, 4, 1]`, claimed answer `18`. Widths:
`W(0,3) = 9`. The cheapest order I can find by hand: weld `(1,4)` (indices 1,2) first, cost `5`,
pieces `[3, 5, 1]`; weld `(5,1)` (the new middle and index 3), cost `6`, pieces `[3, 6]`; final weld
cost `9`. Total `5 + 6 + 9 = 20`. Try another: weld `(3,1)` first, cost `4`, pieces `[4, 4, 1]`; weld
`(4,1)` right, cost `5`, pieces `[4, 5]`; final cost `9`; total `4 + 5 + 9 = 18`. There it is, `18`.
And via the recurrence, the split that produces it is `k = 0` at the top: `dp[0][3] = dp[0][0] +
dp[1][3] + W(0,3) = 0 + dp[1][3] + 9`, and `dp[1][3] = dp[1][1] + dp[2][3] + W(1,3) = 0 + (w2+w3) +
(1+4+1) = 5 + 6 = 11`? Let me redo: `dp[2][3] = w[2] + w[3] = 4 + 1 = 5`, `W(1,3) = 1+4+1 = 6`, so
`dp[1][3] = 0 + 5 + 6 = 11`. Wait that does not match my hand order; let me try `dp[1][3]` splitting
at `k=2`: `dp[1][2] + dp[3][3] + W(1,3) = (1+4) + 0 + 6 = 5 + 6 = 11`. Both splits give `11`. Then
`dp[0][3]` with `k=0` is `0 + 11 + 9 = 20`. With `k=2`: `dp[0][2] + dp[3][3] + 9`; `dp[0][2] =
min(dp[0][0]+dp[1][2], dp[0][1]+dp[2][2]) + W(0,2) = min(0+5, (3+1)+0) + 8 = min(5,4) + 8 = 4 + 8 =
12`, so `dp[0][3] = 12 + 0 + 9 = 21`. With `k=1`: `dp[0][1] + dp[2][3] + 9 = 4 + 5 + 9 = 18`. So the
minimum over `k` is `18` at `k=1` — matching the hand order `(3,1)` then `(4,1)` then all. The
recurrence is right, and the off-by-one-prone pieces are `k in [i, j-1]` and `W = prefix[j+1] -
prefix[i]`.

**First implementation — and immediately a trace, because clean math transcribes dirty.** I write
the prefix-sum array and the interval DP. My first cut of the width term, transcribing "sum over
`[i, j]`", reaches for the prefix indices and I write the *natural-looking* `prefix[j] - prefix[i]`,
because my mental image is "from `i` up to `j`". The loop body:

```
for (int len = 2; len <= n; len++)
  for (int i = 0; i + len - 1 < n; i++) {
    int j = i + len - 1;
    long long best = INF;
    for (int k = i; k < j; k++)
      best = min(best, dp[i][k] + dp[k+1][j]);
    dp[i][j] = best + (prefix[j] - prefix[i]);   // <-- suspicious
  }
```

I trace the smallest input that could expose the width term: `w = [2, 3]`, where the answer is
obviously `2 + 3 = 5` (one weld joining the two slabs, cost = their combined width). `prefix = [0, 2,
5]`. `len = 2`, `i = 0`, `j = 1`. Inner loop `k = 0` only: `best = dp[0][0] + dp[1][1] = 0`. Then
`dp[0][1] = 0 + (prefix[1] - prefix[0]) = 0 + (2 - 0) = 2`. Final answer `dp[0][1] = 2`.

**Diagnosing the first bug (off-by-one in the width index).** The code returns `2`, but the only
weld possible costs `2 + 3 = 5`. The defect is exact: `prefix[j] - prefix[i]` with `j = 1` is
`prefix[1] - prefix[0] = w[0] = 2` — it sums the *half-open* range `[i, j) = [0, 1)`, which contains
only slab `0` and **drops slab `j` itself**. My range is the *closed* `[i, j]`, which must include
slab `j`, so the width is `prefix[j+1] - prefix[i]`, not `prefix[j] - prefix[i]`. This is precisely
the half-open-prefix-versus-inclusive-index mismatch I flagged during derivation: I mixed a half-open
prefix array (`prefix[t]` excludes `w[t]`) with an inclusive endpoint `j`. The fix is one character of
index: `prefix[j + 1] - prefix[i]`. Re-trace `[2, 3]`: `dp[0][1] = 0 + (prefix[2] - prefix[0]) = 0 +
(5 - 0) = 5`. Correct. The bug was real, the trace caught it on the minimal case, and the cause is
the documented inclusive/exclusive boundary. This is exactly the pitfall this problem is built
around.

**A second trace, because off-by-ones travel in packs — the split bound.** With the width fixed I am
still nervous about the inner loop bound, since `k` indexes the same kind of boundary. Suppose, in a
moment of "let me make sure I do not miss a split", I had written `for (k = i; k <= j; k++)` instead
of `k < j`. I trace `w = [2, 3]` again with that variant. `len = 2`, `i = 0`, `j = 1`. Now `k` runs
`0, 1`. At `k = 0`: `dp[0][0] + dp[1][1] = 0`, fine. At `k = 1`: I read `dp[i][k] + dp[k+1][j] =
dp[0][1] + dp[2][1]`. But `dp[0][1]` is the very cell I am *computing right now* (still `INF` /
unset), and `dp[2][1]` is an inverted range `[2, 1]` that was never filled — both are garbage. So
`k = j` is illegal: it corresponds to an *empty right half* `[j+1, j]`, which is not a real split. The
legal splits keep both halves non-empty, which forces `k <= j-1`, i.e. `k < j`. The correct bound is
`k < j`, and my original code already had it; the second trace confirms that the alternative I was
tempted by would have read self-referential and out-of-range cells. So the inner bound stays `k < j`.

I run the same paranoia on the *outer* indexing. The interval `[i, j]` with `j = i + len - 1` must
satisfy `j <= n-1`, i.e. `i + len - 1 < n`, which is exactly my outer loop condition `i + len - 1 <
n`. If I had written `i + len <= n` I would also be fine (`i + len - 1 < n` <=> `i + len <= n`), but
if I had written `i + len - 1 <= n` I would let `j = n`, an out-of-range slab. I keep `i + len - 1 <
n`. And `len` runs `2 .. n` because length-1 intervals are the base case (cost 0) already in the
table, and the full row is `len = n`. If I had started `len` at `1` it would be harmless (the body
would recompute `dp[i][i]` but with an empty `k`-loop leaving `best = INF`, which would then corrupt
`dp[i][i]` to `INF`!). So starting at `len = 2` is not cosmetic — starting at `len = 1` would set
every singleton to `INF + width` and poison everything. I keep `len` from `2`.

**Re-verifying the fixed DP against the recurrence and the sample.** With `prefix[j+1] - prefix[i]`
and `k < j` and `len` from `2`, I recompute the sample `[3, 1, 4, 1]` symbolically: I derived
`dp[0][3] = 18` above via the recurrence, and the code now implements that recurrence faithfully, so
it returns `18`. The minimal cases `[2, 3] -> 5` and `[2] -> 0`, `[] -> 0` also follow. I am
confident the transcription matches the math.

**Edge cases, deliberately, because this is where interval DP dies.**
- `n = 0`: I short-circuit with `if (n <= 1) print 0`. The empty row needs no welds. Correct, and it
  also dodges building a `0 x 0` table.
- `n = 1`: same short-circuit prints `0`. A lone slab needs no welds. Correct. (Without the guard,
  the `len` loop from `2` would never run and `dp[0][0] = 0` would print correctly anyway, but the
  guard makes the intent explicit and avoids a zero-size vector edge.)
- `n = 2`: `dp[0][1] = w[0] + w[1]`, the single forced weld. Correct.
- All-equal widths, e.g. `[5, 5, 5]`: any order gives `(5+5) + (10+5) = 10 + 15 = 25` or `(5+5) +
  (5+10) = 25` — symmetric, `dp` returns `25`. Sanity holds.
- Overflow: the DP table and accumulators are `long long`. The largest `dp` value is `~1.6*10^11`,
  and `INF = LLONG_MAX/4 ~ 2.3*10^18`; `dp[i][k] + dp[k+1][j]` with real (non-INF) values is at most
  `~3.2*10^11`, never near INF, and I only ever *read* INF inside a `min`, never add a width to a
  cell that stayed INF (every `dp[i][j]` for `i < j` gets a real value because the `k`-loop always
  has at least one iteration when `j > i`). So no overflow and no INF leakage. Safe.
- Memory: the table is `n*n` `long long` = `400*400*8 = 1.28*10^6` bytes ~ 1.3 MB, well under 256 MB.
- Time: `O(n^3) = 6.4*10^7` simple operations, ~0.01 s in practice. Comfortable under 2 s.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace so parsing is
  format-agnostic.

**Final solution.** I convinced myself the *idea* is right by deriving the recurrence from the
"last weld pays the whole range width" structure and checking it on the sample, and I convinced
myself the *code* is right by tracing the minimal case to a precise off-by-one cause
(`prefix[j]` dropping slab `j`), fixing it to `prefix[j+1]`, and a second trace ruling the split
bound `k < j` correct (its `k = j` alternative reads self-referential and inverted cells). That is
what I ship — one self-contained `O(n^3)` interval DP:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    if (n <= 1) { cout << 0 << "\n"; return 0; }

    // prefix[i] = w[0] + ... + w[i-1], a half-open prefix sum over [0, i).
    vector<long long> prefix(n + 1, 0);
    for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + w[i];

    // dp[i][j] = minimum total merge cost to combine the slabs whose indices
    // lie in the CLOSED interval [i, j] into one slab. The cost of the final
    // merge that unites the two children [i,k] and [k+1,j] is the combined
    // width prefix[j+1]-prefix[i] (sum of all w over the closed range).
    const long long INF = LLONG_MAX / 4;
    vector<vector<long long>> dp(n, vector<long long>(n, 0));
    // length-1 intervals already cost 0 (initialized above).

    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;          // closed interval [i, j]
            long long best = INF;
            // split between k and k+1, so k ranges over i .. j-1 (inclusive).
            for (int k = i; k < j; k++) {
                long long cur = dp[i][k] + dp[k + 1][j];
                if (cur < best) best = cur;
            }
            // width of the closed range [i, j] is prefix[j+1] - prefix[i].
            dp[i][j] = best + (prefix[j + 1] - prefix[i]);
        }
    }

    cout << dp[0][n - 1] << "\n";
    return 0;
}
```

**Causal recap.** The cost structure tempted me toward a Huffman-style smallest-first greedy, but the
*adjacency* restriction (only neighbours may fuse) is a global trade-off greedy cannot see, so I
moved to an interval DP whose recurrence falls out of the fact that the last weld over a closed range
always pays that range's total width. The two failure modes were both off-by-ones at the range
boundary: writing the width as the half-open `prefix[j] - prefix[i]` silently drops slab `j` (a trace
of `[2, 3]` returning `2` instead of `5` pinned it, fixed to `prefix[j+1] - prefix[i]`), and the
split index must keep both halves non-empty, which a second trace showed forces `k < j` rather than
`k <= j` (the latter reads the cell being computed and an inverted range). With the width index, the
split bound, the outer `j <= n-1` guard, and `len` starting at `2` all aligned to the closed-interval
convention, and with `long long` throughout for the `~1.6*10^11`-magnitude answer, the `O(n^3)` DP is
correct and well within the limits.
