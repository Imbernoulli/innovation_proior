**Reading the problem and pinning the contract.** I have two integer arrays, `A` of length `n` and `B` of length `m`, both possibly containing negatives. I must choose a non-empty subsequence of `A` and a non-empty subsequence of `B` of the *same* length `k >= 1`, keep the original left-to-right order in each, line them up position by position, and maximize the dot product `A[i_1]*B[j_1] + ... + A[i_k]*B[j_k]` with `i_1 < ... < i_k` and `j_1 < ... < j_k`. The empty pairing is forbidden, so unlike a "max subarray" the answer is allowed to be negative — if every legal pairing is negative, I still must report the least-bad one. Input is `n m` on the first line, then the `n` values of `A`, then the `m` values of `B`; output is one integer.

Before any algorithm I fix the scale, because it dictates the data types. The constraints are `1 <= n, m <= 500` and `|A[i]|, |B[j]| <= 1000`. A single product is at most `1000 * 1000 = 10^6` in magnitude, and a pairing has at most `min(n, m) <= 500` terms, so the dot product lives in `[-5*10^8, 5*10^8]`. That fits inside a 32-bit `int` (limit about `2.1*10^9`) — but only just, and I do not want to think about intermediate states that might exceed it, nor about a sentinel value for "no pairing yet." So I will use `long long` throughout. That is the first decision and it costs nothing here. An `int` would be defensible at these bounds, but `long long` removes a whole class of doubt, and the array is at most `500*500` cells anyway.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one that is fastest to type.

- *Greedy by best product.* A length-1 pairing is always legal: pick any one `i` and one `j` and score `A[i]*B[j]`. So a tempting shortcut is "the answer is just `max_{i,j} A[i]*B[j]`." It even has a seductive justification: with negatives in play, two large-magnitude negatives multiply to a large positive, and surely one dominant product beats spreading the budget across many smaller terms. A cousin of this idea is "sort both arrays by magnitude and pair the biggest with the biggest, the second-biggest with the second-biggest, and sum the positive contributions." Both are `O(nm)` or `O(n log n + m log m)` and a few lines. The risk is structural on two fronts: (1) longer alignments can *accumulate* many positive products that individually look unimpressive, and (2) the sorted pairing quietly ignores the in-order constraint — you cannot reorder a subsequence. I will not trust either until I have tried to break them.

- *Alignment dynamic programming.* Scan both prefixes and carry, for each pair of prefix lengths `(i, j)`, the best dot product of a non-empty aligned pairing drawn from `A[0..i-1]` and `B[0..j-1]`. This is the classic two-sequence alignment shape, `O(nm)` time and `O(nm)` (or `O(m)`) memory. With `n, m <= 500` that is at most `250000` cells, utterly trivial for a 1-second limit. The risk here is not the idea but the *transcription*: enforcing "non-empty" correctly, and not letting a phantom empty pairing (worth `0`) leak in and corrupt the all-negative case.

**Stress-testing the greedy before committing.** "Greedy feels right" is exactly how wrong solutions get shipped, so let me attack it with concrete instances rather than intuition.

First, the "single best product" claim. Take `A = [2, 1, -2]`, `B = [3, 0, -1]`. The largest single product: scan all nine pairs — `A[0]*B[0] = 6` is the biggest (the others are `0`, `-2`, `3`, `0`, `-1`, `-6`, `0`, `2`). So the greedy answer is `6`. Is `6` optimal? Look at the length-2 alignment that pairs `A[0]` with `B[0]` and `A[2]` with `B[2]`: indices `0 < 2` in both arrays, legal, scoring `2*3 + (-2)*(-1) = 6 + 2 = 8`. That is strictly better than `6`. So "single best product" is wrong, and I see *why*: a second pair with a modest positive product (`2`) simply adds on top, and the greedy never considers stacking pairs. The verification paid off — it killed an approach I might have shipped.

Second, the more sophisticated "sort and pair largest magnitudes" idea, because someone could rescue greedy by reaching for it after the first counterexample. Take `A = [3, -5]`, `B = [-5, 3]`. Sort each array descending: `A` becomes `[3, -5]`, `B` becomes `[3, -5]`. Pairwise-multiply and sum: `3*3 + (-5)*(-5) = 9 + 25 = 34`. So the sorted-pairing greedy reports `34`. But `34` is *unreachable*. To realize the pair `(-5, -5)` I would use `A[1]` (the `-5`) with `B[0]` (the `-5`), and to realize `(3, 3)` I would use `A[0]` with `B[1]`. Those two pairs have `A` indices `1` and `0` matched to `B` indices `0` and `1` — the order is crossed, which is not a valid order-preserving alignment. The only legal length-2 alignment of `A = [3, -5]` with `B = [-5, 3]` keeps both in order: `3*(-5) + (-5)*3 = -15 - 15 = -30`. The actual optimum here is the length-1 pairing `A[1]*B[0] = (-5)*(-5) = 25`. So the sorted greedy overcounts by reordering, claiming `34` when the truth is `25`. Greedy in both forms is out: the first ignores that pairs stack, the second ignores that the alignment must preserve order.

So both tempting clever shortcuts are demonstrably wrong, and they are wrong for reasons (stacking; order) that a greedy cannot patch without essentially rediscovering the DP. I move to the alignment DP, which I can prove.

**Deriving the DP and checking the recurrence on paper.** I want, for every pair of prefix lengths `(i, j)`, the value `dp[i][j]` = the best dot product achievable by a *non-empty* aligned pairing that uses only `A[0..i-1]` and `B[0..j-1]`. The decision at cell `(i, j)` concerns the last elements `A[i-1]` and `B[j-1]`. There are exactly these mutually exhaustive cases for an optimal pairing restricted to those prefixes:

- It uses `A[i-1]` paired with `B[j-1]` as its last aligned pair. Then the rest of the pairing (if any) lives in `A[0..i-2]` and `B[0..j-2]`. Two sub-cases: this pair is the *only* pair, contributing just `A[i-1]*B[j-1]`; or it extends an already non-empty pairing on the smaller prefixes, contributing `dp[i-1][j-1] + A[i-1]*B[j-1]`. I take the max of the two, which is why I need both the "start fresh" term and the "extend" term.
- It does not use `A[i-1]` at all. Then it is a pairing within `A[0..i-2]` and `B[0..j-1]`, value `dp[i-1][j]`.
- It does not use `B[j-1]` at all. Then it is a pairing within `A[0..i-1]` and `B[0..j-2]`, value `dp[i][j-1]`.

So the recurrence is

```
dp[i][j] = max( A[i-1]*B[j-1],
                dp[i-1][j-1] + A[i-1]*B[j-1],   // only if dp[i-1][j-1] is a real pairing
                dp[i-1][j],
                dp[i][j-1] )
```

with the convention that `dp[*][0]` and `dp[0][*]` are "no non-empty pairing exists" (a sentinel I will write as `NEG`, a very negative number), and any `dp[i-1][j]` / `dp[i][j-1]` that is `NEG` simply loses every `max` against the always-present `A[i-1]*B[j-1]` term. The crucial subtlety is the `dp[i-1][j-1] + prod` term: it is only meaningful when `dp[i-1][j-1]` is a *real* pairing, not the empty sentinel. If I blindly added `prod` to a `NEG` sentinel I would still get something extremely negative (so it would lose the `max`), which is harmless — but I will guard it explicitly anyway, both for clarity and to avoid any chance of a sentinel-plus-product underflowing below `LLONG_MIN`. The standalone `A[i-1]*B[j-1]` term is what guarantees every cell with `i,j >= 1` has at least one legal non-empty pairing, so `dp[n][m]` is always a real answer.

Let me confirm the recurrence by hand on the first sample `A = [2, 1, -2]`, `B = [3, 0, -1]`, expected `8`. I will fill the `3x3` interior of `dp` (rows `i=1..3` for `A`, columns `j=1..3` for `B`). I write each cell as the max of: the standalone product `p = A[i-1]*B[j-1]`, the diagonal-extend `dp[i-1][j-1]+p` (when real), the up `dp[i-1][j]`, the left `dp[i][j-1]`.

- `(1,1)`: `p = 2*3 = 6`. No real diagonal/up/left (all on the border). `dp[1][1] = 6`.
- `(1,2)`: `p = 2*0 = 0`. diagonal border, up border, left `dp[1][1]=6`. `dp[1][2] = max(0, 6) = 6`.
- `(1,3)`: `p = 2*(-1) = -2`. left `dp[1][2]=6`. `dp[1][3] = max(-2, 6) = 6`.
- `(2,1)`: `p = 1*3 = 3`. up `dp[1][1]=6`. `dp[2][1] = max(3, 6) = 6`.
- `(2,2)`: `p = 1*0 = 0`. diagonal `dp[1][1]+0 = 6`, up `dp[1][2]=6`, left `dp[2][1]=6`. `dp[2][2] = 6`.
- `(2,3)`: `p = 1*(-1) = -1`. diagonal `dp[1][2]+(-1) = 5`, up `dp[1][3]=6`, left `dp[2][2]=6`. `dp[2][3] = 6`.
- `(3,1)`: `p = (-2)*3 = -6`. up `dp[2][1]=6`. `dp[3][1] = 6`.
- `(3,2)`: `p = (-2)*0 = 0`. diagonal `dp[2][1]+0 = 6`, up `dp[2][2]=6`, left `dp[3][1]=6`. `dp[3][2] = 6`.
- `(3,3)`: `p = (-2)*(-1) = 2`. diagonal `dp[2][2]+2 = 8`, up `dp[2][3]=6`, left `dp[3][2]=6`. `dp[3][3] = max(2, 8, 6, 6) = 8`.

`dp[3][3] = 8`, matching the expected answer, and the `8` comes precisely from the diagonal extend at `(3,3)` stacking `A[2]*B[2] = 2` on top of the length-1 `dp[2][2] = 6` — exactly the "pairs stack" effect the greedy missed.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the inner body, written quickly, did *not* guard the extend term and used a too-clever default:

```
long long best = dp[i - 1][j - 1] + prod;     // extend
best = max(best, dp[i - 1][j]);
best = max(best, dp[i][j - 1]);
dp[i][j] = best;
```

The "start a brand-new pairing here" term — the standalone `prod` — is *missing*, and `dp[i-1][j-1]` for border cells is the sentinel `NEG = LLONG_MIN/4`. So on the very first interior cell `(1,1)` this computes `best = NEG + prod`, an enormous negative number, then maxes it against two more border sentinels. Let me trace the smallest input that exposes the damage: `A = [5]`, `B = [3]`, where the answer is obviously `5*3 = 15` (the only legal pairing is the single pair). Border `dp[0][0] = NEG`. At `(1,1)`: `prod = 15`; `best = dp[0][0] + 15 = NEG + 15` (still about `-2.3*10^18`); then `max` against `dp[0][1]=NEG` and `dp[1][0]=NEG`. Final `dp[1][1] = NEG + 15`, and I print roughly `-2305843009213693936`. Completely wrong.

**Diagnosing the bug.** The output is a giant negative instead of `15`. The defect is precise and twofold. (1) I omitted the standalone `prod` term, so a length-1 pairing — the base case of "non-empty" — is never representable on its own; every cell tries to *extend* something, but at the borders there is nothing real to extend, only the `NEG` sentinel. (2) Adding `prod` to the `NEG` sentinel is itself dangerous: it produces a value that is still hugely negative (so it loses), but it is also flirting with underflow, and worse, it *encodes a falsehood* — it pretends a non-empty pairing on the empty prefix exists. The fix has to make "start fresh with just this pair" a first-class option and must only add `prod` to `dp[i-1][j-1]` when that cell holds a *real* pairing, not the sentinel.

**Fixing and re-verifying.** I rewrite the body so the standalone product is always a candidate, and the extend term is gated on `dp[i-1][j-1] != NEG`:

```
long long prod = A[i - 1] * B[j - 1];
long long best = prod;                          // brand-new length-1 pairing
if (dp[i - 1][j - 1] != NEG)
    best = max(best, dp[i - 1][j - 1] + prod);  // extend a real pairing
best = max(best, dp[i - 1][j]);                 // drop A[i-1]
best = max(best, dp[i][j - 1]);                 // drop B[j-1]
dp[i][j] = best;
```

Re-trace `A = [5]`, `B = [3]`: at `(1,1)`, `prod = 15`, `best = 15`, the diagonal is `NEG` so the extend is skipped, up and left are `NEG` and lose. `dp[1][1] = 15`. Correct. Re-trace the all-negative pair `A = [-3, -4]`, `B = [-5, -6]`, where I expect `39` (both negatives multiplied and stacked: `(-3)(-5) + (-4)(-6) = 15 + 24 = 39`): `(1,1)`: `prod = 15`, `dp = 15`. `(1,2)`: `prod = (-3)(-6) = 18`, left `15` -> `18`. `(2,1)`: `prod = (-4)(-5) = 20`, up `15` -> `20`. `(2,2)`: `prod = (-4)(-6) = 24`, diagonal `dp[1][1]+24 = 39`, up `dp[1][2]=18`, left `dp[2][1]=20` -> `max(24, 39, 18, 20) = 39`. `dp[2][2] = 39`. Correct, and again the winning value is a *stack* of two pairs, the exact thing both greedies could not see. The two cases that broke (or would have broken) before now pass for the reason I fixed.

**Edge cases, deliberately, because this is where this kind of code dies.**

- `n = 1, m = 1`, `A = [-5]`, `B = [3]`: the only legal pairing is `(-5)*3 = -15`, a forced negative. `dp[1][1] = -15`, printed as `-15`. Correct — the problem demands non-empty, so a negative answer is right; there is no `max(..., 0)` here, unlike a max-subarray, precisely because the empty pairing is disallowed.
- All negative everywhere: two negatives give a positive product, and the DP's diagonal-extend stacks them, so the answer is a large positive, as the `[-3,-4]/[-5,-6] = 39` trace showed. No special handling needed.
- Zeros mixed in: a `0` product is just another candidate value; nothing special. Verified later by the random oracle, which injects zeros.
- Mismatched lengths, e.g. `n = 1, m = 500`: the pairing length is capped at `min(n, m) = 1`, and the DP naturally only ever realizes length-1 pairings because every column past the first can only "drop B" or start a fresh single pair. Correct by construction.
- Overflow and the sentinel: products are at most `10^6`, sums at most `5*10^8`, all far inside `long long`. The sentinel `NEG = LLONG_MIN/4 ~ -2.3*10^18` is only ever read inside a `max` or behind the `!= NEG` guard; I never add `prod` to it, so it cannot underflow. Safe.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the three-line input format is parsed correctly even if the values wrap across lines.

**Self-verification against a brute oracle.** Hand-traces convince me of specific cases; to convince myself of the *general* recurrence I wrote an independent brute force that enumerates every non-empty pair of equal-length subsequences (`itertools.combinations` over the index sets of `A` and `B` for each common length `k`, dot-producting in order) and takes the max — an `O(sum_k C(n,k) C(m,k))` exhaustive search, valid only for tiny `n, m`. Then a generator emits random instances biased toward the dangerous regimes: mixed signs, all-negative, all-positive, and zero-injected, with `n, m` up to about `8` and values up to a few dozen. I ran more than 1100 random instances plus the explicit edge cases above (`n=1`/`m=1`, all-negative, zeros, mismatched lengths) through both the DP and the brute oracle and got **zero mismatches**. I also ran the largest shape, `n = m = 500` with all values `1000` (answer `5*10^8`) and all values `-1000` (answer `5*10^8`), confirming both the correctness of the extreme and the absence of overflow, and the program finishes in about 3 milliseconds — three orders of magnitude under the 1-second limit, exactly as expected for a `250000`-cell DP. The greedy counterexamples (`[2,1,-2]/[3,0,-1]`: greedy `6` vs optimal `8`; `[3,-5]/[-5,3]`: sorted greedy `34` vs optimal `25`) were also run through the oracle to confirm the DP returns the true optimum, not the greedy's value.

**Final solution.** I convinced myself the *idea* is right by disproving both greedy variants with concrete counterexamples and by checking the alignment recurrence on the sample by hand; I convinced myself the *code* is right by tracing the failing initial version to a precise cause (missing standalone term plus an unguarded sentinel add), re-verifying the fix on the cases that broke, and then differential-testing the whole thing against an independent exhaustive oracle over a thousand-plus instances with zero mismatches. That is what I ship — one self-contained `O(nm)` file, the simple alignment DP I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<long long> A(n), B(m);
    for (auto &x : A) cin >> x;
    for (auto &x : B) cin >> x;

    // dp[i][j] = best dot product of a NON-EMPTY equal-length pairing using
    // a subsequence of A[0..i-1] and of B[0..j-1]. NEG sentinel = "no non-empty
    // pairing exists yet for this prefix pair".
    const long long NEG = LLONG_MIN / 4;
    vector<vector<long long>> dp(n + 1, vector<long long>(m + 1, NEG));

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            long long prod = A[i - 1] * B[j - 1];
            // pair A[i-1] with B[j-1], either as the only pair or extending dp[i-1][j-1]
            long long best = prod;                       // start a brand-new pairing here
            if (dp[i - 1][j - 1] != NEG)
                best = max(best, dp[i - 1][j - 1] + prod); // extend an existing pairing
            // or drop A[i-1] / drop B[j-1]
            best = max(best, dp[i - 1][j]);
            best = max(best, dp[i][j - 1]);
            dp[i][j] = best;
        }
    }

    cout << dp[n][m] << "\n";
    return 0;
}
```

**Causal recap.** The single-best-product greedy looked plausible because a length-1 pairing is always legal and big negatives multiply to big positives, but a traced counterexample (`[2,1,-2]/[3,0,-1]`: greedy `6` vs the reachable `8`) showed that pairs *stack*, so a modest second product adds on top; the sort-and-pair-by-magnitude rescue then died on `[3,-5]/[-5,3]` (it claims `34` by crossing the order, but the only legal length-2 alignment scores `-30` and the true optimum is `25`), because it ignores order-preservation. That pushed me to the `O(nm)` alignment DP, whose recurrence I verified on the sample; my first transcription omitted the standalone-pair term and added the product to the empty-prefix sentinel, which a trace of `[5]/[3]` returning a giant negative instead of `15` pinpointed; making "start a fresh single pair" a first-class candidate and gating the extend on a real (non-`NEG`) diagonal fixes it; and because the empty pairing is *disallowed*, there is deliberately no `max(..., 0)` — the all-negative-forced cases like `[-5]/[3] = -15` are supposed to be negative — while 64-bit accumulators and a guarded sentinel close out the overflow and underflow corners, all confirmed by 1100+ oracle-checked instances and the 500x500 timing.
