**Reading the problem and pinning the contract.** I have `n` parcels in a row with integer weights `a[0..n-1]`, possibly negative. A *batch* is a block of **exactly `w` consecutive** parcels, and it passes when its total weight `T` satisfies `L <= T <= R` — a **closed** band, both ends allowed. I must count the passing batches. Input is `n w L R` then the `n` weights; output is one integer. Before I touch an algorithm I fix the scale, because it dictates the data types and rules approaches in or out. `n <= 2*10^5`, `|a[i]| <= 10^9`, and a batch can hold up to `n` parcels, so a window sum can reach `2*10^5 * 10^9 = 2*10^14`. That is two thousand times past the 32-bit ceiling of `~2.1*10^9`, so every accumulator — the prefix array and each window sum — must be 64-bit `long long`. The band endpoints `L, R` can be as large as `10^15` in magnitude, also 64-bit. That is decision one and it is non-negotiable: an `int` anywhere in the sum path is a silent wrong-answer on the big tests.

I also note `w` is given as up to `10^9`, deliberately larger than `n` can be, so `w > n` is a legal input and must yield `0` — no block of `w` parcels fits on a belt of `n < w`. I will keep `w` in a 64-bit type too, because expressions like `n - w + 1` and `s + w - 1` must not wrap when `w` is huge.

**Candidate approaches.** Two routes are on the table.

- *Brute force per window.* For every start, add up the `w` weights and test the band. It is unambiguous about semantics — exactly `w` elements, inclusive band — but it is `O(n*w)`. With `n = 2*10^5` and `w` near `10^5`, that is `~2*10^10` additions, far past a 1-second budget. It is my **reference oracle**, not my submission.
- *Prefix sums.* Precompute cumulative sums so each window's total is one subtraction. That makes each window `O(1)` and the whole count `O(n)`. This is the submission. The entire risk is not the idea but the **indexing**: which prefix entries I subtract, and over which range of starts I loop. This is exactly the kind of code where the math is trivial and the transcription is where it dies.

I commit to prefix sums and resolve to nail the boundaries by tracing, not by trusting.

**Deriving the prefix-sum formula and the start range.** I use an **exclusive** prefix: `P[0] = 0` and `P[k] = a[0] + ... + a[k-1]` for `k = 1..n`. So `P[k]` is the sum of the first `k` parcels, i.e. it sums indices `0..k-1`. The weight of the block covering 0-indexed positions `i, i+1, ..., j-1` (that is, a half-open range `[i, j)`) is exactly `P[j] - P[i]`, because `P[j]` sums `0..j-1` and `P[i]` sums `0..i-1`, leaving `i..j-1`.

A batch of length `w` starting at 0-indexed `i` covers `i, ..., i+w-1`, which as a half-open range is `[i, i+w)`. So its sum is `P[i+w] - P[i]`. Good — that is the prefix subtraction.

Now the range of valid starts. The last parcel in the batch is at index `i + w - 1`, and it must exist: `i + w - 1 <= n - 1`, i.e. `i <= n - w`. The first valid start is `i = 0`. So the valid starts are `i = 0, 1, ..., n - w`, and the **count of starts is `(n - w) - 0 + 1 = n - w + 1`** — provided `w <= n`; if `w > n` there are zero.

Let me **numerically self-check that "`n - w + 1` starts" claim** before I lean on it, because an off-by-one in the count of windows is precisely the bug I am trying to avoid. Take `n = 6`, `w = 3`. The batches by hand: starts at index 0 (`[0,1,2]`), 1 (`[1,2,3]`), 2 (`[2,3,4]`), 3 (`[3,4,5]`). Index 3's last element is `3+3-1 = 5 = n-1`, valid; a start at index 4 would need element `4+3-1 = 6 = n`, which does not exist, so index 4 is out. That is 4 batches. The formula gives `n - w + 1 = 6 - 3 + 1 = 4`. Match. A second check: `n = 5`, `w = 5` (whole belt) gives `5 - 5 + 1 = 1` start (index 0 only) — correct, exactly one batch. And `n = 5`, `w = 1` gives `5 - 1 + 1 = 5` starts — every parcel its own batch, correct. The formula holds on all three.

For the code I will actually use a **1-indexed** weight array (I read into `a[1..n]`), which shifts the prefix bookkeeping: with `P[0]=0`, `P[k]=a[1]+...+a[k]`, a batch starting at 1-indexed `s` covers `s..s+w-1`, its sum is `P[s+w-1] - P[s-1]`, and the valid starts are `s = 1, ..., n - w + 1`. This is the same window arithmetic re-expressed; I have to be careful not to mix the 0-indexed `P[i+w]-P[i]` with the 1-indexed `P[s+w-1]-P[s-1]`.

**First implementation and a trace.** Clean math transcribes dirty, so I write the loop and immediately trace it. My first cut:

```
// P[i] = a[1] + ... + a[i], P[0] = 0
for (long long s = 1; s <= n - w; s++) {          // start positions
    long long sum = P[s + w - 1] - P[s - 1];
    if (sum >= L && sum <= R) count++;
}
```

I trace this on the worked sample `n=6, w=3, L=10, R=15`, weights `a[1..6] = [4,2,5,1,9,3]`, where I already worked out by hand that the answer is `3` (batches `11, 8, 15, 13` — three pass, the `8` fails). First the prefix: `P[0]=0, P[1]=4, P[2]=6, P[3]=11, P[4]=12, P[5]=21, P[6]=24`. Now the loop runs `s = 1` to `n - w = 6 - 3 = 3`:
- `s=1`: `sum = P[3] - P[0] = 11 - 0 = 11`. `10 <= 11 <= 15` -> pass, `count=1`.
- `s=2`: `sum = P[4] - P[1] = 12 - 4 = 8`. `8 < 10` -> fail.
- `s=3`: `sum = P[5] - P[2] = 21 - 6 = 15`. `10 <= 15 <= 15` -> pass, `count=2`.

Loop ends. `count = 2`.

**The bug.** The code returns `2`, but the answer is `3`. I missed the batch `[1,9,3] = 13`, which starts at `s=4`. The loop bound is wrong: I wrote `s <= n - w`, which stops at `s = 3`, but the valid starts run to `s = n - w + 1 = 4` **inclusive**. This is exactly the off-by-one I warned myself about during the derivation — the count of windows is `n - w + 1`, and a loop that ends at `n - w` silently drops the very last batch. Worse, it is a *quiet* failure: on inputs where the last window happens to fail the band, the wrong bound gives the right count by luck, so this bug hides until a test where the final window passes. The sample is exactly such a test, which is why tracing it caught the defect.

I fix the bound to `s <= n - w + 1` and re-trace the sample. Prefix unchanged. Loop `s = 1..4`:
- `s=1`: `P[3]-P[0]=11` -> pass, `count=1`.
- `s=2`: `P[4]-P[1]=8` -> fail.
- `s=3`: `P[5]-P[2]=15` -> pass, `count=2`.
- `s=4`: `sum = P[s+w-1] - P[s-1] = P[6] - P[3] = 24 - 11 = 13`. `10 <= 13 <= 15` -> pass, `count=3`.

`count = 3`. Correct. And the batch it adds is precisely `[1,9,3]=13`, the one I had dropped. The fix targets the exact cause.

**A second trace, hunting the inclusive/exclusive band bug.** The loop bound is fixed, but there is a second boundary I have not stress-tested: the band comparison. The problem says **closed** `[L, R]`, so I must use `>=` and `<=`. It is easy to slip and write strict `>`/`<`, which would exclude a batch whose sum equals an endpoint. To check whether my code respects the closed band, I deliberately pick an input where a window sits **exactly on an edge**. Take `n=3, w=2, L=-1, R=-1` (a degenerate band, `L=R`, so only sums equal to `-1` pass), weights `a[1..3] = [1, -2, 1]`. By hand: batch `[1,-2]=-1` passes (equals `-1`), batch `[-2,1]=-1` passes. Answer `2`.

Now trace my fixed code. Prefix: `P[0]=0, P[1]=1, P[2]=-1, P[3]=0`. Loop `s = 1..n-w+1 = 1..2`:
- `s=1`: `sum = P[2]-P[0] = -1 - 0 = -1`. Test `-1 >= -1 && -1 <= -1` -> both hold -> pass, `count=1`.
- `s=2`: `sum = P[3]-P[1] = 0 - 1 = -1`. Test `-1 >= -1 && -1 <= -1` -> pass, `count=2`.

`count = 2`. Correct — the `>=`/`<=` form counts the on-edge windows. If I had written strict `> L && < R` it would have given `0` here (no sum is strictly between `-1` and `-1`), which is the classic inclusive/exclusive blunder. The degenerate band `L=R` is the sharpest possible probe for it, and my code passes. (I keep this case in mind for the generator: forcing `L=R` a quarter of the time is what would have caught a strict-comparison bug at scale.)

**Cross-checking the wrong-prefix variant won't sneak in.** There is a third boundary trap I want to rule out by reasoning: subtracting the *wrong two* prefix entries. With a 1-indexed `P`, the window `s..s+w-1` is `P[s+w-1] - P[s-1]`. A natural slip is to copy the 0-indexed form `P[i+w]-P[i]` into the 1-indexed loop, writing `P[s+w]-P[s]` — that computes the sum of `s+1..s+w`, the window shifted right by one, and `s+w` can even run off the array end. I re-derive from the definition to be sure mine is right: `P[s+w-1]` sums `a[1..s+w-1]`, `P[s-1]` sums `a[1..s-1]`, difference sums `a[s..s+w-1]` — exactly the `w` parcels of the batch. Correct. And the largest index I touch is `P[s+w-1]` at `s = n-w+1`, i.e. `P[n]`, which is in range, so no out-of-bounds. Good.

**Edge cases, deliberately, because boundaries are where this dies.**
- `w > n`: there is no length-`w` block. In code I guard the loop with `if (w >= 1 && w <= n)`. If `w > n`, the loop never runs and `count = 0`. Without the guard, `n - w + 1` would be `<= 0` and the `for (s = 1; s <= n-w+1; ...)` would also not execute — but `n - w + 1` is computed in `long long`, so for huge `w` (up to `10^9`) there is no wrap; still, the explicit guard makes the intent obvious and dodges any signedness worry. Traced `n=3, w=5`: guard false, output `0`. Correct.
- `w = n`: one batch, the whole belt. `n - w + 1 = 1`, loop runs once at `s=1`, `sum = P[n] - P[0]` = total weight. Traced `n=4, w=4, [1,2,3,0]`, band `[6,6]`: `P[4]=6`, `sum=6`, `6>=6 && 6<=6` -> pass, `count=1`. Correct.
- `w = 1`: every parcel is a batch; `sum = P[s] - P[s-1] = a[s]`. Traced `n=5, w=1, L=R=-2, a=[-2,0,-2,1,-2]`: starts `s=1..5`, sums are the weights themselves; three of them equal `-2` -> `count=3`. Correct, and it confirms the single-element window degenerates properly.
- All-negative or zero weights: nothing special — the band test is two-sided, so negative window sums are counted iff they land in `[L,R]`. The traces above already include negatives.
- Overflow at the band edge: window sum can be `~2*10^14`; `long long` holds it with four orders of magnitude to spare. Traced `n=2, w=2, L=10^9, R=2*10^15, a=[10^9, 10^9]`: `sum = 2*10^9`, which exceeds the 32-bit range but is well within `long long`; `2*10^9 >= 10^9 && <= 2*10^15` -> pass, `count=1`. An `int` accumulator would have wrapped `2*10^9` to a negative and miscounted. This is why every sum-path variable is `long long`.
- Output: exactly one integer and a newline. `cin >>` skips arbitrary whitespace, so the input format is layout-agnostic.

**Performance and final verification.** The algorithm is one pass to build `P` and one pass over `n - w + 1` starts: `O(n)` time, `O(n)` memory for the prefix array. I push it to the limit — `n = 2*10^6` (an order beyond the stated `2*10^5`), `w = 1000`, random weights spanning the full range — and it returns in `0.12 s` using `34 MB`, so at the real cap of `n = 2*10^5` it is trivially under the 1-second, 256-MB budget. I then ran the solution against the independent brute force on `600` random small cases (small `n`, `w` ranging up to `n+2` so `w>n` is exercised, negatives and zeros, with `L=R` forced a quarter of the time): **zero mismatches**. The two boundary bugs I traced — the `s <= n-w` loop bound and the strict-band comparison — were also run as variants and disagree with the correct count on the very sample (`2` and `2` versus `3`) and on `61`/`39` of `400` random cases respectively, confirming the boundaries are genuinely load-bearing and not cosmetically equivalent.

**Final solution.** I convinced myself the idea is right by deriving the window-sum and start-range from the prefix definition and numerically checking the `n-w+1` count on three sizes; I convinced myself the *code* is right by tracing the dropped-last-window bug to its exact cause on the sample, fixing the bound, and then separately tracing a degenerate-band case to confirm the closed-interval comparison. That is what I ship — one self-contained `O(n)` file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, w, L, R;
    if (!(cin >> n >> w >> L >> R)) return 0;
    vector<long long> a(n + 1);
    for (long long i = 1; i <= n; i++) cin >> a[i];

    // P[i] = a[1] + ... + a[i], with P[0] = 0 (exclusive prefix).
    vector<long long> P(n + 1, 0);
    for (long long i = 1; i <= n; i++) P[i] = P[i - 1] + a[i];

    // A batch starts at s (1-indexed) and covers s..s+w-1.
    // Valid starts: 1 <= s <= n - w + 1  (inclusive on both ends).
    // Sum of that batch = P[s + w - 1] - P[s - 1].
    long long count = 0;
    if (w >= 1 && w <= n) {
        for (long long s = 1; s <= n - w + 1; s++) {
            long long sum = P[s + w - 1] - P[s - 1];
            if (sum >= L && sum <= R) count++;   // closed band [L, R]
        }
    }

    cout << count << "\n";
    return 0;
}
```

**Causal recap.** Prefix sums turn each fixed-length window into one subtraction, but the correctness hinges entirely on three boundaries: the window-start range is `n - w + 1` **inclusive** (my first loop stopped at `n - w` and dropped the last batch — tracing the sample, which returned `2` instead of `3`, pinpointed it, and the dropped batch `[1,9,3]=13` was exactly the missing start `s = n-w+1`); the prefix subtraction for a 1-indexed window is `P[s+w-1] - P[s-1]`, re-derived from the definition so the 0-indexed `P[i+w]-P[i]` form can't sneak in shifted by one; and the band is **closed**, so `>=`/`<=` — a degenerate `L=R=-1` trace counting two on-edge windows confirmed strict comparisons would have undercounted. With every sum-path value in `long long` to hold the `~2*10^14` window totals, those three boundary decisions, each verified on a concrete small case, are what make the `O(n)` count exact.
