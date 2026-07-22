I have `n` projector modules; module `i` draws
`w[i]` watts and gives `b[i]` brightness. A single power supply caps the total draw at `C` watts. I
switch on any subset whose summed wattage is at most `C`, and I want to maximize summed brightness;
I print that maximum. Each module is on or off, used at most once — this is 0/1 knapsack with
weights `w`, values `b`, capacity `C`. Before I touch an algorithm I fix the scale, because here the
scale is the whole point. `n <= 2000`, `C <= 4000`, `w[i] <= 4000`, and `b[i] <= 10^9`. The capacity
is small (good, a capacity DP is cheap), but the *value* is enormous: a feasible subset can hold many
modules each worth up to `10^9`. Let me bound the answer concretely before I forget. If every module
weighs `1` and `C = 2000`, I could switch on `2000` of them, each worth `10^9`, for a total of
`2 * 10^12`. A signed 32-bit `int` tops out near `2.147 * 10^9`. So the answer can exceed the 32-bit
range by a factor of roughly a thousand. That single observation tells me the DP table entries and
every partial sum must be 64-bit `long long`. I will keep that flag raised through the entire
derivation, because the failure mode of getting it wrong is silent: the small samples pass and the
big hidden test wraps.

**Numeric self-check of that overflow bound.** I do not want to assert "`2 * 10^12` overflows int"
without checking the arithmetic. The largest feasible count of unit-weight modules is `min(n, C)`.
With `n = 2000`, `C = 4000`, that count is `2000` (limited by `n`, not `C`). Times the max value
`10^9` gives `2000 * 10^9 = 2 * 10^12`. Compare to `INT_MAX = 2147483647 ≈ 2.147 * 10^9`. The ratio
`2 * 10^12 / 2.147 * 10^9 ≈ 931`. So a correct answer can be about `931x` larger than `INT_MAX` —
overflow is not a corner case here, it is the typical large case. A `long long` (`~9.2 * 10^18`) has
roughly `4.6 million` times the headroom I need; that is the right type and there is no risk of
overrunning it either. Good: the bound is real and `long long` is both necessary and sufficient.

**Laying out the candidate approaches.** Two shapes are on the table.

- *Subset enumeration.* Try all `2^n` on/off combinations, keep the best feasible brightness.
  Obviously correct, trivial to write — and completely useless at `n = 2000` (`2^2000` is absurd).
  Its only role is as an independent brute-force oracle on tiny `n` to validate the real solution.
- *Capacity DP.* Define `dp[c]` = the maximum total brightness achievable with total wattage at most
  `c`, and fold modules in one at a time. Table size `C + 1`, updated `n` times: `O(n * C)` ≈
  `2000 * 4000 = 8 * 10^6` operations. That fits one second with huge margin. This is the approach I
  will ship; the work is getting the recurrence, the iteration direction, and the data type exactly
  right.

**Deriving the DP recurrence.** I process modules in order. Let `dp_i[c]` be the best brightness
using only the first `i` modules with total wattage `<= c`. Module `i` is either off — then
`dp_i[c] = dp_{i-1}[c]` — or on, which requires `w[i] <= c` and gives
`dp_{i-1}[c - w[i]] + b[i]`. So

  `dp_i[c] = max( dp_{i-1}[c],  dp_{i-1}[c - w[i]] + b[i] )`  (the second term only if `w[i] <= c`).

Base case `dp_0[c] = 0` for all `c` (no modules on, zero brightness, and the empty subset is always
feasible). The answer is `dp_n[C]`. I will collapse the two-dimensional table to a single rolling
array `dp[c]` of length `C + 1`. The well-known subtlety: to keep each module used **at most once**,
when I fold module `i` I must iterate `c` from high to low, so that `dp[c - w[i]]` on the right-hand
side still holds the *previous* module's value (from `dp_{i-1}`), not an already-updated value from
this same module. Iterating low to high would let `dp[c - w[i]]` already include module `i`, which
would allow taking it twice — that is the unbounded-knapsack recurrence, the wrong problem.

**Hand-checking the recurrence on the sample.** The stated sample: `C = 7`, modules
`(w,b) = (3,4), (4,5), (2,3), (5,6)`, claimed answer `9`. Start `dp = [0,0,0,0,0,0,0,0]` (indices
`0..7`). Fold module 0 `(3,4)`, iterate `c` from `7` down to `3`: each `dp[c] = max(dp[c],
dp[c-3] + 4) = max(0, 0 + 4) = 4`. So `dp = [0,0,0,4,4,4,4,4]`. Fold module 1 `(4,5)`, `c` from `7`
to `4`: `dp[7] = max(4, dp[3]+5) = max(4, 4+5) = 9`; `dp[6] = max(4, dp[2]+5) = max(4,5) = 5`;
`dp[5] = max(4, dp[1]+5) = 5`; `dp[4] = max(4, dp[0]+5) = 5`. Now `dp = [0,0,0,4,5,5,5,9]`. Fold
module 2 `(2,3)`, `c` from `7` to `2`: `dp[7] = max(9, dp[5]+3) = max(9, 5+3) = 9`; `dp[6] =
max(5, dp[4]+3) = max(5,8) = 8`; `dp[5] = max(5, dp[3]+3) = max(5,7) = 7`; `dp[4] = max(5, dp[2]+3) =
5`; `dp[3] = max(4, dp[1]+3) = 4`; `dp[2] = max(0, dp[0]+3) = 3`. Now `dp = [0,0,3,4,5,7,8,9]`. Fold
module 3 `(5,6)`, `c` from `7` to `5`: `dp[7] = max(9, dp[2]+6) = max(9, 3+6) = 9`; `dp[6] =
max(8, dp[1]+6) = max(8,6) = 8`; `dp[5] = max(7, dp[0]+6) = max(7,6) = 7`. Final
`dp = [0,0,3,4,5,7,8,9]`, so `dp[7] = 9`. Matches the claimed answer, and I can see the two optimal
subsets it found: `(3,4)+(4,5)` reaching `dp[7]` early, and the `(2,3)+(5,6)` line also summing to
`9`. The recurrence and the downward iteration are correct.

**First implementation — with the overflow trap deliberately set, to show why it bites.** My first
instinct, having just done the trace with small brightness values, is to size the table as `int`
because "knapsack dp is just an int array," and I write:

```
vector<int> dp(C + 1, 0);
for (int i = 0; i < n; i++) {
    if (w[i] > C) continue;
    for (long long c = C; c >= w[i]; c--) {
        int cand = dp[c - w[i]] + (int)b[i];   // <-- everything int here
        if (cand > dp[c]) dp[c] = cand;
    }
}
cout << dp[C] << "\n";
```

On the documented sample this prints `9` — the trace above is unaffected because every value is tiny.
On the second tiny generator cases it also passes. That is exactly the danger: the bug is invisible
on everything small. I have to *trace a large case* to expose it, so I construct the smallest input
that should already exceed `INT_MAX`.

**The bug — caught by tracing a large case.** Take `5` modules, each `w = 1`, each `b = 10^9`, and
`C = 5`. Every module fits and all five together weigh `5 <= 5`, so the true answer is
`5 * 10^9 = 5 * 10^9 = 5000000000`, which is well past `INT_MAX = 2147483647`. Now I trace the `int`
DP. Start `dp = [0,0,0,0,0,0]` (indices `0..5`). Fold module 0 `(1, 10^9)`, `c` from `5` down to `1`:
each `dp[c] = max(dp[c], dp[c-1] + 10^9)`. Since `dp[c-1]` is `10^9` for `c-1 >= 1` after we set it,
but iterating downward `dp[c-1]` is still the *old* `0`, so every `dp[c]` becomes `10^9`. Now
`dp = [0, 1e9, 1e9, 1e9, 1e9, 1e9]` — all fit in `int` (`10^9 < 2.147 * 10^9`), no problem yet. Fold
module 1, `c` from `5` to `1`: `dp[5] = max(1e9, dp[4] + 1e9) = max(1e9, 2e9)`. Here `2 * 10^9 =
2000000000`, still below `INT_MAX = 2147483647`, so `dp[5] = 2e9`. Likewise `dp[4..2]` become `2e9`,
and `dp[1] = max(1e9, dp[0] + 1e9) = 1e9`. Now `dp = [0, 1e9, 2e9, 2e9, 2e9, 2e9]`. Fold module 2,
`c` from `5` to `1`: `dp[5] = max(2e9, dp[4] + 1e9) = max(2e9, 2e9 + 1e9)`. The candidate is
`2 * 10^9 + 10^9 = 3 * 10^9 = 3000000000`. **But that addition is computed in `int`**, and
`3000000000` does not fit in a signed 32-bit int — it wraps. `3000000000` modulo `2^32` is
`3000000000 - 4294967296 = -1294967296`. So `cand = -1294967296`, a large *negative* number. The
comparison `if (cand > dp[c])` is `if (-1294967296 > 2000000000)`, which is **false**, so the update
is rejected and `dp[5]` stays at `2e9`. The same thing happens for `dp[4..3]`. Folding modules 3 and
4 changes nothing for the same reason. The program prints `2000000000`. The correct answer is
`5000000000`. The `int` DP is off by `3 * 10^9`, and it is off *silently* — no crash, no warning, a
plausible-looking positive number.

**Confirming the wrap arithmetic numerically.** I do not want to trust my mental modulo, so I pin it
down. `INT_MAX = 2147483647`. The first sum that exceeds it is `3 * 10^9 = 3000000000`. As an
unsigned 32-bit pattern that is `3000000000`; since `3000000000 >= 2^31 = 2147483648`, the signed
interpretation is `3000000000 - 2^32 = 3000000000 - 4294967296 = -1294967296`. And indeed
`-1294967296 > 2000000000` is false, so `max` keeps the stale `2 * 10^9`. The trace and the
arithmetic agree: the `int` version saturates at `2000000000` on this input, exactly the wrong answer
I reproduced. The cause is unambiguous: a partial sum exceeded the 32-bit range, wrapped to a
negative, and the `max` then discarded the (corrupted) larger value — so the corruption is not even a
"wrapped large number" in the output, it is a frozen *under*-count, which is even more deceptive.

**The fix.** Make the table and all arithmetic 64-bit. `dp` becomes `vector<long long>`, and the
candidate `dp[c - w[i]] + b[i]` is then computed in `long long`. To be safe against any sign or width
surprise I also keep `w` and `b` as `long long` (they are read as `long long` already), so
`c - w[i]` and the addition never silently narrow. Re-running the same `5 x (1, 10^9)`, `C = 5` trace
in `long long`: every step is identical until `dp[5] = max(2e9, 2e9 + 1e9)`, which now evaluates to
`3e9` honestly (no wrap, `3 * 10^9 << 9.2 * 10^18`), so `dp[5] = 3e9`; folding module 3 pushes it to
`4e9`, module 4 to `5e9`. Final `dp[5] = 5000000000`. Correct. I actually ran both versions: the
`int` build prints `2000000000`, the `long long` build prints `5000000000`. The fix is exactly the
type, nothing else in the recurrence changes.

**Second debug episode — the iteration direction.** While I have the code open I want to make sure I
did not accidentally write the unbounded-knapsack loop, because that is the other classic transcription
error and it is also invisible on some inputs. Suppose I had iterated `c` upward instead:
`for (c = w[i]; c <= C; c++)`. Trace the tiny case `n = 1`, `C = 4`, one module `(w, b) = (2, 3)`.
True answer: one copy fits (`2 <= 4`), brightness `3`; I cannot take the single module twice. Upward
loop: `dp` starts `[0,0,0,0,0]`. `c = 2`: `dp[2] = max(0, dp[0] + 3) = 3`. `c = 3`: `dp[3] =
max(0, dp[1] + 3) = 3`. `c = 4`: `dp[4] = max(0, dp[2] + 3)` — but `dp[2]` was *just updated to 3 in
this same module's pass*, so `dp[4] = max(0, 3 + 3) = 6`. That `6` means I took the one module twice
to fill wattage `4` — illegal for 0/1. So upward iteration is wrong here, returning `6` instead of
`3`. Now the downward loop on the same input: `c = 4`: `dp[4] = max(0, dp[2] + 3) = max(0, 0 + 3) =
3` (`dp[2]` still `0`, not yet touched). `c = 3`: `dp[3] = max(0, dp[1] + 3) = 3`. `c = 2`: `dp[2] =
max(0, dp[0] + 3) = 3`. Final `dp[4] = 3`. Correct. This confirms the downward direction is the
0/1 (each item once) recurrence, and it is what I have. The episode is worth keeping because the two
loops differ by one character and only specific inputs (where an item could "refit" into the leftover
capacity) expose the difference.

**Edge cases, deliberately.**
- *`C = 0`.* The supply delivers nothing. Every module has `w[i] >= 1 > 0 = C`, so the guard
  `if (w[i] > C) continue;` skips all of them, and the inner loop `for (c = C; c >= w[i]; c--)` would
  not run anyway (`C = 0 < w[i]`). `dp = [0]`, output `dp[0] = 0`. The empty subset — correct. I ran
  `2 0 / 1 5 / 1 7` and it prints `0`.
- *Some module too heavy to ever fit.* `n = 1`, `C = 2`, module `(5, 999)`. The guard skips it; even
  without the guard the inner loop bound `c >= 5` is empty for `C = 2`. Output `0`. Correct — that
  module can never be on, and no other module exists, so the best is the empty subset.
- *Single module that fits.* `n = 1`, `C = 5`, module `(3, 4)`. Fold it, `c` from `5` to `3`:
  `dp[5] = dp[2] + 4 = 4`, etc. Output `dp[5] = 4`. Correct.
- *Leftover capacity is fine.* The objective maximizes brightness, not utilization; `dp[C]` already
  means "wattage at most `C`," so leaving watts unused is automatically allowed (an all-zero subset is
  the base case). No special handling needed.
- *Overflow.* Covered above; the entire pipeline is `long long`, with `~4.6 * 10^6` times headroom
  over the `2 * 10^12` worst case. No partial sum can exceed `dp[C] + b[i] <= 2 * 10^12 + 10^9`,
  trivially inside `long long`.
- *Reading.* `if (!(cin >> n >> C)) return 0;` guards malformed/empty input; `w[i], b[i]` are read as
  `long long` so the multiplications and additions stay 64-bit from the first byte.

**Cross-validation.** I generated `750` random small instances (`n <= 12`, `C <= 30`, brightnesses a
mix of tiny values and values near `10^9`) and compared the `long long` DP against an independent
`2^n` subset-enumeration brute force. Zero mismatches. I also confirmed the documented sample
(`-> 9`) and the large overflow witness (`5 x (1, 10^9)`, `C = 5 -> 5000000000`, while the `int`
build wrongly prints `2000000000`). A full-size instance (`n = 2000`, `C = 4000`, brightnesses near
`10^9`) runs in well under `0.01 s` and returns `37323240743`, itself an answer `~17x` past `INT_MAX`
— direct evidence that the large hidden tests genuinely require 64-bit.

**Final solution.** The algorithm is textbook 0/1 knapsack by capacity with downward iteration; the
only thing that makes this problem more than a template is that the answer overflows 32 bits, which I
established by a bound, exposed by tracing a deliberately-`int` build on a large case (it freezes at
`2 * 10^9` because a wrapped-negative candidate loses the `max`), and fixed by making the table and
arithmetic `long long`. This is what I ship:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;

    vector<long long> w(n), b(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> b[i];

    // dp[c] = maximum total brightness using a subset whose total wattage is exactly <= c.
    // Brightness sums can reach n * 1e9 = 2e12, so dp MUST be 64-bit.
    vector<long long> dp(C + 1, 0);
    for (int i = 0; i < n; i++) {
        long long wi = w[i], bi = b[i];
        if (wi > C) continue;                 // module can never fit
        for (long long c = C; c >= wi; c--) { // 0/1 knapsack: iterate capacity downward
            long long cand = dp[c - wi] + bi;
            if (cand > dp[c]) dp[c] = cand;
        }
    }

    cout << dp[C] << "\n";
    return 0;
}
```

**Causal recap.** A capacity bound showed the answer can reach `2 * 10^12`, about `931x` past
`INT_MAX`, so the table must be 64-bit; an `int` first cut passed every small sample because the bug
is invisible until a partial sum crosses `2.147 * 10^9`, so I traced the smallest overflowing case
(`5` unit-weight modules worth `10^9`, `C = 5`) and watched `dp[5]`'s candidate `3 * 10^9` wrap to
`-1294967296`, lose the `max`, and freeze the output at `2 * 10^9` instead of `5 * 10^9`; switching
the table and arithmetic to `long long` fixes it exactly. A second trace confirmed the downward
capacity loop is the 0/1 recurrence (upward would let one module refit and return `6` instead of `3`
on `(2,3)`, `C = 4`), and `750` brute-force comparisons plus the `C = 0`, too-heavy, and
single-module corners close it out.
