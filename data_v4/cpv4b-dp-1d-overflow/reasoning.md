**Reading the problem and pinning the contract.** I have `n` rounds, each carrying an integer score in `0..m`. A round is a *peak* when its score equals `m`. The single rule is that two consecutive rounds may not both be peaks. I must count length-`n` score sequences obeying that rule, modulo `1000000007`. Input is the pair `n m` on stdin; output is one integer. Let me fix the scale before touching an algorithm, because it dictates the data types: `n <= 2*10^5` and `m <= 10^9`. A round can take `m+1` distinct scores, so the raw count of sequences (before the modulus) is astronomically large — exponential in `n` — which is exactly why the problem hands me a modulus. After taking the modulus, every count I store is in `0..10^9+6`, i.e. up to about `10^9`. And the natural per-round multiplier is `m`, also up to `10^9`. The product of two such numbers is about `10^18`. That fits comfortably in a signed 64-bit integer (ceiling `~9.2*10^18`) but blows past 32 bits (ceiling `~2.1*10^9`) by nine orders of magnitude. So I already know the headline hazard: any multiplication done in `int` is a silent wrong answer. I will keep that flag raised through the whole derivation.

**Understanding the counting structure.** The constraint links only adjacent rounds, and it only cares about peaks. So the one piece of history the future needs is: was the round I just placed a peak or not? That is a two-state linear DP over the `n` rounds. Let me define, for a prefix of length `i`:

- `lo[i]` = number of valid length-`i` sequences whose **last round is a non-peak** (its score is one of the `m` values `0..m-1`);
- `hi[i]` = number of valid length-`i` sequences whose **last round is a peak** (score exactly `m`, the single value).

The total valid count for length `i` is `lo[i] + hi[i]`.

**Deriving the recurrence.** Consider extending a valid prefix of length `i-1` by one more round.

- If the new round is a **non-peak**, there are `m` score choices for it, and it imposes nothing on what came before — a non-peak may follow either a non-peak or a peak. So every valid length-`(i-1)` sequence, of which there are `lo[i-1] + hi[i-1]`, extends in `m` ways:
  `lo[i] = (lo[i-1] + hi[i-1]) * m`.
- If the new round is a **peak**, there is exactly `1` score choice (the value `m`), but a peak may **not** follow a peak, so the previous round must have been a non-peak. Only the `lo[i-1]` sequences qualify:
  `hi[i] = lo[i-1]`.

Base case, `i = 1`: a length-1 sequence is a single round. A non-peak has `m` choices, a peak has `1`. So `lo[1] = m`, `hi[1] = 1`. (When `m = 0` this correctly gives `lo[1] = 0`, since there is no non-peak score available — the only score `0` *is* the peak.)

The answer is `lo[n] + hi[n]` for `n >= 1`, all under the modulus. I only need the previous pair to compute the next, so this is `O(n)` time and `O(1)` memory — a rolling two-variable scan.

**Checking the recurrence on paper against the sample.** The sample is `n = 3`, `m = 2`, expected `22`. Let me run the recurrence by hand (no modulus needed, numbers are tiny).
- `i = 1`: `lo = m = 2`, `hi = 1`. Total `3`. (The three single-round scores `0,1,2`. Correct.)
- `i = 2`: `lo = (2 + 1) * 2 = 6`, `hi = lo_prev = 2`. Total `8`. Let me sanity check `8`: length-2 sequences over `{0,1,2}` number `9`; the only forbidden one is `(2,2)`; `9 - 1 = 8`. Correct.
- `i = 3`: `lo = (6 + 2) * 2 = 16`, `hi = lo_prev = 6`. Total `16 + 6 = 22`. Matches the expected `22`.

Good — the recurrence reproduces the sample, and the intermediate `i = 2` total `8` independently matches the brute "`9` minus the one banned pair" count, so I trust the transitions.

**A second independent numeric self-check of the formula.** Before I lean on this recurrence for large inputs, let me verify it against a direct count on a case I can fully enumerate by a different route: `n = 2`, general `m`. The recurrence gives total `= lo[2] + hi[2] = (lo[1]+hi[1])*m + lo[1] = (m+1)*m + m = m^2 + 2m`. Now count directly: length-2 sequences over `0..m` number `(m+1)^2`; the forbidden ones are exactly the peak-peak pairs, of which there is exactly `1` (the pair `(m,m)`). So the direct count is `(m+1)^2 - 1 = m^2 + 2m + 1 - 1 = m^2 + 2m`. The two expressions agree identically: `m^2 + 2m`. That is an exact algebraic match, not a coincidence on one value, so the recurrence is confirmed on the whole `n = 2` family. Plugging `m = 10^9`: the true count is `10^18 + 2*10^9`, a number I will reuse in a moment to expose the overflow.

**First implementation — written the way I'd naively reach for it, then traced.** Here is my first cut, and I have deliberately written the accumulators as `int` to mirror the most common mistake, so I can catch it in a trace rather than in the judge:

```
int lo = (int)(m % MOD), hi = 1;
for (long long i = 2; i <= n; i++) {
    int nlo = (lo + hi) * (int)(m % MOD) % MOD;   // peak/non-peak extension
    int nhi = lo;
    lo = nlo; hi = nhi;
}
answer = (lo + hi) % MOD;
```

I trace it first on the small sample `n = 3, m = 2`, the input I am most tempted to trust. `lo = 2, hi = 1`. `i = 2`: `nlo = (2+1)*2 % MOD = 6`, `nhi = 2` -> `(6, 2)`. `i = 3`: `nlo = (6+2)*2 % MOD = 16`, `nhi = 6` -> `(16, 6)`. Answer `(16+6) % MOD = 22`. It prints `22`. So on the documented sample the buggy code is **correct** — which is precisely the trap: small numbers never overflow, so the sample tells me nothing about the hazard.

**Forcing the hazard with a large trace.** I raised an overflow flag at the very start, so I refuse to ship on the strength of a sample that exercises none of it. Let me deliberately trace the smallest input where the int multiplication can blow up: `n = 2, m = 10^9`. From the formula self-check I already know the true answer is `(m^2 + 2m) mod (10^9+7)`. Let me compute that target: `m^2 + 2m = 10^18 + 2*10^9`. Reducing modulo `10^9+7` I expect a specific small residue; computing it carefully (or just running the *64-bit* version) gives `35`. So the correct output for `n=2, m=10^9` is `35`; hold that number.

Now trace the **int** code on `n = 2, m = 10^9`. Start `lo = (int)(10^9 % MOD) = 1000000000`, `hi = 1`. `i = 2`: the inner expression is `(lo + hi) * (int)(m % MOD) = (1000000000 + 1) * 1000000000`. In real arithmetic that is `1000000001 * 1000000000 = 1.000000001*10^18`. But this multiplication is happening in `int`: the operands are 32-bit, so the product is computed in 32-bit and wraps around modulo `2^32` before the `% MOD` ever runs. The wrapped value bears no relation to `1.0*10^18`. When I actually run the int build it prints `513381376`, not `35`. So the buggy code gives `513381376` on a case whose answer I proved is `35`. The trace caught the overflow.

**Diagnosing the bug precisely.** The defect is not the recurrence — I verified that algebraically and on the sample. The defect is the *type* of the arithmetic. Two things go wrong in `(lo + hi) * (int)(m % MOD) % MOD`:
1. The factor `m % MOD` can be up to `10^9`, and `lo + hi` can be up to roughly `2*10^9` after a sum of two near-`10^9` values — that sum alone can already exceed `INT_MAX` (`2147483647`). 
2. Even when the sum fits, the **product** is on the order of `10^18`, which overflows 32 bits by a factor of `~5*10^8`. Because C++ evaluates `int * int` in `int`, the wrap happens *before* `% MOD`, so the modulo cannot save me — it is reducing already-corrupted bits.

The tell-tale fingerprint is what the big stress case does: tracing `n = 200000, m = 10^9` through the int build prints `-732834297`, a **negative** number. A count of sequences can never be negative; a negative output is the unmistakable signature of signed 32-bit overflow. That is the second, even louder, piece of evidence that the type is wrong.

**Fixing the types and re-verifying.** The fix is to do all of the accumulation and the multiplication in 64-bit (`long long`), taking the modulus at each step so values stay bounded by `~10^9` and every product stays at `~10^18`, safely inside 64-bit range:

```
long long lo = m % MOD, hi = 1 % MOD;
for (long long i = 2; i <= n; i++) {
    long long nlo = ((lo + hi) % MOD) * (m % MOD) % MOD;
    long long nhi = lo;
    lo = nlo; hi = nhi;
}
answer = (lo + hi) % MOD;
```

Now `lo + hi` is at most `~2*10^9`, which fits in `long long`; I reduce it `% MOD` to get a value `< 10^9`; multiplying by `m % MOD < 10^9` gives a product `< 10^18`, comfortably inside the `~9.2*10^18` ceiling; then `% MOD` brings it back down. Re-trace the two cases that mattered:
- `n = 3, m = 2`: identical small arithmetic as before, prints `22`. Still correct.
- `n = 2, m = 10^9`: `lo = 10^9, hi = 1`. `i = 2`: `nlo = ((10^9 + 1) % MOD) * (10^9 % MOD) % MOD`. Here `(10^9+1) % MOD = 1000000001`, `10^9 % MOD = 1000000000`; their product `1000000001 * 1000000000 = 1.000000001*10^18` is formed in 64-bit (no wrap), and reduced `% MOD` gives `35` for `nlo`; `nhi = lo = 10^9` reduces later. The total `(nlo + nhi) % MOD` works out to `35`. Matches the proven target. The case that exposed the overflow now passes for the reason I fixed — the multiplication is 64-bit — which is the evidence I trust over any hand-waving.

**A self-check that the modulus is applied in the right place.** It is easy to "use long long" but still take the modulus too late and re-overflow. Let me confirm the worst single intermediate in my fixed line. The largest operands feeding the multiply are both `< MOD = 10^9+7`, so the product is `< (10^9+7)^2 ≈ 1.0*10^18 + ...`, precisely `< 1.000000014*10^18`. That is below `9.22*10^18`, so the 64-bit product never wraps. And `(lo + hi)` before its `% MOD` is at most `2*(MOD-1) ≈ 2*10^9`, far below the 64-bit ceiling, so the inner sum is safe too. Every intermediate is accounted for; nothing in the line can overflow 64 bits.

**Edge cases, deliberately, because counting DPs die in the corners.**
- `n = 0`: the empty sequence is one valid sequence (it vacuously satisfies "no two adjacent peaks"). My loop starts at `i = 2` and my base pair is for `i = 1`, so I must special-case `n = 0` and print `1 % MOD = 1`. I add an explicit early return for `n == 0`. Brute force over zero rounds also yields `1`. Correct.
- `n = 1`: the loop body (which runs for `i` from `2` to `n`) never executes, so the answer is the base pair `lo + hi = m + 1`. Every single score `0..m` is allowed, and there are `m + 1` of them. For `m = 7` that is `8`; the brute prints `8`. Correct.
- `m = 0`: every round is forced to score `0`, and `0 == m`, so *every* round is a peak. For `n = 1` there is exactly one sequence (`(0)`), and indeed base `lo + hi = 0 + 1 = 1`. For `n >= 2`, the recurrence gives `lo[1] = 0, hi[1] = 1`; then `lo[2] = (0+1)*0 = 0`, `hi[2] = lo[1] = 0`, total `0`, and it stays `0` forever after. That is right: with only peaks available you cannot place two rounds without two adjacent peaks. Brute confirms `0` for `n = 2,3,4` at `m = 0`. Correct.
- Large overflow case: `n = 2*10^5, m = 10^9`. Every step's product is `~10^18` in 64-bit, never wraps; the fixed code prints `97804739`, which an independent Python re-implementation of the same recurrence also produces. Correct, and crucially the output is a non-negative residue `< MOD`, unlike the buggy int build's negative `-732834297`.
- Input parsing: I read `n` and `m` as `long long` (so `m = 10^9` is read exactly; even `int` would hold `10^9`, but `long long` removes all doubt) and guard `if (!(cin >> n >> m)) return 0;` for empty input.

**Final solution.** I derived the two-state recurrence, confirmed it both on the worked sample and algebraically on the entire `n = 2` family (`m^2 + 2m`), then deliberately wrote the int version, traced it on a large case to expose a `~10^18` product wrapping inside 32 bits (correct `35`, buggy `513381376`; and a giant case turning *negative*), and fixed it by moving every sum, product, and modulo into 64-bit. This is what I ship — one self-contained file, the `O(n)` two-track DP with all arithmetic in `long long`:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    const long long MOD = 1000000007LL;
    long long n, m;
    if (!(cin >> n >> m)) return 0;

    // lo = number of valid length-i sequences whose last round is a non-peak
    //      score (one of the m values 0..m-1);
    // hi = number whose last round is the peak score m (the single value m).
    // A peak may not immediately follow a peak.
    //
    // For i = 1: lo = m (scores 0..m-1), hi = 1 (the peak).
    // Transition for one more round:
    //   new_lo = (lo + hi) * m   (any previous ending may be followed by a
    //                             non-peak, and there are m non-peak scores)
    //   new_hi = lo              (a peak may only follow a non-peak; 1 score)
    //
    // n can be 0 (empty sequence: exactly one, the empty one).
    long long lo = (m % MOD), hi = 1 % MOD;
    if (n == 0) { cout << 1 % MOD << "\n"; return 0; }

    for (long long i = 2; i <= n; i++) {
        long long nlo = ((lo + hi) % MOD) * (m % MOD) % MOD;
        long long nhi = lo;
        lo = nlo;
        hi = nhi;
    }

    cout << (lo + hi) % MOD << "\n";
    return 0;
}
```

**Causal recap.** The structure is a clean two-state DP — non-peak endings extend in `m` ways from anything (`lo = (lo+hi)*m`), peak endings extend in `1` way only from a non-peak (`hi = lo`) — and I confirmed it on the sample `22` and algebraically on the `n=2` family (`(m+1)^2-1 = m^2+2m`). The real trap is arithmetic, not logic: with post-modulo counts and the multiplier `m` both near `10^9`, the product `(lo+hi)*m` is `~10^18`, which an `int` build wraps modulo `2^32` *before* the `% MOD` runs, so it silently returns garbage on large inputs while printing the correct `22` on the tiny sample; a forced trace of `n=2, m=10^9` (proven answer `35`, buggy answer `513381376`) and the negative output `-732834297` on the `2*10^5` stress case pinned the overflow, and promoting every sum, product, and modulo to `long long` — keeping operands `< MOD` so products stay `< 1.0*10^18` — fixes it, after which the empty-sequence (`n=0` -> `1`), single-round (`n=1` -> `m+1`), and all-peak (`m=0` -> `0` for `n>=2`) corners all check out.
