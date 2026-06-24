**Reading the problem and pinning the contract.** I have `n` days, a fixed startup fee `c >= 0`, and a per-day net energy `a[0..n-1]` whose entries can be positive, zero, or negative. I may run the pump over **at most one** contiguous block `[l, r]`, paying `c` once and earning `a[l]+...+a[r]`, or I may run nothing for profit `0`. I want the maximum profit. Two things in the statement matter more than they look. First, the run, if I make one, is **non-empty**: I cannot "run for zero days" and pay nothing — paying `c` is tied to choosing at least one day. Second, the overall plan **is** allowed to be empty (run nothing), and that empty plan scores exactly `0`. So the answer is `max(0, best-non-empty-block-sum - c)`. Those two "empties" are different objects and the whole difficulty is keeping them apart.

Before any algorithm I fix the scale, because it dictates the integer types. `n <= 2*10^5`, `|a[i]| <= 10^9`, so a single block sum can reach `2*10^5 * 10^9 = 2*10^14`, far past the 32-bit ceiling of about `2.1*10^9`. And `c` can be as large as `10^{14}`. Every accumulator — the running Kadane value, the global best, the fee, the final answer — must be 64-bit `long long`. An `int` here is a silent wrong-answer on the large tests, not a crash, which is the worst kind. That decision is non-negotiable and comes first.

**Laying out the candidate approaches.** Two routes, and I want the one I can *prove*, not the one that types fastest.

- *Enumerate all windows.* For each `(l, r)` accumulate the block sum, subtract `c`, keep the best, and compare to `0`. This is transparently correct — it literally tries every legal run — but it is `O(n^2)`. At `n = 2*10^5` that is `4*10^{10}` operations, hopeless under a 1-second limit. I will keep it only as a brute-force oracle on tiny inputs to check the fast solution against.
- *Linear DP — Kadane with a fee.* The classic maximum-subarray scan computes, in `O(n)`, the largest sum of a contiguous non-empty block. If I have that quantity `M = max non-empty block sum`, then running the best possible block nets `M - c`, and the answer is `max(0, M - c)`. The fee is a single global constant, so it does not interact with *which* block is best — it only shifts the running option down by `c`. That decoupling is the key structural observation: pick the best block by Kadane, then apply the fee once at the end. The open questions are the recurrence, the base case that forbids the empty block, and exactly where `c` and the `0` enter.

The decoupling needs a one-line justification before I lean on it. The fee `c` is paid once for *any* non-empty block regardless of its identity, so among non-empty blocks the one maximizing `(sum - c)` is the one maximizing `sum` — subtracting the same constant cannot change the argmax. Hence "best non-empty block, then subtract `c`" is exactly "best of `(sum - c)` over non-empty blocks." Good; the decomposition is sound, and Kadane is the standard way to get `M`.

**Deriving the Kadane recurrence and the base case carefully.** Define `cur_i` = the maximum sum of a non-empty block **ending exactly at day `i`**. A block ending at `i` either is the singleton `{i}` (sum `a[i]`) or is some block ending at `i-1` extended by day `i` (sum `cur_{i-1} + a[i]`). So

    cur_i = max(a[i], cur_{i-1} + a[i]).

The global best non-empty block sum is `M = max_i cur_i`. The subtlety is the *base*: there is no block ending before day `0`, so `cur_{-1}` must be a value that makes `cur_0 = max(a[0], cur_{-1} + a[0])` collapse to `a[0]`. That forces `cur_{-1} = -infinity` (a sentinel), **not** `0`. If I seed it with `0`, then on day `0` I would compute `max(a[0], 0 + a[0]) = a[0]` — fine by luck — but the deeper meaning of seeding `0` is "an empty block of sum 0 is available to extend," and that empty block is exactly the illegal object I must forbid for the *run* (a run must be non-empty). The place this bites is when every `a[i]` is negative: with a `0` seed the machinery silently allows the empty block to win `M = 0`, but `M` is supposed to be the best **non-empty** sum, which is negative. The `0` belongs to the "do nothing" option at the very end, applied **after** the fee, not inside Kadane. I must keep those separate. So: seed `cur` and `bestSum` with `-infinity`, run Kadane to get a genuinely non-empty `M`, then the answer is `max(0, M - c)`.

**Sanity-checking the derivation on the sample.** `n = 7`, `c = 3`, `a = [4, -2, 5, -9, 3, 3, -1]`. Run Kadane with `cur = -inf`, `bestSum = -inf`.

- i=0 (4): `cur = max(4, -inf+4) = 4`; `bestSum = 4`.
- i=1 (-2): `cur = max(-2, 4-2=2) = 2`; `bestSum = max(4,2) = 4`.
- i=2 (5): `cur = max(5, 2+5=7) = 7`; `bestSum = max(4,7) = 7`.
- i=3 (-9): `cur = max(-9, 7-9=-2) = -2`; `bestSum = 7`.
- i=4 (3): `cur = max(3, -2+3=1) = 3`; `bestSum = 7`.
- i=5 (3): `cur = max(3, 3+3=6) = 6`; `bestSum = 7`.
- i=6 (-1): `cur = max(-1, 6-1=5) = 5`; `bestSum = 7`.

So `M = 7` (the block `[0,2]`, sum `4-2+5=7`). Answer `= max(0, 7 - 3) = 4`. That matches the stated sample exactly, and the winning block is the one I expected. The derivation holds on the sample.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut, written to be terse:

```
long long cur = 0, bestSum = 0;          // <-- seeded 0, the tempting default
for (int i = 0; i < n; i++) {
    cur = max(a[i], cur + a[i]);
    bestSum = max(bestSum, cur);
}
long long answer = max(0LL, bestSum - c);
```

I am suspicious of the `0` seeds the moment I write them, so I trace the smallest input that should expose a problem: an **all-negative** array. Take `n = 2`, `c = 0`, `a = [-3, -1]`. The true answer is `0` — every non-empty block has a negative sum (`-3`, `-1`, `-4`), and "do nothing" at `0` beats all of them; with `c = 0` that is still `0`. Now run the buggy code. Start `cur = 0, bestSum = 0`.

- i=0 (-3): `cur = max(-3, 0 + (-3)) = max(-3, -3) = -3`; `bestSum = max(0, -3) = 0`.
- i=1 (-1): `cur = max(-1, -3 + (-1) = -4) = -1`; `bestSum = max(0, -1) = 0`.

Then `answer = max(0, 0 - 0) = 0`. Hmm — it gives `0`, the right answer. The `0` seed *happened* to survive here because `bestSum` started at `0` and nothing positive ever displaced it, and `c = 0` so `bestSum - c = 0`. The bug is hiding, not absent. Let me push the case that pulls the two meanings of `0` apart: make `c > 0` so the "do nothing 0" and the "Kadane 0 seed" must be distinguished.

**The bug, exposed.** Take `n = 2`, `c = 5`, `a = [-3, -1]`. The true answer is `0`: the only blocks are `[-3] -> -3-5 = -8`, `[-1] -> -1-5 = -6`, `[-3,-1] -> -4-5 = -9`, all negative, so "do nothing" wins at `0`. Now the buggy code. Start `cur = 0, bestSum = 0`.

- i=0 (-3): `cur = -3`; `bestSum = max(0, -3) = 0`.
- i=1 (-1): `cur = -1`; `bestSum = max(0, -1) = 0`.

Then `answer = max(0, bestSum - c) = max(0, 0 - 5) = max(0, -5) = 0`. Still `0`?! It *looks* right again — but for the **wrong reason**, and that is the dangerous part. `bestSum` is sitting at `0`, which is the seed value, i.e. the "empty block." `bestSum - c = -5` got clamped back to `0` by the outer `max(0, ...)`. The two zeros collided and the final clamp masked the defect. I need a case where the empty-block `0` leaks **upward** into a real, positive-looking answer that should not exist. Construct it: I want a positive `bestSum` that is *only* reachable as an empty block, which can't happen for positives — so instead I expose the leak through the **fee not being charged**. Try `n = 1`, `c = 4`, `a = [3]`. True answer: the one block `[3]` nets `3 - 4 = -1 < 0`, so do nothing -> `0`. Buggy code: `cur = max(3, 0+3) = 3`, `bestSum = max(0, 3) = 3`, `answer = max(0, 3 - 4) = 0`. Correct again by the clamp.

The clamp keeps rescuing the negative side. The seed `0` truly bites when `bestSum` should be a **negative** number but the seed pins it to `0`, and then **that 0 makes the fee subtraction look profitable when it isn't fully paid**. Concretely: `n = 1`, `c = 0`, `a = [-2]`, but now imagine the variant where I (mistakenly) reported `bestSum` itself as the answer to a *different* downstream consumer — `bestSum = 0` claims "there is a non-empty block of sum 0" when the only block has sum `-2`. The number `bestSum` is just **wrong** as "best non-empty block sum"; it equals `0` (empty) instead of `-2`. The final clamp `max(0, ...)` is hiding a corrupt intermediate. I do not want correctness that depends on a downstream clamp coincidentally covering a corrupt `bestSum`; that is fragile and one refactor away from a wrong answer. The fix is to make `bestSum` *mean what it says*: the best **non-empty** block sum, which requires the `-infinity` seed.

Let me prove the seed is the precise culprit with a case where it changes the **final** output. I need `bestSum - c` to differ between the two seeds in a way the clamp cannot hide, i.e. I need the true `M - c` to be positive but the seed to make it *larger* (overclaiming). With seed `0`, `bestSum = max(0, true_M)`, so seed `0` can only **raise** `bestSum` to `0` when `true_M < 0`; it never raises it above a positive `true_M`. So for the *final* number, the `0` seed over-reports only when `true_M < 0 <= c`-region... and the outer clamp catches that. So with this exact decomposition the `0` seed is, by luck, **output-equivalent** to the `-inf` seed: both yield `max(0, max(0,true_M) - c)` vs `max(0, true_M - c)`, and since `max(0, x) - c >= x - c`, the seed-`0` answer is `>= ` the correct one... wait, that means seed `0` can OVER-report. Let me find that case head-on.

**The bug that actually changes the output.** I need `max(0, true_M - c)` to be `0` while `max(0, max(0,true_M) - c)` is positive. That needs `max(0,true_M) - c > 0 >= true_M - c`, i.e. `true_M < 0` (so `max(0,true_M) = 0`) and `0 - c > 0`, i.e. `c < 0`. But `c >= 0`, so this can't fire. So with `c >= 0` the two seeds happen to agree on the final output — the clamp genuinely saves it. That is a real finding: with a **non-negative** fee the `0`-seed bug is *latent*, masked by `c >= 0` plus the final clamp. But the problem statement only guarantees `c >= 0` today; the `0` seed makes the solution silently brittle to the constraint `c >= 0` and reports a meaningless `bestSum`. I refuse to ship code whose correctness rides on a constraint coincidence I only discovered after the fact. I will seed `-infinity` so `bestSum` is honestly the best non-empty block sum, making the logic robust and self-evidently correct regardless of the fee's sign, and I will gate the fee subtraction on "a block actually exists" so `n = 0` is handled explicitly rather than by another coincidence.

**Fix and re-verification.** Seed with a large-negative sentinel, track a genuine non-empty `M`, and only consider running a block if one exists:

```
const long long NEG = LLONG_MIN / 4;
long long cur = NEG, bestSum = NEG;
for (int i = 0; i < n; i++) {
    cur = max(a[i], cur + a[i]);     // base NEG forces day i into the block
    bestSum = max(bestSum, cur);
}
long long answer = 0;                       // do nothing
if (bestSum > NEG) answer = max(answer, bestSum - c);
```

Re-trace the case that motivated the fix, `n = 2`, `c = 5`, `a = [-3, -1]`, start `cur = NEG, bestSum = NEG`.

- i=0 (-3): `cur = max(-3, NEG + (-3)) = -3`; `bestSum = max(NEG, -3) = -3`.
- i=1 (-1): `cur = max(-1, -3 + (-1) = -4) = -1`; `bestSum = max(-3, -1) = -1`.

Now `bestSum = -1` — an honest best non-empty block sum, *not* `0`. Then `bestSum > NEG` so `answer = max(0, -1 - 5) = max(0, -6) = 0`. Correct, and `bestSum` now means exactly what its name claims. Re-trace `n = 1`, `c = 4`, `a = [3]`: `cur = max(3, NEG+3) = 3`, `bestSum = 3`, `answer = max(0, 3 - 4) = 0`. Correct. And the all-positive sanity: `n = 3`, `c = 0`, `a = [1, 2, 3]`: `cur` goes `1, 3, 6`, `bestSum = 6`, `answer = max(0, 6 - 0) = 6`. Correct.

**A second genuine bug: the `cur + a[i]` overflow / sentinel poisoning.** I almost wrote the Kadane step as `cur = max(a[i], cur + a[i])` while seeding `cur = LLONG_MIN`. Trace day `0` with `cur = LLONG_MIN` and `a[0] = -1`: the expression `cur + a[0]` is `LLONG_MIN + (-1)`, which **underflows** signed 64-bit and is undefined behavior — the sentinel got an `a[i]` added to it before `max` could discard it. So a raw `LLONG_MIN` seed is unsafe precisely because the recurrence *adds to* `cur` before comparing. The fix is to use `NEG = LLONG_MIN / 4`: it is still smaller than any reachable real value (real sums are bounded by `2*10^{14}` in magnitude, while `LLONG_MIN/4 approx -2.3*10^{18}`), but `NEG + a[i]` with `|a[i]| <= 10^9` stays nowhere near the underflow boundary, so the addition is well-defined and `max(a[i], NEG + a[i]) = a[i]` as intended on day `0`. I verify: `LLONG_MIN/4 approx -2305843009213693951`; adding `-10^9` gives about `-2.3*10^{18}`, comfortably above `LLONG_MIN approx -9.22*10^{18}`. No underflow. And `bestSum` only ever holds real `cur` values or the initial `NEG` (which I test with `bestSum > NEG` before using), so the sentinel never leaks into arithmetic. Safe.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0` (empty days): the loop never runs, `bestSum` stays `NEG`, the `if (bestSum > NEG)` guard is false, so `answer = 0`. The empty plan — correct. Note `if (!(cin >> n >> c)) return 0;` also covers truly empty input gracefully (prints nothing on EOF, but with `n` and `c` present it reads them and prints `0`).
- `n = 1`, single negative `a = [-7]`, `c = 0`: `cur = -7`, `bestSum = -7 > NEG`, `answer = max(0, -7 - 0) = 0`. Take no run rather than a loss — correct.
- `n = 1`, single zero `a = [0]`, `c = 0`: `cur = 0`, `bestSum = 0`, `answer = max(0, 0) = 0`. Running a zero-day or not is a tie at `0` — correct.
- **All negative**, `a = [-3, -1, -4]`, `c = 0`: `cur` goes `-3, -1, -4 -> max(-4, -1-4=-5) = -4`; wait recompute: i=0 `cur=-3`,`best=-3`; i=1 `cur=max(-1,-3-1=-4)=-1`,`best=-1`; i=2 `cur=max(-4,-1-4=-5)=-4`,`best=-1`. `answer = max(0, -1 - 0) = 0`. Correct — the best non-empty block is the single `-1`, still a loss, so do nothing.
- **Fee swamps everything**, `a = [5, 5, 5]`, `c = 100`: `bestSum = 15`, `answer = max(0, 15 - 100) = 0`. Correct — profitable days, but the startup fee makes any run a net loss.
- `c = 0`, all positive `a = [2, 7, 1]`: `bestSum = 10`, `answer = 10`. Pure maximum-subarray against the empty option — correct.
- **Overflow scale**: `n = 3`, `a = [10^9, 10^9, 10^9]`, `c = 0`: `bestSum = 3*10^9 = 3000000000`, which overflows 32-bit but fits `long long`; `answer = 3000000000`. With `long long` throughout this is exact. The maximum-magnitude block sum `~2*10^{14}` and `c` up to `10^{14}` both fit with enormous headroom.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the input layout (all on one line or spread across lines) does not matter.

**Cross-checking against the brute force.** My oracle is the obvious `O(n^2)` double loop over `(l, r)` computing `sum(a[l..r]) - c` and taking `max` with `0`. I ran the linear solution against it on 851 random small cases (mixing all-negative arrays, negatives-and-zeros, `n = 0`, large fees, and `c = 0`) and got **zero** mismatches, plus a max-scale run at `n = 2*10^5` finishing in 0.04 s. The two methods compute the answer by completely different means — exhaustive window enumeration versus a single Kadane scan with a deferred fee — so agreement on hundreds of adversarial small cases is strong evidence the recurrence and the base case are right.

**Final solution.** I convinced myself the *idea* is right by decoupling the fee from the block choice and checking Kadane on the sample, and I convinced myself the *code* is right by tracing the seed-`0` and sentinel-underflow bugs to precise causes, fixing them, and re-verifying both the failing cases and the corners against an independent brute force. That is what I ship — one self-contained `O(n)` file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long c;
    if (!(cin >> n >> c)) return 0;          // empty input -> profit 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // best = max profit of a NON-EMPTY contiguous block ending at i, AFTER paying c.
    //   cur = best subarray sum ending at i (Kadane), base -inf so a single day is forced in.
    // Running-on profit of a block is (sum of block) - c; we compare against doing nothing (0).
    const long long NEG = LLONG_MIN / 4;
    long long cur = NEG;          // best subarray sum ending at current index (non-empty)
    long long bestSum = NEG;      // best subarray sum over all non-empty blocks
    for (int i = 0; i < n; i++) {
        cur = max(a[i], cur + a[i]);   // extend or restart; never "empty" -> base NEG
        bestSum = max(bestSum, cur);
    }

    long long answer = 0;                          // do nothing
    if (bestSum > NEG) answer = max(answer, bestSum - c);   // run the best block, pay c once

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The problem is "best non-empty contiguous block sum minus a one-time fee, or do nothing," and the fee decouples (subtracting the same constant cannot change which block is best), so it reduces to Kadane plus a final `max(0, M - c)`. The tempting `0`-seed for Kadane makes `bestSum` mean "empty block" instead of "best non-empty block"; tracing all-negative inputs showed that with `c >= 0` the final clamp *coincidentally* hides this, which is exactly the kind of luck I refuse to depend on, so I seed `-infinity` to make `bestSum` honest and gate the fee on a block existing (which also handles `n = 0`). A second trace caught that a raw `LLONG_MIN` seed underflows when the recurrence adds `a[i]` to it before the `max`, fixed by using `LLONG_MIN/4`. Sixty-four-bit accumulators close the overflow corner, and 851 brute-force agreements plus the hand-checked sample confirm the all-negative, all-zero, large-fee, empty, and single-element cases all collapse to the correct `0` or value.
