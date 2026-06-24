**Reading the problem and pinning the contract.** I have `n` attendees, each with a non-negative influence `a[i]`, and every unordered pair `i < j` contributes `a[i] * a[j]` to a running total. I must print that total, `S = sum_{i<j} a[i] * a[j]`. Input on stdin is `n` followed by the `n` values; I print one integer. Before I pick an algorithm I fix the scale, because here the scale is the whole danger: `n <= 10^5` and `0 <= a[i] <= 10^4`. The largest the answer can get is when every value is maxed: there are `C(n,2) = n(n-1)/2` pairs, each contributing at most `10^4 * 10^4 = 10^8`. So `S` can reach about `(10^5)^2 / 2 * 10^8 = 0.5*10^10 * 10^8 = 5*10^17`. That is comfortably past the 32-bit signed range (`~2.1*10^9`) — by roughly *eight orders of magnitude*. A 32-bit `int` anywhere on the accumulation path is a silent wrong answer on the big tests. I note this loudly now and will come back to police it. The answer fits a 64-bit `long long` (`~9.2*10^18`) with room to spare, so `long long` is the target type for every accumulator and for the result.

**Laying out the candidate approaches.** Two routes, and I want the one I can both prove and afford.

- *Direct double loop.* For every `i < j`, add `a[i] * a[j]`. Unarguably correct — it is the definition — but `O(n^2)`. At `n = 10^5` that is `~5*10^9` multiply-adds inside a tight loop; even at a generous `10^9` simple ops per second that is several seconds, well over the 1-second limit. So the double loop is only good as a *reference brute force* on small inputs, not as the shipped solution.
- *Closed-form identity.* I recall the square-of-sum expansion. Let `T = sum_i a[i]` and `Q = sum_i a[i]^2`. Then `T^2 = (sum_i a[i])(sum_j a[j]) = sum_{i,j} a[i] a[j]`, summing over *all* ordered pairs including `i = j`. Split that ordered double sum into the diagonal `i = j` and the off-diagonal `i != j`: `T^2 = Q + sum_{i != j} a[i] a[j]`. The off-diagonal counts each unordered pair `{i,j}` twice (once as `(i,j)`, once as `(j,i)`), so `sum_{i != j} a[i] a[j] = 2 * S`. Therefore `T^2 = Q + 2S`, i.e. `S = (T^2 - Q) / 2`. That is one `O(n)` pass accumulating `T` and `Q`, then a constant-time finish. This is the one to ship — provided I get the `- Q`, the `/2`, and the integer widths exactly right.

**Deriving the identity carefully and a sanity-check on the sample.** I do not want to trust a half-remembered formula, so I re-derive it on the documented sample `a = [3, 1, 4, 1, 5]`, whose stated answer is `72`. First `T = 3+1+4+1+5 = 14`, so `T^2 = 196`. Next `Q = 9 + 1 + 16 + 1 + 25 = 52`. Then `S = (196 - 52)/2 = 144/2 = 72`. That matches the sample exactly, which is strong evidence the identity is `S = (T^2 - Q)/2` and not, say, `(T^2 - Q)` or `(T^2 + Q)/2`. Good — the algebra and the worked example agree.

A second sanity-check on a trivial instance where I can see every term: `a = [2, 3]`. The only pair contributes `2*3 = 6`. Identity: `T = 5`, `T^2 = 25`, `Q = 4 + 9 = 13`, `S = (25 - 13)/2 = 12/2 = 6`. Correct. And the empty/singleton boundary: with `n = 1`, `a = [7]`, there are no pairs so `S = 0`; identity gives `T = 7`, `T^2 = 49`, `Q = 49`, `S = (49 - 49)/2 = 0`. Correct — the formula self-zeroes when there is nothing to pair. That reassures me the closed form handles the small-`n` corners without special-casing.

**First implementation — and immediately a stress trace, because the scale is the trap.** My first cut, transcribing the identity directly:

```
int n;
cin >> n;
int sum = 0, sumsq = 0;
for (int i = 0; i < n; i++) {
    int x;
    cin >> x;
    sum += x;
    sumsq += x * x;
}
long long answer = (sum * sum - sumsq) / 2;
cout << answer << "\n";
```

It looks like a faithful copy of the math, and `answer` is even a `long long`, which gives a false sense of safety. So I deliberately trace the worst case rather than a friendly one, because friendly inputs would hide the bug. Take the maximal instance: `n = 100000` with every `a[i] = 10000`. By hand, `T = 100000 * 10000 = 10^9`. The true `T^2 = 10^18`. The true `Q = 100000 * 10^8 = 10^13`. The true `S = (10^18 - 10^13)/2 = (1000000000000000000 - 10000000000000)/2 = 999999990000000000 / 2 = 499999995000000000`, about `5*10^17`.

Now trace what the *code* computes. `sum` is declared `int`. After the loop, the mathematical value of `sum` is `10^9`, but `int` maxes out at `2147483647 ≈ 2.1*10^9`; `10^9` actually still fits in `int` (barely — it is under `2.1*10^9`). So `sum` survives this particular case. But `sum * sum` is an `int * int` multiplication: both operands are `int`, so C++ computes the product *in `int`* and only then widens to `long long` for the assignment. `10^9 * 10^9 = 10^18` cannot fit in 32 bits; the multiplication overflows and wraps to some garbage 32-bit value before it is ever promoted. Likewise `sumsq` is an `int` accumulating `x*x = 10^8` a hundred thousand times toward `10^13`, which wrapped around long ago. Even though I stored the result in a `long long`, every quantity feeding it was computed in 32-bit arithmetic. The printed answer is nonsense, not `499999995000000000`.

**Diagnosing the first bug — int overflow on the accumulation path.** The defect is precise and it is exactly the pitfall I flagged at the start: declaring `sum`, `sumsq`, and `x` as `int` means `x*x`, the running `sumsq`, and above all `sum * sum` are all evaluated in 32-bit. The widening to `long long` happens *after* the damage, on assignment. Making only `answer` a `long long` is the classic trap — the wide type has to be on the *operands*, not just the destination. The fix is to make every value on the arithmetic path 64-bit: `x`, `sum`, and `sumsq` all `long long`. Then `sum * sum` is `long long * long long` (`10^18`, fits), `x * x` is `long long * long long` (`10^8`, fits), and the running `sumsq` (`10^13`) and final `S` (`5*10^17`) all fit. I confirm the maxima against the `long long` ceiling `~9.2*10^18`: intermediate `sum * sum = 10^18` < `9.2*10^18` (ok), final `S ≈ 5*10^17` < `9.2*10^18` (ok). Nothing on the path exceeds 64 bits.

To make the failure concrete rather than asserted, let me compute what the broken `int` code actually prints. The 32-bit product `sum * sum` with `sum = 10^9` is `10^18 mod 2^32` interpreted as a signed 32-bit value. `2^32 = 4294967296`, and `10^18 = 232830643 * 2^32 + r`; reducing, `10^18 mod 2^32 = 2808348672`, which as a signed `int` (above `2^31 = 2147483648`) reads as `2808348672 - 4294967296 = -1486618624`. Meanwhile the `int` `sumsq` accumulating toward `10^13` has wrapped many times and lands on its own garbage residue. So the buggy line subtracts two wrapped 32-bit values, halves the difference, and prints a small, often *negative* number — nothing like `499999995000000000`. The instructive part is that the bug is invisible on every small test (where `sum * sum` stays under `2^31`) and only detonates near the maximum, which is exactly why I forced the maximal trace instead of trusting the friendly samples. This is the signature of a silent overflow: correct on the cases you glance at, wrong on the cases the judge actually weights.

**Fixing the types and re-verifying the worst case.** Rewrite with 64-bit accumulators:

```
long long sum = 0, sumsq = 0;
for (int i = 0; i < n; i++) {
    long long x;
    cin >> x;
    sum += x;
    sumsq += x * x;
}
long long answer = (sum * sum - sumsq) / 2;
```

Re-trace `n = 100000`, all `10000`: `sum = 10^9` (fits ll), `sum * sum = 10^18` (fits ll), `sumsq = 10^13` (fits ll), `answer = (10^18 - 10^13)/2 = 499999995000000000`. I run this exact case through the compiled program and it prints `499999995000000000`, matching the hand computation. The overflow is gone because the wide type now sits on the operands. The bug broke precisely where I predicted — `int * int` — and the fix removes precisely that, which is the evidence I trust.

**Second implementation episode — a subtler arithmetic bug, traced on the sample.** While cleaning up I briefly wrote the finish as `answer = (sum * sum - sumsq)` and forgot the `/2`, reasoning loosely that "the square of the sum minus the squares is the cross terms." Let me trace that on the sample `a = [3, 1, 4, 1, 5]` before trusting it. `sum = 14`, `sum * sum = 196`, `sumsq = 52`, so this buggy line gives `answer = 196 - 52 = 144`. But the documented answer is `72`, and `144 = 2 * 72`. The trace exposes the defect immediately: `T^2 - Q` equals the sum over all *ordered* off-diagonal pairs, which counts `{i,j}` twice (as `(i,j)` and `(j,i)`); the unordered total `S` is exactly half of it. Dropping the `/2` double-counts every handshake. Restoring `(sum * sum - sumsq) / 2` brings the sample back to `72`. And the division is exact: `T^2 - Q = 2S` is always even by construction, so integer division by 2 loses nothing — I verify on the sample (`144/2 = 72`, no remainder) and note it holds in general because the right side is literally `2S`.

**Edge cases, deliberately, because this is where ad-hoc arithmetic dies.**
- `n = 0`: the loop never runs, `sum = sumsq = 0`, `answer = (0 - 0)/2 = 0`. No attendees, no handshakes — correct. The `if (!(cin >> n)) return 0;` also guards a completely empty stdin.
- `n = 1`, `a = [9999]`: loop adds once, `sum = 9999`, `sumsq = 9999^2 = 99980001`, `answer = (9999^2 - 9999^2)/2 = 0`. One attendee can shake no hands — correct.
- `n = 2`, `a = [2, 3]`: `sum = 5`, `sumsq = 13`, `answer = (25 - 13)/2 = 6`. Single handshake `2*3 = 6` — correct.
- All zeros, `a = [0, 0, 0]`: `sum = 0`, `sumsq = 0`, `answer = 0`. Every product is zero — correct.
- Equal values, `a = [5, 5, 5]`: `sum = 15`, `sumsq = 75`, `answer = (225 - 75)/2 = 75`. Check: three pairs each `5*5 = 25`, total `75` — correct.
- Overflow boundary: intermediate `sum * sum` peaks at `10^18` and the final `S` at `~5*10^17`, both safely under the `long long` ceiling `~9.2*10^18`; with all operands `long long` there is no 32-bit step anywhere. Safe.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace, so the values may be on one line or many.

**Mechanized re-verification.** Beyond the hand traces I run the `O(n)` solution against the obvious `O(n^2)` double-loop brute force on several hundred random small instances (`n` from 0 to 40, value ranges including `0`, small caps, and `10^4`), and they agree on every case. The brute force is the literal definition of `S` — a plain nested loop summing `a[i] * a[j]` for `i < j` in Python (arbitrary-precision, so the reference itself can never overflow) — so agreement over that many varied inputs is strong, type-independent evidence that my closed form is the same function. I deliberately seed the generator across several value caps: cap `0` exercises the all-zero path, cap `1` produces lots of pair collisions and tiny answers, and cap `10^4` pushes the per-pair products to their maximum so the small-`n` totals already climb into the millions and would expose a `/2` or sign error instantly. The combination I trust is three-legged: the algebraic re-derivation (so I know *why* the formula holds), the maximal-case overflow trace (so I know the *types* are wide enough where the brute force is too slow to reach), and the brute-force agreement on hundreds of small cases (so I know the *transcription* matches the definition). Any one of these alone could miss a bug the other two catch — the overflow trace says nothing about a wrong constant factor, and the brute force says nothing about overflow because its inputs are tiny.

**Final solution.** I convinced myself the idea is right by re-deriving `S = (T^2 - Q)/2` and checking it on `[3,1,4,1,5] -> 72` and on hand instances, and I convinced myself the *code* is right by tracing the maximal case to catch the silent `int * int` overflow, tracing the sample to catch the dropped `/2`, and cross-checking against a brute force. That is what I ship — one self-contained `O(n)` file with every accumulator 64-bit:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    long long sum = 0, sumsq = 0;
    for (int i = 0; i < n; i++) {
        long long x;
        cin >> x;
        sum += x;
        sumsq += x * x;            // x up to 1e4 => x*x up to 1e8, fits; long long anyway
    }
    // sum over i<j of a[i]*a[j] = (sum^2 - sum of squares) / 2
    long long answer = (sum * sum - sumsq) / 2;
    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The naive `O(n^2)` double loop is correct but too slow at `n = 10^5`, so I moved to the closed form `S = (T^2 - Q)/2`, which I re-derived from `(sum a)^2 = sum a^2 + 2 sum_{i<j} a_i a_j` and confirmed on the sample (`196 - 52 = 144`, halved to `72`). My first transcription declared the accumulators `int`, so even with a `long long` destination the operands `sum * sum` (`10^18`) and the running `sumsq` (`10^13`) were computed in 32-bit and silently wrapped — a maximal-case trace (`n = 100000`, all `10^4`, true answer `499999995000000000`) exposed it, and widening `x`, `sum`, and `sumsq` to `long long` so the wide type sits on the operands fixes it; a second trace of the sample caught a dropped `/2` that double-counted every unordered pair (`144` instead of `72`); and the small-`n`, all-zero, and equal-value corners plus a few hundred brute-force comparisons close out the verification.
