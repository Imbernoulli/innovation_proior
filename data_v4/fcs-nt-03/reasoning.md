**Reading the problem and pinning the contract.** I am given a single integer `n` and must output `D(n) = d(1) + d(2) + ... + d(n)`, where `d(k)` is the number of positive divisors of `k`. The contract is plain: read one integer from stdin, print one integer to stdout, time limit one second. The only thing that makes this more than a textbook exercise is the scale: `n` can be as large as `10^12`. Before I think about any algorithm I want to fix the magnitudes, because they decide both the data types and which approaches are even admissible. The answer `D(n)` grows like `n * ln(n)` (the average value of `d(k)` is about `ln(k)`), so for `n = 10^12` I should expect `D(n)` on the order of `10^12 * 27.6 ≈ 2.8 * 10^13`. That is far beyond the 32-bit range of about `2.1 * 10^9`, so every accumulator must be 64-bit. I will use `long long` throughout. That is decision number one and it is non-negotiable; an `int` here is a silent wrong-answer on the large test.

**The obvious approach, stated honestly.** The most direct reading of `D(n) = sum_{k=1}^{n} d(k)` says: for each `k` from `1` to `n`, compute the number of divisors of `k`, and add them up. Computing `d(k)` by trial division up to `sqrt(k)` costs `O(sqrt(k))`, so the whole thing is roughly `O(n * sqrt(n))` — completely hopeless. The standard speed-up is a divisor sieve: allocate an array `cnt[1..n]`, and for every `j` from `1` to `n`, walk its multiples `j, 2j, 3j, ...` and increment `cnt` at each. The total work is `n/1 + n/2 + ... + n/n ≈ n ln n`, so the sieve is `O(n log n)` time and `O(n)` memory. That is genuinely correct and I trust it completely — so much so that I will use exactly this sieve as my independent brute-force oracle later. But as a *solution* it dies twice over at `n = 10^12`: an array of `10^12` longs is `8` terabytes of memory (instant out-of-memory), and `n log n ≈ 4 * 10^13` increment operations would take minutes to hours, not one second. So the dividend-side view, in any form, is out for the real constraints.

**A better representation, but still linear.** There is a second way to write the same sum that avoids ever materializing per-`k` divisor counts. Instead of asking "how many divisors does each `k` have," I count divisor-incidences by the *divisor*. A fixed integer `i` divides exactly `floor(n / i)` of the numbers in `1..n` (namely `i, 2i, 3i, ..., floor(n/i) * i`). Summing over all possible divisors `i`,

```
D(n) = sum_{i=1}^{n} floor(n / i).
```

Let me sanity-check this identity on `n = 6` before I lean on it. `floor(6/1)=6, floor(6/2)=3, floor(6/3)=2, floor(6/4)=1, floor(6/5)=1, floor(6/6)=1`, total `6+3+2+1+1+1 = 14`. And directly, `d(1..6) = 1,2,2,3,2,4` sums to `14`. They agree, so the identity holds and I have a clean `O(n)` time, `O(1)` memory formula — a single loop, no array. That is a real improvement over the sieve: no memory blowup at all. But `O(n)` at `n = 10^12` is still a trillion iterations of the loop body. Even at an optimistic `10^9` simple iterations per second that is `~1000` seconds. I cannot get under one second by shaving constants off a loop that fundamentally runs `10^12` times. The linear barrier has to be broken, not optimized.

**Where the obvious approach breaks, made concrete.** Let me look at the term sequence `floor(n / i)` for a concrete `n`, say `n = 30`, to see whether it has structure I can exploit:

```
i      : 1  2  3  4  5  6  7  8  9 10 11 ... 15 16 ... 30
floor  :30 15 10  7  6  5  4  3  3  3  2 ... 2  1 ...  1
```

The striking thing is how *few distinct values* appear and how they clump. For `i` from `8` to `10` the value is constant at `3`; for `i` from `11` to `15` it is constant at `2`; for `i` from `16` to `30` it is constant at `1`. The large values (`30, 15, 10, 7, 6, 5, 4`) each occur for a single small `i`, and the small values each occur over a long run of `i`. This is not an accident: `floor(n/i)` is a step function in `i`, and the number of *distinct* values it takes is only about `2 * sqrt(n)`. So the trillion-term sum at `n = 10^12` really only has about `2 * 10^6` distinct values, each repeated over a block of consecutive `i`. If I could sum each block in `O(1)` I would be done in `O(sqrt(n))` time — roughly `2 * 10^6` operations, trivially under a second. That is the door the obvious linear loop leaves open.

**Deriving the insight: the Dirichlet hyperbola method.** I could implement the "block decomposition" directly: find each maximal run of `i` over which `floor(n/i)` is constant and add `value * (block length)`. That works and is `O(sqrt(n))`. But there is an even cleaner and more symmetric way to get to `O(sqrt(n))` that the geometry of the problem hands me for free, and it is the canonical tool for exactly this sum: the **Dirichlet hyperbola method**. Here is the idea. `D(n)` counts ordered pairs `(a, b)` of positive integers with `a * b <= n` — because each `k <= n` contributes one such pair `(a, b)` for every way to write `k = a * b`, i.e. one pair per divisor `a` of `k`, so `d(k)` pairs in total, and summing over `k` counts all pairs under the hyperbola `a * b <= n`. So the question "what is `D(n)`?" is exactly "how many lattice points lie in the region `a >= 1, b >= 1, a*b <= n`?"

That region is symmetric about the line `a = b`: if `a * b <= n` then `b * a <= n`. Let `s = floor(sqrt(n))`. Split the lattice points into three groups by comparing `a` and `b`:

- points with `a < b`,
- points with `a > b`,
- points with `a = b`.

By symmetry the first two groups have equal size. The points with `a = b` are exactly those with `a * a <= n`, i.e. `a` from `1` to `s`, so there are `s` of them. Now count the points with `a <= b` (the diagonal plus the upper part) by iterating over the *smaller* coordinate `a`: for a fixed `a` with `1 <= a <= s`, the partner `b` ranges over `a <= b <= floor(n/a)`. But it is cleaner to count all points with `a <= s` regardless of how `b` compares: for each `a` from `1` to `s`, the number of `b >= 1` with `a * b <= n` is `floor(n / a)`. That counts every point whose smaller-or-equal coordinate... let me be precise to avoid double counting, because this is exactly the kind of step where an off-by-something creeps in.

Count `T = sum_{a=1}^{s} floor(n / a)`. This `T` counts every lattice point `(a, b)` with `1 <= a <= s` and `a * b <= n`. Geometrically that is the vertical strip `a <= s` under the hyperbola. By the symmetry `a <-> b`, the same number `T` also counts the horizontal strip `b <= s` under the hyperbola. Adding the two strips counts every point in the region, but it double-counts exactly the points lying in *both* strips, i.e. with `a <= s` and `b <= s` — and since `a, b <= s = floor(sqrt(n))` implies `a * b <= s*s <= n` automatically, that overlap is the full `s × s` square of `s*s` points. Therefore

```
D(n) = (strip a<=s) + (strip b<=s) - (overlap) = 2 * T - s*s,
       where T = sum_{a=1}^{s} floor(n / a),  s = floor(sqrt(n)).
```

That is the formula. It evaluates `D(n)` in `O(sqrt(n))` time and `O(1)` memory — exactly the `2 * 10^6`-iteration loop the structure promised, with no array and no per-block bookkeeping. Let me verify it on `n = 6`. `s = floor(sqrt(6)) = 2`. `T = floor(6/1) + floor(6/2) = 6 + 3 = 9`. `D(6) = 2 * 9 - 2*2 = 18 - 4 = 14`. Matches the hand computation. One more, `n = 10`: `s = 3`, `T = 10 + 5 + 3 = 18`, `D = 36 - 9 = 27`. Brute (sieve) on `1..10` gives `1+2+2+3+2+4+2+4+3+4 = 27`. Matches. I am convinced the math is right.

**First implementation — and the integer-sqrt trap I anticipate.** The loop is trivial; the dangerous line is computing `s = floor(sqrt(n))`. Floating-point `sqrt` on a 64-bit `n` near `10^12` is a classic source of silent off-by-one: `sqrtl((long double)n)` can round either up or down, and if it gives `s` one too large then `s*s > n` and the whole symmetry argument breaks (I would be claiming an `s × s` square sits under the hyperbola when its far corner pokes out). My first cut:

```
long long s = (long long)sqrtl((long double)n);
long long T = 0;
for (long long i = 1; i <= s; i++) T += n / i;
long long answer = 2 * T - s * s;
```

Before trusting this I want to stress the `s` computation specifically, because that is the one line that can be wrong without the small samples ever noticing. I trace the perfect-square boundary `n = 10^12`, where `sqrt(n)` is exactly `10^6`. If `sqrtl` returns `999999.9999...` truncated to `999999`, then I would compute `T` over `i = 1..999999` and `s*s = 999998000001`, missing the `i = 10^6` term and using the wrong correction — a wrong answer. Conversely if it returns `1000000.0000001` truncated to `1000000`, that happens to be correct here, but for some other `n` just below a square the same upward rounding would give `s` with `s*s > n`, which is corrupt.

**Diagnosing and hardening the sqrt.** The defect is real and not hypothetical: floating `sqrt` is only guaranteed correct to within rounding, so I must *not* trust its result as an exact integer floor. The fix is to clamp `s` to the true integer square root with two short adjustment loops that cost `O(1)` (they execute at most a couple of times):

```
long long s = (long long)sqrtl((long double)n);
while (s * s > n) s--;                 // pull s down if sqrt rounded up
while ((s + 1) * (s + 1) <= n) s++;    // push s up if sqrt rounded down
```

After these two loops, `s` provably satisfies `s*s <= n < (s+1)*(s+1)`, i.e. `s = floor(sqrt(n))` exactly, regardless of how the floating-point `sqrt` rounded. I must also be sure the guard expressions themselves do not overflow: the largest `s` I ever test is around `10^6`, so `(s+1)*(s+1)` is about `10^12`, comfortably inside `long long`. Good. (I briefly considered C++20's `std::sqrt` exactness guarantees, but the clamp loops make the code robust without depending on the standard library version, so I keep them.)

**Re-verifying the fix on the boundary cases.** Let me retrace exactly the inputs that the trap targets.
- `n = 10^12`: floating `s` lands at or near `10^6`. The `while (s*s > n)` loop pulls it down if it overshot; the `while ((s+1)^2 <= n)` loop pushes it up if it undershot. Either way `s = 10^6`. `T = sum_{i=1}^{10^6} floor(10^12 / i)`. `answer = 2*T - 10^12`. The compiled program returns `27785452449086`, and an independent Python hyperbola check with `math.isqrt` (an exact integer sqrt, a completely different code path) returns the same value. They agree.
- `n = 49` (perfect square): `s` clamps to `7`, `s*s = 49 <= 49`, `(s+1)^2 = 64 > 49`. `T = 49+24+16+12+9+8+7 = 125`, `answer = 250 - 49 = 201`. Brute sieve gives `201`. Match.
- `n = 48` (just below a square): `s = 6` (since `49 > 48`), `T = 48+24+16+12+9+8 = 117`, `answer = 234 - 36 = 198`. Brute gives `198`. Match.
- `n = 50` (just above a square): `s = 7`, `T = 50+25+16+12+10+8+7 = 128`, `answer = 256 - 49 = 207`. Brute gives `207`. Match.
The three values straddling a perfect square — the place the symmetry-correction term `s*s` is most fragile — all check out, and they check out for the reason I hardened: the clamp pins `s` to the exact floor so the `s × s` overlap is exactly the square that lies under the hyperbola.

**Edge cases, deliberately.**
- `n = 0`: an empty sum, so `D(0) = 0`. I special-case `n <= 0` and print `0` before touching `sqrt` (passing `0` into the loop would also yield `s = 0`, empty `T`, `answer = 0`, but the early return is cleaner and documents the convention). Correct.
- `n = 1`: `s = 1`, `T = floor(1/1) = 1`, `answer = 2 - 1 = 1`. And `d(1) = 1`. Correct.
- `n = 2`: `s = 1`, `T = 2`, `answer = 4 - 1 = 3`. `d(1)+d(2) = 1 + 2 = 3`. Correct.
- Empty input (no integer at all): `cin >> n` fails, I `return 0` printing nothing, which is the right behavior for a missing testcase token.
- Overflow: every accumulator is `long long`. `T` is bounded by `D(n) + s*s ≈ 2.8*10^13 + 10^12`, and `2*T` by `~5.7*10^13`, both far under `9.2*10^18`. The guard products `(s+1)^2 ≈ 10^12` are safe. No 64-bit overflow anywhere.

**Differential testing to close it out.** Hand traces are necessary but not sufficient, so I compiled the solution with `g++ -O2 -std=c++17` and ran it against the independent `O(n log n)` divisor-sieve brute force on over a thousand cases: every `n` in a set of explicit edges (`0,1,2,...,12`, perfect squares `16,25,36,49`, square-straddlers, powers-of-ten neighbours, primes like `99991`) plus a thousand random `n` drawn from a mix of tiny, small, and mid-sized bands up to `2*10^4` (the ceiling where the sieve stays fast). Zero mismatches over `1028` cases. The sieve never computes a single `floor(n/i)`, so a shared bug between the two methods is essentially impossible — the agreement is real evidence, not a tautology. And the `n = 10^12` run finishes in well under a millisecond using a few megabytes, so the time and memory limits are met with enormous margin.

**Final solution.** The dividend-side sum and even the `sum floor(n/i)` linear loop both scale with `n` and cannot survive `n = 10^12`; the resolution was to read `D(n)` as the count of lattice points under the hyperbola `a*b <= n` and exploit its symmetry across `a = b`, giving `D(n) = 2 * sum_{i=1}^{s} floor(n/i) - s*s` with `s = floor(sqrt(n))` in `O(sqrt(n))` time and `O(1)` memory. The one real hazard — the integer square root — I hardened with clamp loops that pin `s` to the exact floor, verified on the perfect-square boundaries where it matters most, and confirmed against an independent sieve over a thousand cases. This is what I ship, one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long n;
    if (!(cin >> n)) return 0;          // no input -> nothing to do
    if (n <= 0) { cout << 0 << "\n"; return 0; } // D(n)=0 for n<=0 by convention

    // Dirichlet hyperbola method for D(n) = sum_{i=1..n} d(i) = sum_{i=1..n} floor(n/i).
    // Pairs (a,b) with a*b <= n are counted by symmetry across the line a=b:
    //   D(n) = 2 * sum_{i=1..s} floor(n/i) - s*s,  where s = floor(sqrt(n)).
    long long s = (long long)sqrtl((long double)n);
    while (s * s > n) s--;               // guard sqrt rounding from above
    while ((s + 1) * (s + 1) <= n) s++;  // guard sqrt rounding from below

    long long sum = 0;
    for (long long i = 1; i <= s; i++) sum += n / i;

    long long answer = 2 * sum - s * s;
    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The dividend view (`sum d(k)`, via sieve) is `O(n log n)` time and `O(n)` memory and dies on both at `n = 10^12`; rewriting as `sum_{i=1}^{n} floor(n/i)` removes the memory problem but is still `O(n)` — a trillion iterations — so the linear barrier had to be broken. Inspecting the term sequence (`n = 30`) showed `floor(n/i)` takes only `~2*sqrt(n)` distinct values, and the geometric reading of `D(n)` as lattice points under `a*b <= n` turned that observation into the exact hyperbola identity `2 * sum_{i<=s} floor(n/i) - s*s`, an `O(sqrt(n))` loop. The single fragile line is `s = floor(sqrt(n))`: floating `sqrt` can round either way, so I clamped `s` with two `O(1)` loops and re-verified on the perfect-square straddlers (`48, 49, 50`) where the `s*s` correction is sharpest; a thousand-case differential test against an independent divisor sieve, plus an exact-`isqrt` Python cross-check at `n = 10^12`, confirmed zero mismatches.
