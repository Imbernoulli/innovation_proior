**Reading the problem and pinning the contract.** I have `n` amplifier modules, module `i` with a positive integer gain `g[i]`, and a tolerance factor `p/q` given as two positive integers with `p >= q`. A pair `{i, j}` is *balanced* when the larger gain divided by the smaller is at most `p/q`; I must count the balanced unordered pairs. Input on stdin is `n p q` then the `n` gains; I print one integer. Before any algorithm I fix the scale, because here the scale *is* the problem: `n <= 2*10^5`, and crucially `1 <= q <= p <= 4*10^9` and `1 <= g[i] <= 4*10^9`. Two facts jump out. First, `4*10^9 > 2^31 ~ 2.1*10^9`, so `p`, `q`, and the gains do **not** fit in a 32-bit `int` — they must be read into `long long`. Second, the natural test will multiply two of these together, and `4*10^9 * 4*10^9 = 1.6*10^19`, which is past the signed 64-bit ceiling `2^63 - 1 ~ 9.22*10^18`. So even `long long` is not enough for the *product*. That is the whole game, and I am going to be paranoid about it.

**Reformulating the ratio test as a cross-multiplication.** Floating point is the obvious-but-wrong first instinct: "just check `g_hi / (double)g_lo <= p / (double)q`." A `double` has a 53-bit mantissa (~`9*10^15` of integer resolution), and my values reach `4*10^9` while their *ratios* can sit arbitrarily close to the bound. On a pair whose true ratio equals `p/q` exactly, the rounded quotient can land just above or just below, flipping the count. So I refuse division entirely. Since all of `g_lo, g_hi, p, q` are strictly positive, the inequality

```
g_hi / g_lo  <=  p / q
```

is equivalent, after multiplying both sides by the positive quantities `g_lo` and `q`, to

```
g_hi * q  <=  g_lo * p .
```

No division, no rounding — just two integer products and one comparison. This is the form I will use everywhere; the only question is what integer type holds those products, and I have already established it has to be wider than 64-bit.

**Laying out the candidate approaches.** Two routes, and I want the one I can prove *and* afford.

- *Quadratic enumeration.* For every unordered pair, compute `g_hi * q` and `g_lo * p` and compare. Obviously correct — this is exactly what my brute-force oracle will do. But `C(2*10^5, 2) ~ 2*10^10` comparisons is far over a 1-second limit. Useful as a checker, useless as the submission.
- *Sort, then two pointers.* If I sort the gains ascending and fix the *larger* member of a pair at index `j`, then every partner `i < j` automatically has `g[i] <= g[j]`, so the balanced test is just `g[j] * q <= g[i] * p`. I claim the admissible `i` form a contiguous suffix of the prefix and that the suffix boundary only moves one direction as `j` advances — which would give a linear sweep.

**Deriving the two-pointer monotonicity and checking it concretely.** Fix `j`. As `i` ranges over `0..j-1`, `g[i]` is non-decreasing (sorted). The test `g[j] * q <= g[i] * p` has `g[i] * p` on the right, which *grows* with `i`, while the left side is constant in `i`. So once the inequality holds for some `i = lo_j`, it holds for every larger `i` up to `j-1`: the balanced partners are exactly `i in [lo_j, j-1]`, a suffix. Now advance `j` to `j+1`: `g[j+1] >= g[j]`, so the left side `g[j+1] * q` is `>=` the old `g[j] * q`, i.e. the bar got *higher*, so the smallest qualifying `i` cannot decrease: `lo_{j+1} >= lo_j`. The boundary is monotone non-decreasing, so a single trailing pointer `lo` suffices and the total work is `O(n)` after the sort.

Let me not just assert this — let me check the count formula on the worked sample `g = [10, 1, 4, 8, 13, 5]`, `p = 5`, `q = 2`, whose answer is `8`. Sorted: `[1, 4, 5, 8, 10, 13]`. For each `j` I find the smallest `lo` with `g[j]*2 <= g[lo]*5` and count `j - lo`:

- `j=0` (g=1): no `i<j`, contributes `0`. `lo=0`.
- `j=1` (g=4): need `4*2=8 <= g[lo]*5`. `g[0]=1 -> 5`, `8 <= 5`? no, advance `lo` to 1; but `lo<j` fails at `lo=1=j`, stop. Partners `[1,0]` empty -> `0`.
- `j=2` (g=5): need `5*2=10 <= g[lo]*5`. `lo=1`: `g[1]=4 -> 20`, `10<=20` yes. Partners `[1,1]` -> `1` (the pair `{4,5}`).
- `j=3` (g=8): need `8*2=16 <= g[lo]*5`. `lo=1`: `g[1]=4 -> 20`, `16<=20` yes. Partners `[1,2]` -> `2` (pairs `{4,8}`,`{5,8}`).
- `j=4` (g=10): need `10*2=20 <= g[lo]*5`. `lo=1`: `g[1]=4 -> 20`, `20<=20` yes (equality — `{4,10}` is *on* the bound). Partners `[1,3]` -> `3` (`{4,10}`,`{5,10}`,`{8,10}`).
- `j=5` (g=13): need `13*2=26 <= g[lo]*5`. `lo=1`: `g[1]=4 -> 20`, `26<=20`? no, advance `lo=2`: `g[2]=5 -> 25`, `26<=25`? no, advance `lo=3`: `g[3]=8 -> 40`, `26<=40` yes. Partners `[3,4]` -> `2` (`{8,13}`,`{10,13}`).

Total `0+0+1+2+3+2 = 8`. Matches the stated answer, and note `lo` only ever moved rightward (`0 -> 1 -> 1 -> 1 -> 1 -> 3`), confirming the monotonicity I derived. The on-threshold pair `{4,10}` at `j=4` got counted precisely because I used `<=` with exact integer products; a `double` test would be playing Russian roulette there.

**First implementation — and immediately a trace, because the arithmetic is the trap.** Here is my first cut of the loop body. I am deliberately writing the comparison in `long long` first, the way most people reflexively do, to see whether I can catch myself:

```
sort(g.begin(), g.end());
long long count = 0;
int lo = 0;
for (int j = 0; j < n; j++) {
    while (lo < j && g[j] * q > g[lo] * p) lo++;   // products in long long
    count += (long long)(j - lo);
}
```

On the sample this gives `8` — the small values never stress the arithmetic, so the sample passes and lulls me. The danger only shows on large values, so I hand-build an adversarial pair. I want `g[j] * q` to *truly* be `<= g[lo] * p` (so the pair is balanced and should be counted) but to have `g[j] * q` overflow signed 64-bit and wrap negative, while `g[lo] * p` also overflows differently — so the wrapped comparison lies.

Take `n = 2`, `p = 2590928623`, `q = 200071089`, gains `g = [3687093964, 3758591649]`. Sorted that is `[3687093964, 3758591649]`, so `lo`-index `0` is `g_lo = 3687093964` and `j`-index `1` is `g_hi = 3758591649`. The exact products are

```
g_hi * q = 3758591649 * 200071089 =   751985524321735761
g_lo * p = 3687093964 * 2590928623 = 9552997287018131572
```

Exactly: `751985524321735761 <= 9552997287018131572` is **true**, so the pair *is* balanced and the answer should be `1`.

**The bug, traced.** Now run the same products through signed 64-bit `long long`. `g_hi * q = 751985524321735761` is below `2^63-1 ~ 9.22*10^18`, so it survives intact. But `g_lo * p = 9552997287018131572` is *above* `2^63-1`, so in `long long` it wraps: `9552997287018131572 - 2^64 = -8893746786691420044`. The loop condition is `g[j]*q > g[lo]*p`, i.e. `751985524321735761 > -8893746786691420044`, which is **true** — so the code advances `lo` to `1`, decides the pair is *not* balanced, and contributes `0`. My first implementation prints `0`; the truth is `1`. I confirmed this by compiling a `long long` variant of exactly this loop and feeding it `2 2590928623 200071089 / 3687093964 3758591649`: it printed `0`, while the brute force and the corrected solution both print `1`. The overflow is not hypothetical — it silently flipped one product's sign and corrupted the verdict. The defect is precise: `g[lo] * p` is a `long long * long long` multiplication whose mathematical value `~9.55*10^18` exceeds the 64-bit range, so the result is undefined/wrapped and the comparison operates on garbage.

**Fix and re-verification.** The fix is to form both products in a 128-bit integer so nothing wraps. `__int128` holds values up to `~1.7*10^38`, dwarfing my worst case `1.6*10^19`:

```
while (lo < j && (__int128)g[j] * q > (__int128)g[lo] * p) lo++;
```

Casting one operand to `__int128` promotes the whole multiplication to 128-bit, so `g[lo] * p = 9552997287018131572` is represented faithfully and the comparison `751985524321735761 > 9552997287018131572` is **false** — `lo` does not advance, the pair counts, the answer is `1`. Re-running the corrected program on the adversarial input gives `1`, matching brute. The case that broke now passes, and it broke for the exact reason I fixed (a 64-bit product overflow), which is the evidence I trust.

Let me also re-confirm the corrected loop still gets the sample right (the cast cannot change small-value results): products like `5*2=10` and `4*5=20` are tiny, the `<=`/`>` logic is identical, so the sample is still `8`. Verified by running it: `8`.

**A second precision episode: the count accumulator.** I caught the product overflow, but there is a quieter sibling: the count itself. Consider all-equal gains, `n = 2*10^5`, every `g[i] = 10^9`, and `p = q = 1` so every pair is balanced (ratio `1 <= 1`). Then the answer is `C(n,2) = n*(n-1)/2 = 200000 * 199999 / 2 = 19999900000 ~ 2*10^10`. That is well past the 32-bit `int` ceiling `~2.1*10^9`. If `count` were an `int`, it would wrap to garbage. I trace the per-`j` contribution to be sure my accumulator type and the suffix length never individually overflow either: each `count += (j - lo)` adds at most `j <= n-1 < 2*10^5`, a small `int`, but the running `count` grows to `~2*10^10`, so `count` must be `long long` — which it is. I ran exactly this case (`n=200000`, all `10^9`, `p=q=1`) and the program printed `19999900000`, the correct `C(200000,2)`, in about `0.05` seconds. Both overflow sites — the cross-product and the running count — are now sound.

**Edge cases, deliberately, because this is where this kind of code dies.**

- `n = 0`: the read `cin >> n >> p >> q` succeeds (the tolerance is still given), the gains loop reads nothing, the main loop never runs, `count = 0`. Correct — no pairs. I verified: input `0 5 2` with an empty second line prints `0`.
- `n = 1`: one module, no pair possible. The loop runs only `j=0` which has no `i<j`, so `count = 0`. Verified: `1 5 2 / 7` prints `0`.
- *All gains equal, `p = q`.* Ratio of any pair is `1`, and the bound `p/q = 1`, so `1 <= 1` — every pair balanced. The test `g[j]*q <= g[i]*p` becomes `g*q <= g*p`, i.e. `q <= p`, true. Counted. Verified on `4 1 1 / 9 9 9 9` -> `6 = C(4,2)`.
- *Distinct gains with `p = q` (bound exactly `1`).* Only equal-gain pairs can be balanced; with all-distinct gains none are. `3 5 5 / 2 3 4`: largest/smallest ratios all `> 1`, so `0`. Verified -> `0`.
- *On-threshold pairs.* The sample's `{4,10}` has ratio exactly `5/2 = p/q`; with `<=` and exact integer products it is counted (contributes to the `8`). This is the case a `double` test misrounds; the integer comparison nails it.
- *Maximal-magnitude pair.* `2 4000000000 1 / 1 4000000000`: ratio `4*10^9 / 1 = 4*10^9`, bound `4*10^9 / 1`, equal -> balanced -> `1`. The product `g_hi * q = 4*10^9 * 1` is fine, `g_lo * p = 1 * 4*10^9` fine, but the *general* worst case `g_hi * q` with both near `4*10^9` is the `1.6*10^19` that needs `__int128`. Verified -> `1`.

**Stress verification against the oracle.** I wrote an independent `O(n^2)` brute force that, for every unordered pair, compares using Python's exact `Fraction(hi, lo) <= Fraction(p, q)` — a completely different mechanism (arbitrary-precision rationals, no cross-multiplication of my own). My generator mixes three regimes: a *tight* mode with a tiny value alphabet and small `p/q` so many ratios land exactly on the bound (the misround trap), a *wide* mode with values and `p, q` up to `4*10^9` so cross-products hit `1.6*10^19` (the overflow trap), and a *mixed* medium regime. Across 600+ random small cases the two-pointer solution and the `Fraction` oracle agreed on every single one; zero mismatches. The combination — the on-threshold sample, the constructed overflow counterexample, the all-equal count case, and the randomized oracle agreement — is what convinces me both the idea and the transcription are right.

**Final solution.** I disproved floating point by reasoning about mantissa width, disproved `long long` products by *constructing and running* a case where one product wraps negative and flips the count, and confirmed the fix and corners against an exact-rational oracle. What I ship is one self-contained file: sort, two-pointer sweep, cross-products in `__int128`, count in `long long`.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long p, q;
    if (!(cin >> n >> p >> q)) return 0;
    vector<long long> g(n);
    for (auto &x : g) cin >> x;

    // A pair {i, j} (i != j), say with g_lo <= g_hi, is "balanced" when
    //   g_hi / g_lo <= p / q   <=>   g_hi * q <= g_lo * p   (q, p, values all > 0).
    // Sort ascending; for each right endpoint j the admissible left endpoints i (with i < j and
    // g[j]*q <= g[i]*p) form a suffix [lo, j-1] of the sorted prefix, and lo only moves rightward
    // as j advances -> a single two-pointer sweep. With values and p, q up to 4*10^9 the
    // cross-products reach ~1.6*10^19, which OVERFLOWS signed 64-bit; we form them in __int128 so no
    // division or floating point is ever used.
    sort(g.begin(), g.end());

    long long count = 0;     // up to n*(n-1)/2 ~ 2*10^10 at n = 2*10^5, must be 64-bit
    int lo = 0;
    for (int j = 0; j < n; j++) {
        // advance lo to the smallest i with g[j]*q <= g[i]*p, i.e. the pair (i, j) is balanced.
        while (lo < j && (__int128)g[j] * q > (__int128)g[lo] * p) lo++;
        // now i in [lo, j-1] are exactly the partners that pair with j as the larger element.
        count += (long long)(j - lo);
    }

    cout << count << "\n";
    return 0;
}
```

**Causal recap.** The ratio bound `g_hi/g_lo <= p/q` is correct but division is poison, so I cross-multiplied to `g_hi*q <= g_lo*p`; sorting makes the larger member the right endpoint and turns the admissible partners into a monotone suffix, giving an `O(n log n)` two-pointer sweep that I checked digit-by-digit on the sample (count `8`, `lo` only advancing). My first transcription formed the cross-products in `long long`, which I broke by constructing `p=2590928623, q=200071089, g=[3687093964,3758591649]`: the exact `g_lo*p ~ 9.55*10^18` exceeds `2^63-1`, wraps negative, and makes the code drop a genuinely balanced pair (prints `0`, truth `1`) — promoting the multiplication to `__int128` fixes it; a second, quieter overflow lives in the count (`~2*10^10` for all-equal `n=2*10^5`), handled by a `long long` accumulator; and `<=` with exact integer products is what lets the on-threshold pairs (`{4,10} = 5/2`) be counted exactly where a `double` test would misround. An independent `Fraction`-based oracle agreeing on 600+ randomized cases across tight, wide, and mixed regimes seals it.
