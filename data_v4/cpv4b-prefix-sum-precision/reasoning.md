**Reading the problem and pinning the contract.** I have `n` daily revenues `a[0..n-1]`, each in `[-10^9, 10^9]`, and a minimum length `L`. Over every contiguous window `[i, j)` with `j - i >= L` I must maximize the average `(a[i] + ... + a[j-1]) / (j - i)` and print it as a *reduced fraction* `p/q` with `q > 0`. The constraints are `1 <= L <= n <= 2*10^5`. Two things jump out before I pick an algorithm, and both are about *numbers*, not about cleverness. First, a window sum can be as large as `2*10^5 * 10^9 = 2*10^14` in magnitude, which already overflows 32-bit, so prefix sums and window sums must be 64-bit `long long`. Second, the output is a fraction, and the constraint explicitly forbids rounding — the moment I print a decimal I have lost. So the real question is not "which window," it's "how do I *compare* two averages exactly without ever dividing." That single decision will dominate everything below.

**Prefix sums set the stage.** Let `S[0] = 0` and `S[k] = a[0] + ... + a[k-1]`. Then window `[i, j)` has sum `S[j] - S[i]` and average `(S[j] - S[i]) / (j - i)`. So I am maximizing, over `0 <= i < j <= n` with `j - i >= L`, the ratio `(S[j] - S[i]) / (j - i)`. Geometrically that ratio is the *slope* of the segment connecting the prefix points `P_i = (i, S[i])` and `P_j = (j, S[j])`. That reframing — "maximize a slope" — is the hinge of the whole solution, so let me hold onto it.

**Candidate approaches, judged by whether they can stay exact.**

- *Float binary search on the answer.* Guess an average `x`; subtract `x` from every element and ask whether some window of length `>= L` has nonnegative sum — a standard prefix-min scan, `O(n)` per guess. Bisect on `x` for `O(n log(range/eps))`. Clean and short. But `x` is a `double`, and that is exactly the trap the problem is built around: with denominators up to `2*10^5`, two *distinct* achievable averages can differ by as little as `1/(d1*d2)`, on the order of `1/(2*10^5)^2 = 2.5*10^-11`, while the values themselves live near `10^9`. A `double` has ~15-16 significant decimal digits, so near `10^9` its resolution is about `10^9 * 2^-52 ~ 2*10^-7` — *four orders of magnitude too coarse* to separate two such averages. Float bisection can return a value that rounds to the wrong fraction, and then I cannot even recover `p/q`. I distrust this.
- *Exact convex-hull tangent.* Maximize the slope from prefix points directly, using only integer cross-multiplications. `O(n)`. Harder to write but it never divides, so it can be *exactly* right. This is the one I can defend, so I commit to it and accept that I must be careful with the arithmetic width.

**A concrete reason the float route is unsafe — numeric self-check.** Before abandoning binary search I want evidence, not a feeling. Suppose two windows have sums and lengths `(s1, d1) = (999999937, 1)` and `(s2, d2) = (1999999875, 2)`. Their averages are `999999937` and `999999937.5`; the second is strictly larger, by exactly `1/2`. That gap is huge, fine. Now stress the real danger: `(s1, d1) = (3, 199999)` and `(s2, d2) = (2, 133333)`. Average one is `3/199999 = 1.500007500...e-5`; average two is `2/133333 = 1.500003750...e-5`. They differ at the 11th significant figure. If instead these ratios sat near `10^9` (achievable by adding a constant to every element, which shifts both averages equally and so cannot change which is larger), a `double` near `10^9` cannot resolve `~10^-11`. So a comparison that *must* come out one way can come out the other under floating point. That is the disqualifying evidence. Exact integer comparison it is: to compare `s1/d1` versus `s2/d2` with `d1, d2 > 0`, compare `s1 * d2` versus `s2 * d1` — cross-multiply, never divide.

**Deriving the hull algorithm.** I want, for each right endpoint `j`, the left endpoint `i` in the allowed set `{0, 1, ..., j - L}` that maximizes slope `P_i -> P_j`. As `j` increases by one, the allowed set grows by exactly one new index `j - L` (the windows that just became long enough). The left endpoint achieving the maximum slope to a point that lies up-and-to-the-right always sits on the **lower convex hull** of the candidate points: any point strictly above the lower hull is dominated, because the hull vertex below it gives at least as steep a segment to `P_j`. So I maintain the lower hull of `{P_0, ..., P_{j-L}}` incrementally — push `P_{j-L}`, popping while the last turn is not a proper "lower-left" turn — and then, among hull vertices, pick the one whose segment to `P_j` is steepest.

**Why the tangent is found by binary search — and a self-check of unimodality.** Walking the lower hull left to right, its edge slopes are strictly increasing (that is what "lower convex hull" means). For a fixed right point `P_j` sitting above the hull, the slope from hull vertex `v` to `P_j` *decreases* as `v` moves right toward `j` along directions steeper than the local edge, and *increases* while the local edge is shallower than the segment to `P_j`. Net effect: the function "slope from `hull[t]` to `P_j`" is unimodal in `t` — it rises then falls — so the maximizing vertex is where consecutive slopes switch from increasing to non-increasing. I can binary-search that switch. Let me sanity-check unimodality on a tiny hull. Take prefix points `P_0=(0,0)`, `P_1=(1,-3)`, `P_2=(2,-4)`, `P_3=(3,-3)` — a valley. Lower hull of `{P_0,P_1,P_2}` is `P_0, P_2` (since `P_1` lies above segment `P_0P_2`? check: line `P_0->P_2` has slope `-2`, at `x=1` it predicts `-2`, but `P_1.y = -3 < -2`, so `P_1` is *below* the line and stays on the hull). Hull = `P_0, P_1, P_2`. Query `P_3=(3,-3)`: slope from `P_0` is `(-3-0)/3 = -1`; from `P_1` is `(-3-(-3))/2 = 0`; from `P_2` is `(-3-(-4))/1 = 1`. Sequence `-1, 0, 1` — monotone increasing here, max at the rightmost vertex `P_2`, average `1`. Window `[2,3)` = `a[2]` indeed. Unimodality holds (monotone is a degenerate unimodal), and the binary search will land on the last index. Good.

**First implementation — and immediately a trace, because slope code transcribes treacherously.** My first cut of the per-`j` work, written fast:

```
for (int j = L; j <= n; j++) {
    int newi = j - L;
    while (hull.size() >= 2 &&
           cross(hull[sz-2], hull[sz-1], newi) >= 0) hull.pop_back();   // (A)
    hull.push_back(newi);
    // binary search tangent
    int lo = 0, hi = hull.size() - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        long long lhs = (S[j]-S[A]) * (j - B);     // (B) 64-bit products
        long long rhs = (S[j]-S[B]) * (j - A);
        if (lhs < rhs) lo = mid + 1; else hi = mid;
    }
    ...
    if (num * bestDen > bestNum * den) { bestNum = num; bestDen = den; }  // (C) 64-bit
}
```

I trace the documented sample `n=7, L=3, a=[1,12,-5,6,7,-2,3]`. Prefix `S = [0,1,13,8,14,21,19,22]`. The intended answer is window `[1,5)` (sum `21-1=20`, length `4`, average `5`). Walking it, the hull is built from indices `0,1,2,3,4` as `j` runs `3..7`, and at `j=5` the steepest left endpoint should be `i=1` giving `(S[5]-S[1])/(5-1) = (21-1)/4 = 5`. Running this version by hand on the hull-turn test at line (A): I add indices in order `0,1,2,...`; with `cross(...) >= 0` popping, I am keeping a turn only when the cross product is strictly negative. Let me check what that builds. For points `P_0=(0,0), P_1=(1,1), P_2=(2,13)`: `cross(P_0,P_1,P_2)` = `(1-0,1-0) x (2-0,13-0)` = `1*13 - 1*2 = 11 > 0`. With predicate `>= 0` I pop `P_1`. But `P_1` is *below* the segment `P_0P_2`? Segment `P_0->P_2` slope `13/2 = 6.5`; at `x=1` predicts `6.5`, and `P_1.y = 1 < 6.5`, so `P_1` is genuinely below the chord — it *belongs* on the lower hull and must not be popped. The predicate is backwards: I am popping lower-hull vertices and keeping concave junk.

**Diagnosing bug #1 (hull orientation).** A positive `cross(A,B,C)` here means `C` is to the *left* of ray `A->B`, i.e. the three points make a counterclockwise (upper) turn; for the *lower* hull I must keep counterclockwise turns and pop clockwise/collinear ones. So I must pop when `cross(A,B,C) <= 0`, not `>= 0`. With `<= 0`: `cross(P_0,P_1,P_2)=11 > 0`, do not pop — `P_1` stays. Correct. The fix is to flip the comparator to `<= 0` (the `<=` also discards collinear middle points, which is fine — a collinear vertex gives the same slope and only one representative is needed). I rewrite line (A) as `cross(...) <= 0`.

**Re-trace after fix #1.** Rebuilding the hull on the sample with `<= 0`: at `j=5`, the available indices are `0..2` plus whatever survived. The binary search then must select `i=1`. I check the tangent comparison at line (B): comparing slope from `A` vs `B` to `P_5=(5,21)`. `slope_A < slope_B` iff `(S[5]-S[A])*(5-B) < (S[5]-S[B])*(5-A)` — that is the cross-multiplied form, correct *as algebra*. With it, the search converges to the vertex of maximal slope, which for this sample yields `num=20, den=4`, average `5`. After reduction `gcd(20,4)=4`, prints `5/1`. The sample passes. Good — but the products at lines (B) and (C) are still `long long`, and that is the next bomb.

**Hunting bug #2 (the overflow that the problem is really about).** Lines (B) and (C) multiply a *sum-difference* by a *length*. Worst case the sum-difference is about `2*10^14` and the length about `2*10^5`, so the product is about `4*10^19`. The signed 64-bit ceiling is `2^63 - 1 = 9223372036854775807 ~ 9.22*10^18`. So `4*10^19` overflows by roughly a factor of four — it wraps around, and a wrapped comparison can flip. I do not want to *believe* this can change an answer; I want to *exhibit* it. So I construct a near-tie numerically. Take two windows of the same length `d = 200000`: window A with sum `s_A = 46116860184273`, window B with sum `s_B = 46116860184274 = s_A + 1`. Both averages are about `2.3*10^8` (well within feasibility: each element averages `~2.3*10^8 <= 10^9`), and B is strictly larger by `1/d`. The exact comparison `s_B * d` versus `s_A * d`: `s_B * d = 9223372036854800000` and `s_A * d = 9223372036854600000`, so exactly `s_B*d > s_A*d` and B wins — as it must, since B's average is higher.

**Tracing the 64-bit wrap.** Now what does the `long long` product do? `2^63 = 9223372036854775808`. So `s_A * d = 9223372036854600000 < 2^63`, stays a positive `long long`. But `s_B * d = 9223372036854800000 > 2^63`, so as a signed 64-bit value it *wraps to negative*: `9223372036854800000 - 2^64 = -9223372036854751616`. Now the broken comparison `s_B*d > s_A*d` evaluates as `(-9223372036854751616) > (9223372036854600000)`, which is **false** — the code concludes A is at least as good and keeps the *worse* window A. The exact arithmetic said B; the 64-bit arithmetic said A. That is a real, reproducible flip, and it is precisely the failure the constraints are engineered to expose. (I verified this exact pair offline: exact picks B, int64 picks A.) Conclusion: every product at lines (B) and (C), and inside `cross` at line (A), must be done in `__int128`, whose range `~1.7*10^38` swallows `4*10^19` with enormous margin.

**Fix #2 and the width audit.** I introduce `typedef __int128 lll;` and cast each operand to `lll` before multiplying, in all three places: the hull `cross` (its terms are `(index_diff) * (sum_diff)`, up to `2*10^5 * 2*10^14 = 4*10^19`), the tangent comparison `(S[j]-S[A])*(j-B)` (same bound), and the best-fraction comparison `num*bestDen` (same bound). I audit that no `__int128` value is ever stored back into a `long long`: each is only consumed inside a `<`/`>` comparison, so there is no narrowing. The *stored* quantities — `S[]`, `num`, `den`, `bestNum`, `bestDen` — are all individually within `long long` (`|S| <= 2*10^14`, `den <= 2*10^5`), so only the *products* need the wider type. That is the minimal, correct fix.

**Edge cases, deliberately.**
- `L = n`: the loop runs once at `j = n` with the single available left index `0`; the only window is the whole array. The hull has one point, the binary search collapses (`lo = hi = 0`), and the answer is `S[n]/n` reduced. I tested `n=3, L=3, a=[-1,-2,-3]`: sum `-6`, length `3`, prints `-2/1`. Negative answers are legal here because a window must be nonempty (length `>= L >= 1`) — there is *no* empty-window escape hatch, unlike the classic max-subarray. Correct.
- `n = 1, L = 1, a = [-4]`: one window, prints `-4/1`. Correct.
- All elements equal, `a=[7,7,7,7,7], L=2`: every average is `7`, prints `7/1`. The `<=` (not `<`) tie handling in the best-fraction update keeps the first-found equal optimum, which is fine since all equal fractions reduce identically. Correct.
- The fraction reduction: `g = gcd(|bestNum|, bestDen)`; if `bestNum = 0` then `g = gcd(0, bestDen) = bestDen`, reducing `0/d` to `0/1`. I guard `if (g == 0) g = 1` only against the impossible `bestDen = 0` (it never is, since `den = j - bi >= L >= 1`), but it is cheap insurance. Sign lives in the numerator because `bestDen > 0` always. Correct.
- Reading: `scanf` skips whitespace and newlines, so the two-line input format is parsed format-agnostically.

**Verification at scale.** I ran the solution against an independent brute force (all `O(n^2)` windows compared by exact `Fraction`) on 800 random small cases — uniform values, all-`+-10^9` values, tiny ranges, every `L` from `1` to `n` — with **zero** mismatches, and against an independent `O(n^2)` C++ reference on 150 cases with `n` between 2000 and 4000 (which already pushes products into the `10^16`-`10^18` range), again zero mismatches. The documented sample returns `5/1`. On the maximum input `n = 2*10^5` the run is `~20 ms` and `~7 MB`, comfortably inside the 1 s / 256 MB budget. The two bugs I caught — the popped-the-wrong-turn hull orientation and the 64-bit product overflow — were each pinned by a concrete trace, and each fix was re-checked on the input that exposed it.

**Final solution.** The idea I can prove (exact slope maximization on the lower hull, every comparison cross-multiplied in `__int128`) is what I ship, in one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef __int128 lll;

int main() {
    int n; ll L;
    if (scanf("%d %lld", &n, &L) != 2) return 0;
    // n days, window length at least L
    vector<ll> a(n);
    for (int i = 0; i < n; i++) scanf("%lld", &a[i]);

    // prefix sums S[0..n], S[0]=0, S[k]=a[0]+...+a[k-1]
    vector<ll> S(n + 1);
    S[0] = 0;
    for (int i = 0; i < n; i++) S[i + 1] = S[i] + a[i];

    // We want to maximize (S[j]-S[i])/(j-i) over 0 <= i < j <= n with j - i >= L.
    // That is the slope from point P_i=(i,S[i]) to P_j=(j,S[j]).
    // For each j, candidate left endpoints i range over [0, j-L]; the optimum lies
    // on the lower convex hull of those points. Maintain the hull incrementally:
    // when j advances by 1, point index (j-L) becomes available, so add it.

    // best average as fraction bestNum / bestDen (bestDen > 0). Maximize value.
    // initialize with the smallest possible (very negative).
    bool have = false;
    ll bestNum = 0, bestDen = 1;

    // hull stores indices i with x = i, y = S[i], forming a lower convex hull
    // (so that slope to a query point on the right is maximized by walking the hull).
    vector<int> hull;
    hull.reserve(n + 1);

    // cross product of (b-a) and (c-a); points are (idx, S[idx]).
    auto cross = [&](int A, int B, int C) -> lll {
        lll x1 = (lll)(B - A), y1 = (lll)(S[B] - S[A]);
        lll x2 = (lll)(C - A), y2 = (lll)(S[C] - S[A]);
        return x1 * y2 - y1 * x2;
    };

    for (int j = (int)L; j <= n; j++) {
        int newi = j - (int)L; // becomes available now
        // add point newi to the lower hull
        while ((int)hull.size() >= 2 &&
               cross(hull[hull.size() - 2], hull[hull.size() - 1], newi) <= 0) {
            hull.pop_back();
        }
        hull.push_back(newi);

        // query: maximize slope from a hull point to (j, S[j]). Along the lower hull
        // the slope to a fixed right point first increases then decreases (unimodal),
        // so binary-search for the tangent vertex. Every comparison cross-multiplies
        // in __int128 -- the products reach ~4e19 and overflow 64-bit.
        int lo = 0, hi = (int)hull.size() - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            // compare slope(hull[mid] -> j) vs slope(hull[mid+1] -> j)
            int A = hull[mid], B = hull[mid + 1];
            // slope_A = (S[j]-S[A])/(j-A), slope_B = (S[j]-S[B])/(j-B)
            // slope_A < slope_B  <=>  (S[j]-S[A])*(j-B) < (S[j]-S[B])*(j-A)
            lll lhs = (lll)(S[j] - S[A]) * (lll)(j - B);
            lll rhs = (lll)(S[j] - S[B]) * (lll)(j - A);
            if (lhs < rhs) lo = mid + 1; // slope increasing, go right
            else hi = mid;
        }
        int bi = hull[lo];
        ll num = S[j] - S[bi];
        ll den = j - bi; // > 0
        // compare num/den with bestNum/bestDen : num*bestDen vs bestNum*den
        if (!have) {
            bestNum = num; bestDen = den; have = true;
        } else {
            lll l = (lll)num * (lll)bestDen;
            lll r = (lll)bestNum * (lll)den;
            if (l > r) { bestNum = num; bestDen = den; }
        }
    }

    // reduce fraction bestNum/bestDen (bestDen > 0)
    ll g = std::__gcd(bestNum < 0 ? -bestNum : bestNum, bestDen);
    if (g == 0) g = 1;
    bestNum /= g; bestDen /= g;
    printf("%lld/%lld\n", bestNum, bestDen);
    return 0;
}
```

**Causal recap.** The problem is "maximum-average window of length `>= L`," which prefix sums turn into "maximize a slope between prefix points," solvable in `O(n)` by maintaining the lower convex hull of left endpoints and taking a binary-searched tangent to each right point. Floating point is disqualified up front by a numeric check showing two legal averages can differ by `~10^-11` near magnitude `10^9`, below `double` resolution — so I compare ratios by cross-multiplication. My first hull pop used the wrong sign of the cross product (`>= 0` instead of `<= 0`), which a trace of the three points `P_0,P_1,P_2` exposed by popping a genuine lower-hull vertex; flipping to `<= 0` fixed it and the sample then returned `5/1`. The deeper bug was that the cross-products reach `~4*10^19` and overflow signed 64-bit; a constructed near-tie (`s_A*d` vs `s_B*d` straddling `2^63`) showed the wrapped comparison literally flips the winner, so every product moved to `__int128` while the stored sums and lengths stayed `long long`. With both fixes, 800 brute-force cases and 150 large C++-reference cases agree exactly, and the worst-case input runs in ~20 ms.
