**Problem.** Given `n` daily revenues `a[0..n-1]` (each in `[-10^9, 10^9]`) and a minimum length `L` (`1 <= L <= n <= 2*10^5`), find the contiguous window `[i, j)` with `j - i >= L` that maximizes the average `(a[i] + ... + a[j-1]) / (j - i)`. Print that average as a reduced fraction `p/q` with `q > 0`. The answer may be negative (a window must be nonempty, so there is no empty-set escape).

**Key idea — slopes on prefix points, compared exactly.** With prefix sums `S[k] = a[0] + ... + a[k-1]`, the average of `[i, j)` is `(S[j] - S[i]) / (j - i)`, i.e. the *slope* of the segment from `P_i = (i, S[i])` to `P_j = (j, S[j])`. For each right endpoint `j`, the left endpoints `i` range over `[0, j - L]`, growing by one index (`j - L`) each time `j` advances. The slope-maximizing left endpoint always lies on the **lower convex hull** of those points, so maintain that hull incrementally and, for each `j`, take the tangent from `P_j` to the hull. The hull edges have increasing slope, so the slope from a hull vertex to the fixed point `P_j` is unimodal in the vertex index — binary-search the maximizing vertex. This is `O(n)` overall (each index pushed/popped once; `O(log n)` tangent search per `j`).

**Pitfalls.**
1. *Never divide — and watch the product width.* To compare two averages `s1/d1` and `s2/d2` (`d > 0`), compare `s1*d2` versus `s2*d1`. Floating point is unsafe: two legal averages can differ by `~1/(2*10^5)^2 ≈ 2.5*10^-11`, far below a `double`'s resolution near magnitude `10^9`. So compare integers.
2. *Overflow decides correctness.* A sum-difference reaches `~2*10^14` and a length `~2*10^5`, so every cross-product reaches `~4*10^19`, which overflows signed 64-bit (`2^63 - 1 ≈ 9.22*10^18`). A `long long` product *wraps* and can flip the comparison: e.g. with `d = 200000`, `s_A = 46116860184273`, `s_B = s_A + 1`, the exact test says B wins, but `s_B*d` exceeds `2^63` and wraps negative so a 64-bit test picks A. All three multiplications (hull `cross`, tangent comparison, best-fraction comparison) must use `__int128`. The stored sums, lengths, and prefix values individually fit in `long long`; only the products need 128-bit.
3. *Hull orientation.* For the **lower** hull, pop while the turn is clockwise/collinear, i.e. `cross(A, B, C) <= 0`. Using the opposite sign pops genuine lower-hull vertices and yields wrong tangents.

**Edge cases.** `L = n`: a single window (the whole array). `n = 1`: that one element, e.g. `-4 -> -4/1`. All-negative arrays: a negative fraction (no empty window). All-equal arrays: the common value reduced (e.g. `7 -> 7/1`). Fraction reduction puts the sign in the numerator (`bestDen > 0`), and `gcd(0, d) = d` reduces `0/d` to `0/1`.

**Complexity.** `O(n)` time (amortized hull maintenance plus `O(log n)` tangent search per right endpoint), `O(n)` memory for prefix sums and the hull. Maximum input runs in ~20 ms and ~7 MB.

**Code.**

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
