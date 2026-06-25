#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef long long lll; // BROKEN: int64 instead of int128

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

        // query: maximize slope from a hull point to (j, S[j]).
        // On the lower hull, slopes to a fixed right point are unimodal (increase then
        // decrease), so binary-search / two-pointer for the max. Use a pointer that only
        // moves forward is NOT valid because j changes; do a local search with pointer.
        // Simpler robust approach: binary search on the hull for the best tangent.
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
