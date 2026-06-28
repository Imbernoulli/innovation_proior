#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

struct P { ll x, y; };

// 2x signed area of triangle (o,a,b); positive if a->b is a left turn around o.
static inline ll cross(const P& o, const P& a, const P& b) {
    return (ll)(a.x - o.x) * (b.y - o.y) - (ll)(a.y - o.y) * (b.x - o.x);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<P> pts(n);
    for (int i = 0; i < n; i++) cin >> pts[i].x >> pts[i].y;

    // Fewer than 3 points: no triangle.
    if (n < 3) { cout << 0 << "\n"; return 0; }

    // --- Andrew's monotone chain convex hull ---
    sort(pts.begin(), pts.end(), [](const P& a, const P& b) {
        return a.x != b.x ? a.x < b.x : a.y < b.y;
    });
    // Remove exact duplicate points (they cannot help and confuse the chain).
    pts.erase(unique(pts.begin(), pts.end(), [](const P& a, const P& b) {
        return a.x == b.x && a.y == b.y;
    }), pts.end());
    int m = (int)pts.size();
    if (m < 3) { cout << 0 << "\n"; return 0; }

    vector<P> hull(2 * m);
    int k = 0;
    // lower hull
    for (int i = 0; i < m; i++) {
        while (k >= 2 && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) k--;
        hull[k++] = pts[i];
    }
    // upper hull
    for (int i = m - 2, lo = k + 1; i >= 0; i--) {
        while (k >= lo && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) k--;
        hull[k++] = pts[i];
    }
    hull.resize(k - 1); // last point == first, drop it
    int h = (int)hull.size();

    // All points collinear -> hull is a segment (h < 3): max area triangle is 0.
    if (h < 3) { cout << 0 << "\n"; return 0; }

    // --- Maximum-area triangle on the convex hull ---
    // Fix apex i; advance pointers j (= i+1 initially) and l (= i+2 initially)
    // monotonically forward over the hull. For a fixed (i, j), the function
    //   l -> 2*area(hull[i], hull[j], hull[l])
    // is unimodal as l sweeps from j+1 around to i-1, so we push l forward while
    // the area keeps growing. As j advances, the optimal l is monotone non-
    // decreasing (the interleaving property of 2-stable triangles), so l never
    // needs to be reset back behind its current value within one apex.
    // We RESET j and l for every apex i (this is the fix to the broken O(n)
    // Dobkin-Snyder scan, which moves the apex without resetting and misses the
    // optimum). Cost: O(h) per apex => O(h^2) overall.
    ll best = 0;
    for (int i = 0; i < h; i++) {
        int j = (i + 1) % h;
        int l = (i + 2) % h;
        // For this apex, sweep j around; l chases j monotonically.
        // We stop when j has gone all the way around back toward i.
        while (true) {
            // advance l while area strictly increases
            while (true) {
                int ln = (l + 1) % h;
                if (ln == i) break;                 // l must stay strictly before i
                ll cur = llabs(cross(hull[i], hull[j], hull[l]));
                ll nxt = llabs(cross(hull[i], hull[j], hull[ln]));
                if (nxt >= cur) l = ln; else break;
            }
            ll area2 = llabs(cross(hull[i], hull[j], hull[l]));
            if (area2 > best) best = area2;

            int jn = (j + 1) % h;
            if (jn == i) break;                     // j wrapped to apex: done
            // l must remain strictly ahead of j; if j catches up, push l along.
            j = jn;
            if (l == j) {
                int ln = (l + 1) % h;
                if (ln == i) break;                 // no room for a third vertex
                l = ln;
            }
        }
    }

    cout << best << "\n";
    return 0;
}
