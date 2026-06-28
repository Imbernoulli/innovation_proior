#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

struct P { ll x, y; };

// cross product of (B-A) x (C-A)
static inline ll cross(const P& A, const P& B, const P& C) {
    return (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
}
static inline ll dist2(const P& A, const P& B) {
    ll dx = A.x - B.x, dy = A.y - B.y;
    return dx * dx + dy * dy;
}

// Andrew's monotone chain; returns hull in CCW order, strict (no collinear interior pts).
static vector<P> convexHull(vector<P> pts) {
    sort(pts.begin(), pts.end(), [](const P& a, const P& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    pts.erase(unique(pts.begin(), pts.end(), [](const P& a, const P& b) {
        return a.x == b.x && a.y == b.y;
    }), pts.end());
    int n = (int)pts.size();
    if (n <= 2) return pts; // 1 unique pt -> [p]; 2 -> [p,q]
    vector<P> h(2 * n);
    int k = 0;
    // lower hull
    for (int i = 0; i < n; i++) {
        while (k >= 2 && cross(h[k - 2], h[k - 1], pts[i]) <= 0) k--;
        h[k++] = pts[i];
    }
    // upper hull
    for (int i = n - 2, lo = k + 1; i >= 0; i--) {
        while (k >= lo && cross(h[k - 2], h[k - 1], pts[i]) <= 0) k--;
        h[k++] = pts[i];
    }
    h.resize(k - 1); // last point == first point, drop it
    return h;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<P> pts(n);
    for (int i = 0; i < n; i++) cin >> pts[i].x >> pts[i].y;

    vector<P> h = convexHull(pts);
    int m = (int)h.size();

    ll best = 0;
    if (m == 1) {
        // all points identical: only pair distance is 0
        best = 0;
    } else if (m == 2) {
        // hull is a single segment (all points collinear / only 2 unique)
        best = dist2(h[0], h[1]);
    } else {
        // Rotating calipers: for each edge (i, i+1) advance antipodal vertex j
        // while the triangle area (twice) grows; candidate diameters are at the
        // antipodal vertices. O(m).
        int j = 1;
        for (int i = 0; i < m; i++) {
            int ni = (i + 1) % m;
            // advance j while area of triangle(h[i], h[ni], h[j+1]) > area with h[j]
            while (cross(h[i], h[ni], h[(j + 1) % m]) > cross(h[i], h[ni], h[j])) {
                j = (j + 1) % m;
            }
            // both endpoints of the edge are antipodal to h[j]
            best = max(best, max(dist2(h[i], h[j]), dist2(h[ni], h[j])));
        }
    }

    cout << best << "\n";
    return 0;
}
