// TIER: greedy
// The "obvious first idea": collapse the chord to its complex value and do
// floating-point matching pursuit -- repeatedly strike the single bell that
// best reduces the residual magnitude. This never looks at the prime-cycle
// lattice at all. Because the n-th roots of unity are a highly REDUNDANT
// (linearly dependent) dictionary once n is composite, greedy single-atom
// pursuit is not guaranteed to rediscover a sparse decomposition -- it can
// wander and fail to land back on the target exactly. We verify exactness
// with the SAME exact cyclotomic-integer arithmetic the checker uses before
// trusting the floating-point result; if it did not converge to an exact
// match, we fall back to replaying the chord verbatim (no worse than
// trivial).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static vector<ll> polyDiv(vector<ll> a, const vector<ll> &b) {
    int da = (int)a.size() - 1, db = (int)b.size() - 1;
    vector<ll> q(da - db + 1, 0);
    for (int i = da; i >= db; i--) {
        ll coef = a[i];
        q[i - db] = coef;
        if (coef != 0) for (int j = 0; j <= db; j++) a[i - db + j] -= coef * b[j];
    }
    return q;
}
static map<int, vector<ll>> philib;
static vector<ll> getPhi(int n) {
    auto it = philib.find(n);
    if (it != philib.end()) return it->second;
    vector<ll> poly(n + 1, 0);
    poly[0] = -1; poly[n] = 1;
    for (int d = 1; d < n; d++) if (n % d == 0) poly = polyDiv(poly, getPhi(d));
    philib[n] = poly;
    return poly;
}
static vector<ll> reduceModPhi(vector<ll> c, int n) {
    vector<ll> poly = getPhi(n);
    int m = (int)poly.size() - 1;
    c.resize(n, 0);
    for (int i = n - 1; i >= m; i--) {
        if (c[i] != 0) {
            ll coef = c[i];
            for (int j = 0; j < m; j++) c[i - m + j] -= coef * poly[j];
            c[i] = 0;
        }
    }
    c.resize(m);
    return c;
}

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<ll> a(n);
    ll S = 0;
    for (int i = 0; i < n; i++) { cin >> a[i]; S += a[i]; }

    const double PI = acos(-1.0);
    vector<double> rc(n), rs(n);
    for (int i = 0; i < n; i++) { rc[i] = cos(2 * PI * i / n); rs[i] = sin(2 * PI * i / n); }

    double tx = 0, ty = 0;
    for (int i = 0; i < n; i++) { tx += (double)a[i] * rc[i]; ty += (double)a[i] * rs[i]; }

    double resx = tx, resy = ty;
    vector<ll> cnt(n, 0);
    ll cap = min((ll)200000, max((ll)50, 6 * S));
    bool converged = false;
    for (ll step = 0; step < cap; step++) {
        double mag2 = resx * resx + resy * resy;
        if (mag2 < 1e-14) { converged = true; break; }
        int best = 0; double bestProj = -1e18;
        for (int e = 0; e < n; e++) {
            double proj = resx * rc[e] + resy * rs[e];
            if (proj > bestProj) { bestProj = proj; best = e; }
        }
        resx -= rc[best]; resy -= rs[best];
        cnt[best]++;
    }
    if (!converged && resx * resx + resy * resy < 1e-14) converged = true;

    bool exact = false;
    if (converged) {
        vector<ll> ra = reduceModPhi(a, n);
        vector<ll> rb = reduceModPhi(cnt, n);
        exact = (ra == rb);
    }

    if (exact) {
        ll t = 0; for (int i = 0; i < n; i++) t += cnt[i];
        cout << t << "\n";
        for (int i = 0; i < n; i++) for (ll k = 0; k < cnt[i]; k++) cout << i << ' ';
        cout << "\n";
    } else {
        cout << S << "\n";
        for (int i = 0; i < n; i++) for (ll k = 0; k < a[i]; k++) cout << i << ' ';
        cout << "\n";
    }
    return 0;
}
