// chk.cc -- testlib checker/scorer for fsx_S_0985 (cyclotomic-sparse-sum)
// Feasibility: the participant's strike multiset, reduced modulo Phi_n(x),
// must equal the target's strike counts, reduced modulo Phi_n(x) -- i.e.
// exact equality as elements of Z[zeta_n]. Score: minimize the strike count.
#include "testlib.h"
#include <vector>
#include <map>
#include <algorithm>
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

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt(6, 60, "n");
    vector<ll> a(n);
    ll S = 0;
    for (int i = 0; i < n; i++) {
        a[i] = inf.readLong(0LL, 2000LL, "a_i");
        S += a[i];
    }
    if (S <= 0) quitf(_fail, "generator bug: empty chord");

    ll t = ouf.readLong(0LL, 2000000LL, "t");
    vector<ll> b(n, 0);
    for (ll k = 0; k < t; k++) {
        int e = ouf.readInt(0, n - 1, "e_k");
        b[e]++;
    }
    if (!ouf.seekEof()) quitf(_pe, "trailing tokens after output");

    vector<ll> ra = reduceModPhi(a, n);
    vector<ll> rb = reduceModPhi(b, n);
    if (ra.size() != rb.size()) quitf(_fail, "internal size mismatch");
    for (size_t i = 0; i < ra.size(); i++)
        if (ra[i] != rb[i])
            quitf(_wa, "chord mismatch at reduced coordinate %d: got %lld want %lld",
                  (int)i, rb[i], ra[i]);

    ll F = t;
    ll B = S;
    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
