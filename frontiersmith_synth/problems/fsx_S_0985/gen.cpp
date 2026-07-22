// gen.cpp -- testlib generator for fsx_S_0985 (cyclotomic-sparse-sum)
// Builds, for each testId, a target chord a_0..a_{n-1} as:
//   (1) tstar random "base" single-tone strikes (the hidden true signal), then
//   (2) a bunch of "padding" full prime cycles (rotated, random prime p|n and
//       random offset) added on top -- each padding cycle leaves the chord's
//       complex sound unchanged (it is an exact vanishing sum) but inflates
//       the naive strike count, planting the trap for non-lattice-aware
//       solvers. Determinism: all randomness comes from testlib's `rnd`,
//       seeded solely by testId (registerGen contract).
#include "testlib.h"
#include <vector>
#include <array>
#include <map>
using namespace std;
typedef long long ll;

// ---- shared cyclotomic-polynomial machinery (mirrors chk.cc / solutions) ----
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
static bool isZeroVec(const vector<ll> &v) { for (ll x : v) if (x != 0) return false; return true; }

static vector<int> primeFactors(int n) {
    vector<int> ps; int m = n;
    for (int p = 2; (ll)p * p <= m; p++) if (m % p == 0) { ps.push_back(p); while (m % p == 0) m /= p; }
    if (m > 1) ps.push_back(m);
    return ps;
}

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ladder: {n, tstar, padMultiplier}  -- padding target terms ~= padMultiplier * tstar
    // n is always composite with EXACTLY TWO distinct prime factors -- this is
    // the regime where the Lam-Leung theorem guarantees the nonnegative
    // relation lattice is generated purely by rotated p-cycles and q-cycles
    // (with >=3 distinct prime factors this guarantee can fail).
    static const array<array<int, 3>, 10> cfg = {{
        {{6, 3, 2}},    // 1: tiny sanity, n=6  = 2*3
        {{10, 5, 2}},   // 2: n=10 = 2*5
        {{12, 6, 2}},   // 3: n=12 = 2^2*3
        {{14, 10, 3}},  // 4: n=14 = 2*7   -- trap
        {{15, 8, 2}},   // 5: n=15 = 3*5
        {{21, 12, 3}},  // 6: n=21 = 3*7   -- trap
        {{33, 14, 3}},  // 7: n=33 = 3*11  -- trap
        {{35, 16, 3}},  // 8: n=35 = 5*7   -- trap
        {{45, 20, 3}},  // 9: n=45 = 3^2*5 -- trap
        {{55, 24, 4}},  // 10: n=55 = 5*11, largest signal + heaviest padding -- trap
    }};

    int n = cfg[testId - 1][0];
    int tstar = cfg[testId - 1][1];
    int padMul = cfg[testId - 1][2];
    vector<int> primes = primeFactors(n);

    vector<ll> a(n, 0);

    // (1) base signal: tstar random single-tone strikes; retry-safe against the
    // astronomically unlikely event that the base signal itself vanishes
    // (guarantees T != 0 so a nonzero minimal representation always exists).
    for (int attempt = 0; attempt < 25; attempt++) {
        fill(a.begin(), a.end(), 0);
        int k = tstar + attempt; // widen slightly on retry to change the draw
        for (int t = 0; t < k; t++) a[rnd.next(0, n - 1)]++;
        vector<ll> r = reduceModPhi(a, n);
        if (!isZeroVec(r)) break;
    }

    // (2) padding: repeatedly add a full rotated prime cycle (random prime
    // divisor of n, random rotation) until the padding budget is spent. Each
    // cycle sums to exactly 0, so T is unchanged, but the naive strike count
    // (sum a_i) balloons -- the trap for solvers that don't see the lattice.
    long long padTarget = (long long)padMul * tstar;
    long long padAdded = 0;
    int guard = 0;
    while (padAdded < padTarget && guard < 200000) {
        guard++;
        int p = primes[rnd.next(0, (int)primes.size() - 1)];
        int step = n / p;
        int j = rnd.next(0, step - 1);
        for (int k = 0; k < p; k++) a[j + k * step]++;
        padAdded += p;
    }

    printf("%d\n", n);
    for (int i = 0; i < n; i++) printf("%lld%c", a[i], i + 1 < n ? ' ' : '\n');
    return 0;
}
