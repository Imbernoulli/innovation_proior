#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    ll n   = inf.readLong();
    int k  = (int)inf.readLong();
    ll m   = inf.readLong();
    ll lam = inf.readLong();
    ll L   = inf.readLong();
    ll U   = inf.readLong();

    vector<ll> w(n);
    for (ll i = 0; i < n; i++) w[i] = inf.readLong();
    vector<ll> t(k);
    for (int j = 0; j < k; j++) t[j] = inf.readLong();

    ll E = inf.readLong();
    vector<int> eu(E), ev(E);
    vector<ll> ec(E);
    for (ll e = 0; e < E; e++) {
        eu[e] = (int)inf.readLong() - 1;
        ev[e] = (int)inf.readLong() - 1;
        ec[e] = inf.readLong();
    }

    // ---- read participant assignment ----
    vector<int> a(n);
    vector<ll> cnt(k, 0);
    for (ll i = 0; i < n; i++) {
        a[i] = ouf.readInt(1, k, "assign") - 1;
        cnt[a[i]]++;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %lld assignments", n);

    for (int j = 0; j < k; j++)
        if (cnt[j] < L || cnt[j] > U)
            quitf(_wa, "track %d holds %lld cars, outside balance band [%lld,%lld]",
                  j + 1, cnt[j], L, U);

    // ---- objective F of an assignment ----
    auto objective = [&](const vector<int>& asg) -> ll {
        vector<ll> S(k, 0), bnd(k, 0);
        for (ll i = 0; i < n; i++) S[asg[i]] += w[i];
        for (ll e = 0; e < E; e++) {
            if (asg[eu[e]] != asg[ev[e]]) {
                bnd[asg[eu[e]]] += ec[e];
                bnd[asg[ev[e]]] += ec[e];
            }
        }
        ll F = 0;
        for (int j = 0; j < k; j++) {
            int tuned = ((S[j] % m) == t[j]) ? 1 : 0;
            F += bnd[j] * (1 + lam * (ll)tuned);
        }
        return F;
    };

    ll F = objective(a);

    // ---- internal baseline: round-robin routing (always feasible) ----
    vector<int> br(n);
    for (ll i = 0; i < n; i++) br[i] = (int)(i % k);
    ll B = objective(br);
    if (B < 1) B = 1;

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
