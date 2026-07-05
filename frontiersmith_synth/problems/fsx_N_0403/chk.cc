#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static ll powmod(ll a, ll e, ll m){ a%=m; if(a<0)a+=m; ll r=1%m; while(e){ if(e&1) r=(__int128)r*a%m; a=(__int128)a*a%m; e>>=1;} return r; }
static bool isNZQR(ll s, ll p){ s%=p; if(s<0)s+=p; if(s==0) return false; return powmod(s,(p-1)/2,p)==1; }

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    ll N = inf.readInt(1, 200000, "N");
    ll M = inf.readInt(1, (int)N, "M");
    ll C = inf.readLong(1LL, 1000000000LL, "C");
    ll p = inf.readLong(3LL, 2000003LL, "p");

    vector<ll> w(N + 1), c(N + 1), k(N + 1);
    for (ll i = 1; i <= N; i++) {
        w[i] = inf.readLong(1LL, 1000000LL, "w");
        c[i] = inf.readLong(1LL, C, "c");
        k[i] = inf.readLong(0LL, p - 1, "k");
    }

    auto yield_of = [&](ll W, ll U) -> ll {
        ll bonus = (ll)((__int128)W * (__int128)U / (__int128)(2 * C));
        return W + bonus;
    };

    // ---- reference baseline B: the M best individually-stable singletons ----
    vector<ll> singObj;
    singObj.reserve(N);
    for (ll i = 1; i <= N; i++)
        if (isNZQR(k[i], p)) singObj.push_back(yield_of(w[i], c[i]));  // c[i] <= C guaranteed
    sort(singObj.begin(), singObj.end(), greater<ll>());
    ll B = 0;
    for (ll i = 0; i < (ll)singObj.size() && i < M; i++) B += singObj[i];
    if (B < 1) B = 1;

    // ---- read participant output ----
    ll G = ouf.readInt(0, (int)M, "G");
    vector<char> used(N + 1, 0);
    ll F = 0;
    ll totalItems = 0;
    for (ll g = 0; g < G; g++) {
        ll s = ouf.readInt(1, (int)N, "loop_size");
        totalItems += s;
        if (totalItems > N) quitf(_wa, "loop %lld: total rigs across loops exceeds N", g + 1);
        ll W = 0, U = 0, keysum = 0;
        for (ll t = 0; t < s; t++) {
            ll idx = ouf.readInt(1, (int)N, "rig_index");
            if (used[idx]) quitf(_wa, "rig %lld assigned to more than one loop", idx);
            used[idx] = 1;
            W += w[idx];
            U += c[idx];
            keysum = (keysum + k[idx]) % p;
        }
        if (U > C) quitf(_wa, "loop %lld: coolant %lld exceeds budget C=%lld", g + 1, U, C);
        if (!isNZQR(keysum, p))
            quitf(_wa, "loop %lld: key-sum %lld mod p is not a nonzero quadratic residue", g + 1, keysum);
        F += yield_of(W, U);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %lld loops", G);

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld G=%lld Ratio: %.6f", F, B, G, sc / 1000.0);
    return 0;
}
