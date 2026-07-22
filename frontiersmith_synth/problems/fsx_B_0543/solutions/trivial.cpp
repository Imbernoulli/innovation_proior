// TIER: trivial
// Reproduces the checker's baseline exactly: input order, push back ASAP subject
// to release + alley, then cross ALONE in the next free slot (one per slot).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static ll nextFreeTick(ll a, ll P, ll cap, unordered_set<ll>& used) {
    ll t = a < 0 ? 0 : a;
    while (true) {
        ll off = t % P;
        if (off >= cap) { t += (P - off); continue; }
        if (used.count(t)) { t++; continue; }
        return t;
    }
}

int main() {
    int N, P, W, d;
    scanf("%d %d %d %d", &N, &P, &W, &d);
    vector<ll> rel(N), tau(N), w(N);
    vector<int> alley(N);
    int maxA = 0;
    for (int i = 0; i < N; i++) {
        scanf("%lld %lld %lld %d", &rel[i], &tau[i], &w[i], &alley[i]);
        maxA = max(maxA, alley[i]);
    }
    vector<ll> lastPB(maxA + 1, LLONG_MIN);
    unordered_set<ll> used;
    used.reserve(N * 2);
    vector<ll> s(N), x(N);
    for (int i = 0; i < N; i++) {
        ll si = rel[i];
        if (lastPB[alley[i]] != LLONG_MIN) si = max(si, lastPB[alley[i]] + d);
        lastPB[alley[i]] = si;
        ll a = si + tau[i];
        ll xi = nextFreeTick(a, P, 1, used); // one per slot
        used.insert(xi);
        s[i] = si; x[i] = xi;
    }
    for (int i = 0; i < N; i++) printf("%lld %lld\n", s[i], x[i]);
    return 0;
}
