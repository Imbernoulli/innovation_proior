// TIER: greedy
// The obvious FCFS heuristic: push everyone back as early as possible in RELEASE
// order (per alley), then let the runway pass aircraft in ARRIVAL order, each into
// the earliest free slot tick.  It packs waves (cap W) but ignores weight and never
// shapes the pushback queue -> heavy aircraft can spill to late waves.
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
    // Phase 1: pushback ASAP, per alley, in release order.
    vector<vector<int>> byAlley(maxA + 1);
    for (int i = 0; i < N; i++) byAlley[alley[i]].push_back(i);
    vector<ll> s(N), arr(N);
    for (int a = 0; a <= maxA; a++) {
        auto& v = byAlley[a];
        sort(v.begin(), v.end(), [&](int i, int j){
            if (rel[i] != rel[j]) return rel[i] < rel[j];
            return i < j;
        });
        ll last = LLONG_MIN;
        for (int i : v) {
            ll si = rel[i];
            if (last != LLONG_MIN) si = max(si, last + d);
            last = si;
            s[i] = si; arr[i] = si + tau[i];
        }
    }
    // Phase 2: cross in arrival order, earliest free slot tick (cap W).
    vector<int> ord(N);
    iota(ord.begin(), ord.end(), 0);
    sort(ord.begin(), ord.end(), [&](int i, int j){
        if (arr[i] != arr[j]) return arr[i] < arr[j];
        return i < j;
    });
    unordered_set<ll> used;
    used.reserve(N * 2);
    vector<ll> x(N);
    for (int i : ord) {
        ll xi = nextFreeTick(arr[i], P, W, used);
        used.insert(xi);
        x[i] = xi;
    }
    for (int i = 0; i < N; i++) printf("%lld %lld\n", s[i], x[i]);
    return 0;
}
