// TIER: strong
// Insight: don't schedule aircraft one at a time -- schedule the QUEUE into waves.
// (1) Upstream shaping: within each gate alley, order pushbacks by WEIGHT (heavy
//     first) so the aircraft that cost the most reach the crossing earliest --
//     the pushback-blocking spacing then assembles the right wave.
// (2) Wave packing: assign crossing ticks in WEIGHT order, each to the earliest
//     free tick of an open slot (fill each wave's W lanes with the heaviest first).
// (3) Local exchange: swap crossing ticks between pairs whenever it lowers weighted
//     delay and stays feasible -- polishes the wave assignment.
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
    // Phase 1: pushback, per alley, heavy first (tie: earlier release).
    vector<vector<int>> byAlley(maxA + 1);
    for (int i = 0; i < N; i++) byAlley[alley[i]].push_back(i);
    vector<ll> s(N), arr(N);
    for (int a = 0; a <= maxA; a++) {
        auto& v = byAlley[a];
        sort(v.begin(), v.end(), [&](int i, int j){
            if (w[i] != w[j]) return w[i] > w[j];
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
    // Phase 2: cross in weight order, earliest free slot tick (cap W).
    vector<int> ord(N);
    iota(ord.begin(), ord.end(), 0);
    sort(ord.begin(), ord.end(), [&](int i, int j){
        if (w[i] != w[j]) return w[i] > w[j];
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
    // Phase 3: pairwise exchange of crossing ticks (only improving + feasible).
    // Sorting by weight desc, then scanning for a lighter aircraft that currently
    // holds an earlier tick a heavy one could legally take.
    for (int pass = 0; pass < 8; pass++) {
        bool improved = false;
        for (int a = 0; a < N; a++) {
            for (int b = a + 1; b < N; b++) {
                // feasibility of swapping x[a] and x[b]
                if (x[b] < arr[a] || x[a] < arr[b]) continue;
                ll before = w[a] * (x[a] - (rel[a] + tau[a])) +
                            w[b] * (x[b] - (rel[b] + tau[b]));
                ll after  = w[a] * (x[b] - (rel[a] + tau[a])) +
                            w[b] * (x[a] - (rel[b] + tau[b]));
                if (after < before) { swap(x[a], x[b]); improved = true; }
            }
        }
        if (!improved) break;
    }
    for (int i = 0; i < N; i++) printf("%lld %lld\n", s[i], x[i]);
    return 0;
}
