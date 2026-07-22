// TIER: strong
// The insight: sum-of-worst-two lets you sacrifice five seasons for free, so the
// only thing worth optimizing is WHICH pair of seasons ends up binding and how
// little damage lands there -- not the total weighted conflict. Instead of a
// total-weight proxy, run coordinate descent DIRECTLY on the true objective: for
// every tower, evaluate the resulting top-two-season sum for every candidate
// channel (via an O(deg) incremental delta) and take the channel that minimizes
// it. This automatically (a) steers collisions away from whatever pair is
// currently the binding top-two, since that is exactly what the recomputed
// objective penalizes, and (b) load-balances the sacrificed conflict across the
// five non-binding seasons, because once one of them grows enough to threaten
// entering the top two, the same recomputation starts penalizing it too.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, C, M;
vector<vector<pair<int, array<int,8>>>> adj; // adj[t] = {(other, w[1..7])}
ll S[8];

ll topTwoSum(const ll s[8]) {
    ll a = -1, b = -1;
    for (int i = 1; i <= 7; i++) {
        if (s[i] > a) { b = a; a = s[i]; }
        else if (s[i] > b) { b = s[i]; }
    }
    if (b < 0) b = 0;
    return a + b;
}

int main() {
    scanf("%d %d %d", &N, &C, &M);
    adj.assign(N + 1, {});
    for (int i = 0; i < M; i++) {
        int u, v; scanf("%d %d", &u, &v);
        array<int,8> w{};
        for (int s = 1; s <= 7; s++) scanf("%d", &w[s]);
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    vector<int> ch(N + 1);
    for (int i = 1; i <= N; i++) ch[i] = ((i - 1) % C) + 1;

    for (int s = 1; s <= 7; s++) S[s] = 0;
    for (int t = 1; t <= N; t++) {
        for (auto &pr : adj[t]) {
            int other = pr.first;
            if (other <= t) continue; // count each undirected edge once
            if (ch[t] == ch[other]) for (int s = 1; s <= 7; s++) S[s] += pr.second[s];
        }
    }

    const int MAXPASS = 60;
    vector<array<ll,8>> agg(C + 1);
    for (int pass = 0; pass < MAXPASS; pass++) {
        bool improved = false;
        for (int t = 1; t <= N; t++) {
            for (int c = 1; c <= C; c++) for (int s = 0; s <= 7; s++) agg[c][s] = 0;
            for (auto &pr : adj[t]) {
                int other = pr.first;
                int c = ch[other];
                for (int s = 1; s <= 7; s++) agg[c][s] += pr.second[s];
            }
            int old = ch[t];
            ll curF = topTwoSum(S);
            int bestC = old; ll bestF = curF;
            for (int c = 1; c <= C; c++) {
                if (c == old) continue;
                ll trial[8];
                for (int s = 1; s <= 7; s++) trial[s] = S[s] - agg[old][s] + agg[c][s];
                ll f = topTwoSum(trial);
                if (f < bestF) { bestF = f; bestC = c; }
            }
            if (bestC != old) {
                for (int s = 1; s <= 7; s++) S[s] += agg[bestC][s] - agg[old][s];
                ch[t] = bestC;
                improved = true;
            }
        }
        if (!improved) break;
    }

    for (int i = 1; i <= N; i++) printf("%d%c", ch[i], i == N ? '\n' : ' ');
    return 0;
}
