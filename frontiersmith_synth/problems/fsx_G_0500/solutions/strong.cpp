// TIER: strong
// Multi-restart local search over host orientations. Restart 0 starts from the
// canonical baseline; further restarts start from random orientations. Each
// restart sweeps pairs to convergence (accepting any strictly-improving flip);
// the best full assignment found across all restarts is emitted. Deterministic
// RNG (fixed seed) -- no wall-clock, no GPU.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, R, G;
ll lambda;
vector<ll> X, Y;
vector<array<int,2>> gameTeams;
vector<vector<int>> teamSlot;
vector<int> home;

static inline ll dist(int a, int b) { return llabs(X[a] - X[b]) + llabs(Y[a] - Y[b]); }

ll costOfTeam(int t) {
    ll travel = 0, breaks = 0;
    int prev = t, prevHA = -1;
    for (int r = 0; r < R; r++) {
        int venue = home[teamSlot[t][r]];
        travel += dist(prev, venue);
        prev = venue;
        int ha = (venue == t) ? 1 : 0;
        if (prevHA != -1 && ha == prevHA) breaks++;
        prevHA = ha;
    }
    travel += dist(prev, t);
    return travel + lambda * breaks;
}

int main() {
    if (scanf("%d %d %lld", &n, &R, &lambda) != 3) return 0;
    G = n / 2;
    X.assign(n + 1, 0); Y.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) scanf("%lld %lld", &X[i], &Y[i]);
    vector<vector<int>> roundPos(R);
    teamSlot.assign(n + 1, vector<int>(R, -1));
    for (int r = 0; r < R; r++)
        for (int g = 0; g < G; g++) {
            int a, b; scanf("%d %d", &a, &b);
            int pos = (int)gameTeams.size();
            gameTeams.push_back({a, b});
            roundPos[r].push_back(pos);
            teamSlot[a][r] = pos; teamSlot[b][r] = pos;
        }
    int P = (int)gameTeams.size();
    home.assign(P, 0);

    struct Pr { int i, j, pe, pl; };
    vector<Pr> pairs;
    vector<int> baseHome(P, 0);
    {
        map<pair<int,int>, vector<pair<int,int>>> meet;
        for (int r = 0; r < R; r++)
            for (int pos : roundPos[r]) {
                int a = gameTeams[pos][0], b = gameTeams[pos][1];
                if (a > b) swap(a, b);
                meet[{a, b}].push_back({r, pos});
            }
        for (auto& kv : meet) {
            auto lst = kv.second; sort(lst.begin(), lst.end());
            int i = kv.first.first, j = kv.first.second;
            baseHome[lst[0].second] = i; baseHome[lst[1].second] = j;
            pairs.push_back({i, j, lst[0].second, lst[1].second});
        }
    }

    auto flip = [&](const Pr& p) {
        home[p.pe] = (home[p.pe] == p.i) ? p.j : p.i;
        home[p.pl] = (home[p.pl] == p.i) ? p.j : p.i;
    };
    auto setOrient = [&](const Pr& p, int early) {  // early = team hosting early game
        home[p.pe] = early;
        home[p.pl] = (early == p.i) ? p.j : p.i;
    };

    mt19937 rng(987654321u);
    int K = 24, sweepCap = 40;
    ll bestTotal = LLONG_MAX;
    vector<int> bestHome;

    for (int restart = 0; restart < K; restart++) {
        if (restart == 0) home = baseHome;
        else for (auto& p : pairs) setOrient(p, (rng() & 1) ? p.i : p.j);

        for (int sweep = 0; sweep < sweepCap; sweep++) {
            bool improved = false;
            for (auto& p : pairs) {
                ll before = costOfTeam(p.i) + costOfTeam(p.j);
                flip(p);
                ll after = costOfTeam(p.i) + costOfTeam(p.j);
                if (after < before) improved = true;
                else flip(p);
            }
            if (!improved) break;
        }
        ll tot = 0;
        for (int t = 1; t <= n; t++) tot += costOfTeam(t);
        if (tot < bestTotal) { bestTotal = tot; bestHome = home; }
    }

    home = bestHome;
    for (int r = 0; r < R; r++)
        for (int g = 0; g < G; g++) printf("%d%c", home[roundPos[r][g]], g + 1 < G ? ' ' : '\n');
    return 0;
}
