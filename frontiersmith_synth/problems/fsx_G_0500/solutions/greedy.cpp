// TIER: greedy
// One improvement sweep from the canonical baseline: visit each pair once and
// flip its host orientation iff that lowers the two involved teams' combined
// travel+break cost. A single pass -- no restarts, no re-sweeping.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, R, G;
ll lambda;
vector<ll> X, Y;
vector<array<int,2>> gameTeams;
vector<vector<int>> teamSlot;
vector<int> home;                 // home[pos]

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

    // pairs + baseline orientation
    struct Pr { int i, j, pe, pl; };
    vector<Pr> pairs;
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
            home[lst[0].second] = i; home[lst[1].second] = j;
            pairs.push_back({i, j, lst[0].second, lst[1].second});
        }
    }

    auto flip = [&](const Pr& p) {
        home[p.pe] = (home[p.pe] == p.i) ? p.j : p.i;
        home[p.pl] = (home[p.pl] == p.i) ? p.j : p.i;
    };

    // one sweep
    for (auto& p : pairs) {
        ll before = costOfTeam(p.i) + costOfTeam(p.j);
        flip(p);
        ll after = costOfTeam(p.i) + costOfTeam(p.j);
        if (after >= before) flip(p);      // revert if not strictly better
    }

    for (int r = 0; r < R; r++)
        for (int g = 0; g < G; g++) printf("%d%c", home[roundPos[r][g]], g + 1 < G ? ' ' : '\n');
    return 0;
}
