#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, R;
ll lambda;
vector<ll> X, Y;                       // 1-indexed coords
vector<array<int,2>> gameTeams;        // flattened games: pos -> {a,b}
vector<vector<int>> roundPos;          // roundPos[r] = list of flat game positions
vector<vector<int>> teamSlot;          // teamSlot[t][r] = flat game position of team t in round r
int G;                                 // n/2 games per round

static inline ll dist(int a, int b) { return llabs(X[a] - X[b]) + llabs(Y[a] - Y[b]); }

// objective for a full home assignment homeOf[pos] (a team id, one of the two of that game)
ll objective(const vector<int>& homeOf) {
    ll total = 0;
    for (int tt = 1; tt <= n; tt++) {
        ll travel = 0;
        int prev = tt;                 // start at own home city
        int prevHA = -1; ll breaks = 0;
        for (int r = 0; r < R; r++) {
            int pos = teamSlot[tt][r];
            int venue = homeOf[pos];   // home team's city hosts the game
            travel += dist(prev, venue);
            prev = venue;
            int ha = (venue == tt) ? 1 : 0;
            if (prevHA != -1 && ha == prevHA) breaks++;
            prevHA = ha;
        }
        travel += dist(prev, tt);      // return home at the end
        total += travel + lambda * breaks;
    }
    return total;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    R = inf.readInt();
    lambda = inf.readLong();
    G = n / 2;

    X.assign(n + 1, 0); Y.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) { X[i] = inf.readInt(); Y[i] = inf.readInt(); }

    roundPos.assign(R, {});
    teamSlot.assign(n + 1, vector<int>(R, -1));
    for (int r = 0; r < R; r++) {
        for (int g = 0; g < G; g++) {
            int a = inf.readInt(), b = inf.readInt();
            int pos = (int)gameTeams.size();
            gameTeams.push_back({a, b});
            roundPos[r].push_back(pos);
            teamSlot[a][r] = pos;
            teamSlot[b][r] = pos;
        }
    }
    int P = (int)gameTeams.size();     // total games = R * G

    // ---------- internal baseline B ----------
    // For each unordered pair {i,j} (i<j) meeting in rounds r1<r2, the baseline
    // hosts the game in the EARLIER round at the lower-indexed team, and the
    // later game at the higher-indexed team. Deterministic, feasible.
    // Build pair -> its two game positions.
    map<pair<int,int>, vector<pair<int,int>>> meet;   // {i,j} -> list of (round,pos)
    for (int r = 0; r < R; r++)
        for (int pos : roundPos[r]) {
            int a = gameTeams[pos][0], b = gameTeams[pos][1];
            if (a > b) swap(a, b);
            meet[{a, b}].push_back({r, pos});
        }
    vector<int> baseHome(P, 0);
    for (auto& kv : meet) {
        int i = kv.first.first, j = kv.first.second;
        auto& lst = kv.second;
        if ((int)lst.size() != 2) quitf(_fail, "bad instance: pair (%d,%d) meets %d times", i, j, (int)lst.size());
        sort(lst.begin(), lst.end());
        baseHome[lst[0].second] = i;   // earlier round -> lower team hosts
        baseHome[lst[1].second] = j;   // later round   -> higher team hosts
    }
    ll B = objective(baseHome);
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // ---------- read & validate participant assignment ----------
    // R lines, each G integers; the g-th value on round r's line is the home
    // team of the g-th listed game of round r; must be one of its two teams.
    vector<int> home(P, 0);
    for (int r = 0; r < R; r++)
        for (int g = 0; g < G; g++) {
            int pos = roundPos[r][g];
            int a = gameTeams[pos][0], b = gameTeams[pos][1];
            int h = ouf.readInt(1, n, "home");
            if (h != a && h != b)
                quitf(_wa, "round %d game %d: home %d is not a competitor (%d vs %d)", r + 1, g + 1, h, a, b);
            home[pos] = h;
        }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // pair-balance: each pair must host exactly once at each side.
    for (auto& kv : meet) {
        int i = kv.first.first, j = kv.first.second;
        int ci = 0, cj = 0;
        for (auto& rp : kv.second) {
            if (home[rp.second] == i) ci++;
            else cj++;
        }
        if (ci != 1 || cj != 1)
            quitf(_wa, "pair (%d,%d) venue imbalance: team %d hosts %d, team %d hosts %d (need 1 each)",
                  i, j, i, ci, j, cj);
    }

    ll F = objective(home);
    if (F < 0) quitf(_wa, "negative objective");

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
