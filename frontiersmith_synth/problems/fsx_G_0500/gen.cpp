#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// ------------------------------------------------------------------
// League Office: Venue Assignment for a Double Round-Robin
//
// n teams (n even), located at integer cities (Manhattan geometry).
// The OPPONENT timetable is FIXED (a mirrored double round-robin built
// by the circle method): each unordered pair meets in exactly two of the
// R = 2(n-1) rounds. The solver only decides, for each game, which of the
// two teams plays at HOME (venue), subject to the pair-balance rule that
// each pair hosts exactly once each. Two coupled costs are minimized:
//   travel   -- each team starts/ends at home and drives to each away city,
//   breaks   -- consecutive rounds with the same home/away status.
//
// The generator plants geographic clusters (a hidden travel-saving
// structure), an outlier "expansion franchise" (a needle whose venue
// choices are high-leverage), traps where travel-greedy piles up breaks,
// and fills the size envelope with n up to 32.
// ------------------------------------------------------------------

int n, R;
ll lambda;
int X[64], Y[64];

// mirrored double round-robin via the circle method (teams 1..n).
// rounds[r] = list of (a,b) pairings for round r (0-indexed), n/2 games each.
vector<vector<pair<int,int>>> buildSchedule() {
    int m = n - 1;                 // rotating slots
    vector<int> arr(m);
    for (int i = 0; i < m; i++) arr[i] = i + 1;   // teams 1..n-1 ; team n is fixed
    vector<vector<pair<int,int>>> single(m);
    for (int r = 0; r < m; r++) {
        vector<pair<int,int>> games;
        games.push_back({n, arr[r % m]});
        for (int i = 1; i < n / 2; i++) {
            int x = arr[(r + i) % m];
            int y = arr[((r - i) % m + m) % m];
            games.push_back({x, y});
        }
        single[r] = games;
    }
    // double round-robin: append a mirrored copy (each pair now meets twice)
    vector<vector<pair<int,int>>> full;
    for (int r = 0; r < m; r++) full.push_back(single[r]);
    for (int r = 0; r < m; r++) full.push_back(single[r]);
    return full;
}

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);
    if (t < 1) t = 1;
    if (t > 10) t = 10;

    int nn[10]  = {4, 6, 8, 10, 12, 16, 20, 24, 28, 32};
    ll lam[10]  = {2000, 2000, 2500, 2500, 2500, 2500, 3000, 3000, 3000, 3000};
    n = nn[t - 1];
    lambda = lam[t - 1];
    R = 2 * (n - 1);

    // ---- geography: mix of clustered, uniform and an outlier ----
    // Cluster count grows with t; teams snap to a cluster center with jitter.
    // This plants a travel-saving structure: routing away trips within a
    // cluster is cheap, so a good venue pattern can exploit it.
    int nClusters = 1 + t / 2;               // 1..6
    vector<pair<int,int>> centers;
    for (int c = 0; c < nClusters; c++)
        centers.push_back({rnd.next(80, 920), rnd.next(80, 920)});
    int jitter = (t <= 3) ? 400 : 120;       // small tests spread out; big tests tight clusters (TRAP)

    for (int i = 1; i <= n; i++) {
        int c = rnd.next(0, nClusters - 1);
        int x = centers[c].first  + rnd.next(-jitter, jitter);
        int y = centers[c].second + rnd.next(-jitter, jitter);
        X[i] = min(1000, max(0, x));
        Y[i] = min(1000, max(0, y));
    }
    // NEEDLE: on t>=4 make the last team an outlier "expansion franchise"
    // very far from everyone -> its home/away venue choices dominate travel.
    if (t >= 4) {
        X[n] = (rnd.next(0, 1) ? rnd.next(0, 60) : rnd.next(940, 1000));
        Y[n] = (rnd.next(0, 1) ? rnd.next(0, 60) : rnd.next(940, 1000));
    }

    auto sched = buildSchedule();

    // ---- emit ----
    printf("%d %d %lld\n", n, R, lambda);
    for (int i = 1; i <= n; i++) printf("%d %d\n", X[i], Y[i]);
    for (int r = 0; r < R; r++) {
        // shuffle game order within a round (does not change the instance,
        // only the listing order the solver must echo).
        vector<pair<int,int>> g = sched[r];
        for (int i = (int)g.size() - 1; i > 0; i--) {
            int j = rnd.next(0, i);
            swap(g[i], g[j]);
            if (rnd.next(0, 1)) swap(g[i].first, g[i].second);
        }
        for (int i = 0; i < (int)g.size(); i++) {
            printf("%d %d", g[i].first, g[i].second);
            printf(i + 1 < (int)g.size() ? " " : "\n");
        }
    }
    return 0;
}
