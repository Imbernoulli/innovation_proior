// TIER: strong
// Settle-aware teardown with two improvements over greedy:
//   (1) DIRECT PLACEMENT: while tearing down, if the crate can go straight onto
//       its final goal position (that column is already built to the right height),
//       place it there (1 move) instead of parking + rebuilding (2 moves).
//   (2) TRAVEL-AWARE STAGING: park to the NEAREST safe empty position (staging
//       slots, or shelf columns whose goal is empty), shortening gantry rides.
// Same teardown/rebuild skeleton as greedy -> guaranteed feasible & terminating,
// but fewer moves and shorter rides.
#include <bits/stdc++.h>
using namespace std;

int n, p, H;
vector<vector<int>> cur, goal;
vector<int> colOf;

int goalCol[64], goalHt[64];

int main(){
    scanf("%d %d %d", &n, &p, &H);
    vector<vector<int>> start(p);
    goal.assign(p, {});
    for (int c = 0; c < p; c++){ int k; scanf("%d", &k); start[c].resize(k); for (auto& x : start[c]) scanf("%d", &x); }
    for (int c = 0; c < p; c++){ int k; scanf("%d", &k); goal[c].resize(k);  for (auto& x : goal[c])  scanf("%d", &x); }

    for (int c = 0; c < p; c++)
        for (int h = 0; h < (int)goal[c].size(); h++){ goalCol[goal[c][h]] = c; goalHt[goal[c][h]] = h; }

    vector<int> settled(p, 0);
    for (int c = 0; c < n; c++){
        int L = 0, lim = min(start[c].size(), goal[c].size());
        while (L < lim && start[c][L] == goal[c][L]) L++;
        settled[c] = L;
    }

    cur = start;
    colOf.assign(n + 1, -1);
    for (int c = 0; c < p; c++) for (int x : cur[c]) colOf[x] = c;

    auto isSafeEmpty = [&](int col){ return cur[col].empty() && (col >= n || goal[col].empty()); };
    auto nearestSafe = [&](int a){
        int best = -1, bd = INT_MAX;
        for (int e = 0; e < p; e++) if (isSafeEmpty(e)){
            int d = abs(a - e);
            if (d < bd){ bd = d; best = e; }
        }
        return best;
    };

    vector<pair<int,int>> mv;
    auto doMove = [&](int c, int q){
        int a = colOf[c];
        mv.push_back({c, q});
        cur[a].pop_back(); cur[q].push_back(c); colOf[c] = q;
    };

    // teardown with direct placement
    for (int pos = 0; pos < n; pos++){
        while ((int)cur[pos].size() > settled[pos]){
            int c = cur[pos].back();
            int gc = goalCol[c], gh = goalHt[c];
            bool canFinal = (gc < pos) && ((int)cur[gc].size() == gh);
            int q = canFinal ? gc : nearestSafe(pos);
            doMove(c, q);
        }
    }
    // rebuild everything still not in place
    for (int gpos = 0; gpos < n; gpos++){
        while ((int)cur[gpos].size() < (int)goal[gpos].size()){
            int c = goal[gpos][cur[gpos].size()];
            doMove(c, gpos);
        }
    }

    printf("%d\n", (int)mv.size());
    for (auto& m : mv) printf("%d %d\n", m.first, m.second);
    return 0;
}
