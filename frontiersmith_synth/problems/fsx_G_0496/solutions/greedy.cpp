// TIER: greedy
// Settle-aware teardown: leave the already-correct bottom prefixes in place, tear
// down only the misplaced crates (to fixed far staging slots), then rebuild.
// Travel-naive; ~2*U moves where U = # misplaced crates.  Beats trivial when some
// crates are already settled.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int n, p, H;
    scanf("%d %d %d", &n, &p, &H);
    vector<vector<int>> start(p), goal(p);
    for (int c = 0; c < p; c++){ int k; scanf("%d", &k); start[c].resize(k); for (auto& x : start[c]) scanf("%d", &x); }
    for (int c = 0; c < p; c++){ int k; scanf("%d", &k); goal[c].resize(k);  for (auto& x : goal[c])  scanf("%d", &x); }

    // longest common bottom prefix per shelf column = settled height
    vector<int> settled(p, 0);
    for (int c = 0; c < n; c++){
        int L = 0, lim = min(start[c].size(), goal[c].size());
        while (L < lim && start[c][L] == goal[c][L]) L++;
        settled[c] = L;
    }

    vector<vector<int>> cur = start;
    vector<int> colOf(n + 1, -1);
    for (int c = 0; c < p; c++) for (int x : cur[c]) colOf[x] = c;
    auto park = [&](int c){ return p - n + (c - 1); };

    vector<pair<int,int>> mv;
    // disassemble misplaced crates to fixed far slots
    for (int pos = 0; pos < n; pos++){
        while ((int)cur[pos].size() > settled[pos]){
            int c = cur[pos].back(), q = park(c);
            mv.push_back({c, q});
            cur[pos].pop_back(); cur[q].push_back(c); colOf[c] = q;
        }
    }
    // rebuild each goal column above its settled prefix
    for (int gpos = 0; gpos < n; gpos++){
        for (int h = settled[gpos]; h < (int)goal[gpos].size(); h++){
            int c = goal[gpos][h], a = colOf[c];
            mv.push_back({c, gpos});
            cur[a].pop_back(); cur[gpos].push_back(c); colOf[c] = gpos;
        }
    }

    printf("%d\n", (int)mv.size());
    for (auto& m : mv) printf("%d %d\n", m.first, m.second);
    return 0;
}
