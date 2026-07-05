// TIER: trivial
// Reproduces the grader's naive teardown EXACTLY -> F = B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int n, p, H;
    scanf("%d %d %d", &n, &p, &H);
    vector<vector<int>> start(p), goal(p);
    for (int c = 0; c < p; c++){ int k; scanf("%d", &k); start[c].resize(k); for (auto& x : start[c]) scanf("%d", &x); }
    for (int c = 0; c < p; c++){ int k; scanf("%d", &k); goal[c].resize(k);  for (auto& x : goal[c])  scanf("%d", &x); }

    vector<vector<int>> cur = start;
    vector<int> colOf(n + 1, -1);
    for (int c = 0; c < p; c++) for (int x : cur[c]) colOf[x] = c;
    auto park = [&](int c){ return p - n + (c - 1); };

    vector<pair<int,int>> mv;
    for (int pos = 0; pos < n; pos++){
        while (!cur[pos].empty()){
            int c = cur[pos].back(), q = park(c);
            mv.push_back({c, q});
            cur[pos].pop_back(); cur[q].push_back(c); colOf[c] = q;
        }
    }
    for (int gpos = 0; gpos < n; gpos++){
        for (int c : goal[gpos]){
            int a = colOf[c];
            mv.push_back({c, gpos});
            cur[a].pop_back(); cur[gpos].push_back(c); colOf[c] = gpos;
        }
    }

    printf("%d\n", (int)mv.size());
    for (auto& m : mv) printf("%d %d\n", m.first, m.second);
    return 0;
}
