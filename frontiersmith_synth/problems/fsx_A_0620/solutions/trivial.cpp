// TIER: trivial
// Baseline: place one beacon exactly on every site. Distance 0 always covers
// (0 <= r + t_i + res(c) for any r,t_i,res >= 0), so this is always feasible
// and always uses exactly M beacons -- the checker's own baseline B.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int q, n, r, m, M, Kmax;
    if (scanf("%d %d %d %d %d %d", &q, &n, &r, &m, &M, &Kmax) != 6) return 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) { int x; scanf("%d", &x); }

    vector<vector<int>> site(M, vector<int>(n));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < n; j++) scanf("%d", &site[i][j]);
        int t; scanf("%d", &t);
    }

    printf("%d\n", M);
    string buf;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < n; j++) { buf += to_string(site[i][j]); buf += (j + 1 < n) ? ' ' : '\n'; }
        if (buf.size() > (1u << 20)) { fputs(buf.c_str(), stdout); buf.clear(); }
    }
    fputs(buf.c_str(), stdout);
    return 0;
}
