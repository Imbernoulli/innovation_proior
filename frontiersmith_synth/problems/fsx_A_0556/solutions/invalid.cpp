// TIER: invalid
// Deliberately infeasible: declares each router's own id as its next hop, which is
// never a neighbor (no self-loops) -> the checker rejects it and scores 0.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int n, m;
    scanf("%d %d", &n, &m);
    for (int i = 0; i < m; i++){ int u, v; scanf("%d %d", &u, &v); (void)u; (void)v; }
    string out; out.reserve((size_t)n * n * 4);
    char buf[32];
    for (int d = 1; d <= n; d++)
        for (int u = 1; u <= n; u++){
            if (u == d) continue;
            int len = sprintf(buf, "%d %d\n", u, u);   // self = not a neighbor
            out.append(buf, len);
        }
    fputs(out.c_str(), stdout);
    return 0;
}
