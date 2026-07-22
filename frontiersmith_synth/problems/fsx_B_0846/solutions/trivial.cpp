// TIER: trivial
// Picks exactly the zero-chemistry-edge (isolated) candidates. Always safe
// (no synergy possible, no crowding possible) -- this is exactly the
// checker's internal baseline B, so this reference reproduces ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    vector<long long> v(N), cap(N), pen(N);
    for (int i = 0; i < N; i++) scanf("%lld", &v[i]);
    for (int i = 0; i < N; i++) scanf("%lld", &cap[i]);
    for (int i = 0; i < N; i++) scanf("%lld", &pen[i]);
    vector<int> deg(N, 0);
    for (int e = 0; e < M; e++) {
        int a, b, s;
        scanf("%d %d %d", &a, &b, &s);
        deg[a]++; deg[b]++;
    }
    vector<int> pick;
    for (int i = 0; i < N; i++) if (deg[i] == 0) pick.push_back(i);
    printf("%d\n", (int)pick.size());
    for (size_t i = 0; i < pick.size(); i++) printf("%d%c", pick[i], (i + 1 == pick.size()) ? '\n' : ' ');
    if (pick.empty()) printf("\n");
    return 0;
}
