// TIER: trivial
// Fully-serial plan: run every stage one after another on a single global timeline.
// Makespan equals total work == the checker's baseline B, so this scores exactly 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int J, M;
    if (scanf("%d %d", &J, &M) != 2) return 0;
    vector<vector<long long>> start(J);
    long long clock = 0;
    for (int i = 0; i < J; i++) {
        int L; scanf("%d", &L);
        start[i].resize(L);
        for (int k = 0; k < L; k++) {
            int m, d; scanf("%d %d", &m, &d);
            start[i][k] = clock;
            clock += d;
        }
    }
    for (int i = 0; i < J; i++) {
        for (size_t k = 0; k < start[i].size(); k++) {
            printf("%lld%c", start[i][k], k + 1 == start[i].size() ? '\n' : ' ');
        }
    }
    return 0;
}
