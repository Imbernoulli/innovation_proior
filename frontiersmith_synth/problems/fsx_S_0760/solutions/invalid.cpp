// TIER: invalid
// Deliberately infeasible: for the very first convoy it emits an edge id that is out of
// range (M+5), which the checker's bounded read must reject. Must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M, K;
    if (scanf("%d %d %d", &N, &M, &K) != 3) return 0;
    for (int i = 1; i <= M; i++) {
        int u, v, L, C, r, f;
        scanf("%d %d %d %d %d %d", &u, &v, &L, &C, &r, &f);
    }
    vector<int> co(K + 1), cd(K + 1), cready(K + 1);
    for (int i = 1; i <= K; i++) scanf("%d %d %d", &co[i], &cd[i], &cready[i]);

    for (int i = 1; i <= K; i++) {
        if (i == 1) {
            printf("%d 1 %d\n", cready[i], M + 5); // out-of-range edge id
        } else {
            printf("%d 1 1\n", cready[i]); // may also be nonsense, doesn't matter
        }
    }
    return 0;
}
