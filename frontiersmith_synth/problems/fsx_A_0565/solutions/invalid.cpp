// TIER: invalid
// Deliberately infeasible: it gives the LEFTMOST job to crane K and every other job to
// crane 1.  Sorting by position, the crane indices start at K then drop to 1 -- a crossing
// -- so the monotone (non-crossing) rule is violated and the checker must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int K, M; long long S, alpha, gamma;
    if (scanf("%d %d %lld %lld %lld", &K, &M, &S, &alpha, &gamma) != 5) return 0;
    vector<long long> P(K);
    for (int c = 0; c < K; c++) scanf("%lld", &P[c]);
    vector<long long> pos(M), work(M);
    for (int i = 0; i < M; i++) scanf("%lld %lld", &pos[i], &work[i]);

    int lm = 0;
    for (int i = 1; i < M; i++) if (pos[i] < pos[lm]) lm = i;

    for (int i = 0; i < M; i++) {
        int a = (i == lm) ? K : 1;
        printf("%d%c", a, i + 1 < M ? ' ' : '\n');
    }
    for (int c = 0; c < K; c++) printf("%d%c", 0, c + 1 < K ? ' ' : '\n');
    return 0;
}
