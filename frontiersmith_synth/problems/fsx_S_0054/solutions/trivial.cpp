// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    vector<long long> C(N), rem(N);
    for (int j = 0; j < N; j++) { scanf("%lld", &C[j]); rem[j] = C[j]; }
    vector<int> ans(M, 0);
    for (int i = 0; i < M; i++) {
        int chosen = 0;
        for (int j = 0; j < N; j++) {
            long long t, v;
            scanf("%lld %lld", &t, &v);
            if (chosen == 0 && rem[j] >= t) {
                rem[j] -= t;
                chosen = j + 1;
            }
        }
        ans[i] = chosen;
    }
    // exactly the checker's first-fit baseline
    string out;
    out.reserve((size_t)M * 3);
    for (int i = 0; i < M; i++) {
        out += to_string(ans[i]);
        out += (i + 1 < M ? ' ' : '\n');
    }
    fputs(out.c_str(), stdout);
    return 0;
}
