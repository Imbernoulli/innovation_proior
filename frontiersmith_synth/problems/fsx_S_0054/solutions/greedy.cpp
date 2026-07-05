// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    vector<long long> C(N), rem(N);
    for (int j = 0; j < N; j++) { scanf("%lld", &C[j]); rem[j] = C[j]; }
    vector<long long> t((size_t)M * N), v((size_t)M * N);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            scanf("%lld %lld", &t[(size_t)i * N + j], &v[(size_t)i * N + j]);

    // Simple per-cluster value-greedy in INPUT order: for each cluster in turn,
    // assign it to the affordable tracer that neutralises the most risk-weight.
    // A one-pass heuristic that beats the cost-/value-blind first-fit baseline
    // but leaves value on the table versus globally ranking (cluster,tracer)
    // pairs by density and repairing.
    vector<int> ans(M, 0);
    for (int i = 0; i < M; i++) {
        int bestj = -1; long long bestV = 0;
        for (int j = 0; j < N; j++) {
            long long tt = t[(size_t)i * N + j], vv = v[(size_t)i * N + j];
            if (rem[j] >= tt && vv > bestV) { bestV = vv; bestj = j; }
        }
        if (bestj >= 0) { rem[bestj] -= t[(size_t)i * N + bestj]; ans[i] = bestj + 1; }
    }

    string out;
    out.reserve((size_t)M * 3);
    for (int i = 0; i < M; i++) {
        out += to_string(ans[i]);
        out += (i + 1 < M ? ' ' : '\n');
    }
    fputs(out.c_str(), stdout);
    return 0;
}
