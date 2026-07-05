// TIER: strong
#include <bits/stdc++.h>
using namespace std;

int N, M;
vector<long long> t, v, rem;
static inline long long T(int i, int j) { return t[(size_t)i * N + j]; }
static inline long long V(int i, int j) { return v[(size_t)i * N + j]; }

int main() {
    if (scanf("%d %d", &N, &M) != 2) return 0;
    vector<long long> C(N);
    rem.assign(N, 0);
    for (int j = 0; j < N; j++) { scanf("%lld", &C[j]); rem[j] = C[j]; }
    t.assign((size_t)M * N, 0);
    v.assign((size_t)M * N, 0);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            scanf("%lld %lld", &t[(size_t)i * N + j], &v[(size_t)i * N + j]);

    vector<int> ans(M, 0);

    // ---- phase 1: value/cost density greedy ----
    {
        struct P { double d; int i; int j; };
        vector<P> pairs;
        pairs.reserve((size_t)M * N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                pairs.push_back({(double)V(i, j) / (double)T(i, j), i, j});
        sort(pairs.begin(), pairs.end(),
             [](const P& a, const P& b) { return a.d > b.d; });
        for (const auto& p : pairs) {
            if (ans[p.i] != 0) continue;
            if (rem[p.j] >= T(p.i, p.j)) { rem[p.j] -= T(p.i, p.j); ans[p.i] = p.j + 1; }
        }
    }

    // ---- phase 2: local search rounds ----
    const int ROUNDS = 6;
    for (int r = 0; r < ROUNDS; r++) {
        // (a) MOVE assigned clusters to a cheaper tracer with no value loss -> frees budget
        for (int i = 0; i < M; i++) {
            if (ans[i] == 0) continue;
            int j = ans[i] - 1;
            int bestj = j; long long bestCost = T(i, j);
            for (int jj = 0; jj < N; jj++) {
                if (jj == j) continue;
                if (V(i, jj) >= V(i, j) && T(i, jj) < bestCost && rem[jj] >= T(i, jj)) {
                    bestj = jj; bestCost = T(i, jj);
                }
            }
            if (bestj != j) {
                rem[j] += T(i, j);
                rem[bestj] -= T(i, bestj);
                ans[i] = bestj + 1;
            }
        }

        // (b) FILL uncovered clusters into their best affordable tracer (max value)
        for (int i = 0; i < M; i++) {
            if (ans[i] != 0) continue;
            int bestj = -1; long long bestV = 0;
            for (int jj = 0; jj < N; jj++) {
                if (rem[jj] >= T(i, jj) && V(i, jj) > bestV) { bestV = V(i, jj); bestj = jj; }
            }
            if (bestj >= 0) { rem[bestj] -= T(i, bestj); ans[i] = bestj + 1; }
        }

        // (c) EVICTION: an uncovered cluster displaces the lowest-value assigned
        //     cluster on some tracer when the swap yields a net risk-weight gain.
        vector<int> minIdx(N, -1);
        vector<long long> minVal(N, LLONG_MAX);
        for (int i = 0; i < M; i++) {
            if (ans[i] == 0) continue;
            int j = ans[i] - 1;
            if (V(i, j) < minVal[j]) { minVal[j] = V(i, j); minIdx[j] = i; }
        }
        bool any = false;
        for (int i = 0; i < M; i++) {
            if (ans[i] != 0) continue;
            long long bestGain = 0; int bestj = -1;
            for (int jj = 0; jj < N; jj++) {
                int k = minIdx[jj];
                if (k < 0) continue;
                long long room = rem[jj] + T(k, jj);
                if (room >= T(i, jj)) {
                    long long gain = V(i, jj) - V(k, jj);
                    if (gain > bestGain) { bestGain = gain; bestj = jj; }
                }
            }
            if (bestj >= 0) {
                int k = minIdx[bestj];
                rem[bestj] += T(k, bestj); ans[k] = 0;      // evict k
                rem[bestj] -= T(i, bestj); ans[i] = bestj + 1;
                minIdx[bestj] = -1;                          // one eviction per tracer per pass
                any = true;
            }
        }
        if (!any && r > 0) { /* keep iterating a bit for move/fill effects */ }
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
