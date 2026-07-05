// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    vector<long long> C(N);
    for (int j = 0; j < N; j++) scanf("%lld", &C[j]);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) { long long a, b; scanf("%lld %lld", &a, &b); }
    // deliberately infeasible: assign EVERY cluster to tracer 1, which will
    // blow past its contact-hour budget (and index N+1 would also be out of range).
    string out;
    out.reserve((size_t)M * 3);
    for (int i = 0; i < M; i++) {
        out += to_string(N + 1);   // out-of-range tracer index -> infeasible
        out += (i + 1 < M ? ' ' : '\n');
    }
    fputs(out.c_str(), stdout);
    return 0;
}
