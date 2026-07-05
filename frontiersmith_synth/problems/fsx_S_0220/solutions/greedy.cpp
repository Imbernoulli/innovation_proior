// TIER: greedy
// Nearest-feasible-node construction serving ALL relays (no skipping, no local search).
// Start at the depot; repeatedly drive to the nearest not-yet-visited node that is
// currently allowed: any unvisited pickup, or any unvisited delivery whose pickup has
// already been visited. Continue until all 2n sites are visited, then return to depot.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    int N = 2 * n;
    vector<long long> X(N + 1), Y(N + 1), pen(n + 1);
    scanf("%lld %lld", &X[0], &Y[0]);
    for (int i = 1; i <= n; i++) {
        scanf("%lld %lld %lld %lld %lld", &X[i], &Y[i], &X[n + i], &Y[n + i], &pen[i]);
    }
    auto D = [&](int a, int b) -> long long {
        long long dx = X[a] - X[b], dy = Y[a] - Y[b];
        return (long long)llround(sqrt((double)(dx * dx + dy * dy)));
    };

    vector<char> visited(N + 1, 0);
    vector<int> order;
    order.reserve(N);
    int cur = 0;
    for (int step = 0; step < N; step++) {
        int best = -1; long long bestD = LLONG_MAX;
        for (int v = 1; v <= N; v++) {
            if (visited[v]) continue;
            bool allowed;
            if (v <= n) allowed = true;                 // pickup always allowed
            else allowed = visited[v - n];              // delivery: pickup visited?
            if (!allowed) continue;
            long long d = D(cur, v);
            if (d < bestD) { bestD = d; best = v; }
        }
        // best always exists (there is always an allowed pickup while any remain)
        visited[best] = 1;
        order.push_back(best);
        cur = best;
    }

    printf("%d\n", (int)order.size());
    string line;
    for (size_t r = 0; r < order.size(); r++) {
        if (r) line += ' ';
        line += to_string(order[r]);
    }
    printf("%s\n", line.c_str());
    return 0;
}
