#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;            // empty input -> no stations
    if (n == 0) { cout << 0 << "\n"; return 0; } // no run possible -> 0

    vector<long long> v(n + 1);
    for (int i = 1; i <= n; i++) cin >> v[i];

    // parent[i] for i = 2..n; station 1 is the source (root).
    vector<vector<int>> children(n + 1);
    for (int i = 2; i <= n; i++) {
        int p; cin >> p;
        children[p].push_back(i);
    }

    // down[i] = best score of a downstream segment that STARTS at i and goes down.
    // It always contains i itself, then optionally continues into the best child chain
    // (only if that continuation is positive).
    // ans = max over all i of down[i].
    //
    // Iterative post-order DFS to avoid recursion depth blowup on a long chain.
    vector<long long> down(n + 1, LLONG_MIN);
    long long ans = LLONG_MIN;            // a real run must contain >= 1 station

    vector<int> order;
    order.reserve(n);
    vector<int> stk;
    stk.push_back(1);
    while (!stk.empty()) {
        int u = stk.back(); stk.pop_back();
        order.push_back(u);
        for (int c : children[u]) stk.push_back(c);
    }
    // process in reverse pre-order = a valid post-order (children before parent)
    for (int idx = (int)order.size() - 1; idx >= 0; idx--) {
        int u = order[idx];
        long long best = v[u];            // the segment {u} alone
        for (int c : children[u]) {
            // extend downstream into child c only if it adds value
            if (down[c] > 0) best = max(best, v[u] + down[c]);
        }
        down[u] = best;
        ans = max(ans, best);
    }

    cout << ans << "\n";
    return 0;
}
