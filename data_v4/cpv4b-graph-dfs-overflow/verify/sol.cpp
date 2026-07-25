#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;            // empty input -> nothing to report
    vector<vector<int>> children(n + 1);  // 1-indexed; node 1 is the root
    for (int v = 2; v <= n; v++) {
        int p;
        cin >> p;                         // parent of node v (1 <= p < v guaranteed)
        children[p].push_back(v);
    }

    // Iterative DFS from the root (node 1) to avoid recursion-depth blowups on a
    // long chain. We need, for every node, its subtree size and its depth.
    // depth(root) = 0; depth(child) = depth(parent) + 1.
    // The reported total is  sum over all nodes v of  size(v) * depth(v).
    vector<long long> subtreeSize(n + 1, 1); // each node counts itself
    vector<int> depth(n + 1, 0);

    // Post-order via an explicit stack: push with a "processed" flag so we
    // accumulate child subtree sizes into the parent after the children are done.
    vector<pair<int, bool>> stk;
    if (n >= 1) stk.push_back({1, false});
    while (!stk.empty()) {
        auto [v, processed] = stk.back();
        stk.pop_back();
        if (!processed) {
            stk.push_back({v, true});
            for (int c : children[v]) {
                depth[c] = depth[v] + 1;
                stk.push_back({c, false});
            }
        } else {
            for (int c : children[v]) subtreeSize[v] += subtreeSize[c];
        }
    }

    long long total = 0;
    for (int v = 1; v <= n; v++) {
        total += subtreeSize[v] * (long long)depth[v];
    }

    cout << total << "\n";
    return 0;
}
