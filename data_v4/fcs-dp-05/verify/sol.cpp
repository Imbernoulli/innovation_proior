#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    vector<vector<int>> adj(n + 1);
    for (int e = 0; e < n - 1; e++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    if (n == 1) {
        // single vertex: exactly one connected subset containing it (itself)
        cout << 1 << "\n";
        return 0;
    }

    // down[v] = number of connected subsets inside the subtree of v (rooted at 1)
    //           that contain v = prod over children c of (1 + down[c]).
    vector<long long> down(n + 1, 1);
    vector<int> par(n + 1, 0);
    vector<int> order;
    order.reserve(n);

    // iterative DFS to get a parent-before-child order rooted at 1
    {
        vector<char> vis(n + 1, 0);
        vector<int> st;
        st.push_back(1);
        vis[1] = 1;
        par[1] = 0;
        while (!st.empty()) {
            int u = st.back();
            st.pop_back();
            order.push_back(u);
            for (int w : adj[u]) {
                if (!vis[w]) {
                    vis[w] = 1;
                    par[w] = u;
                    st.push_back(w);
                }
            }
        }
    }

    // compute down[] in reverse order (children before parents)
    for (int i = (int)order.size() - 1; i >= 0; i--) {
        int u = order[i];
        long long prod = 1;
        for (int w : adj[u]) {
            if (w == par[u]) continue;
            prod = prod * ((1 + down[w]) % MOD) % MOD;
        }
        down[u] = prod;
    }

    // ans[v] = number of connected subsets of the WHOLE tree containing v.
    // Root the answer at 1: ans[1] = down[1].
    // Rerooting push: for each u (parent-before-child), we know ans[u]; we want
    // to give each child c its "up[c]" = number of connected subsets containing u
    // in the part of the tree OUTSIDE c's subtree (i.e. through the edge c-u),
    // which equals the product over all of u's neighbors EXCEPT c of their factors,
    // where a neighbor w's factor is (1 + g) with g = down[w] if w is a child of u,
    // or g = up[u] if w is u's parent.
    //
    // To get "product over all neighbors except c" without modular division, build
    // prefix/suffix products over u's neighbor list of the factor (1 + g_w).
    vector<long long> up(n + 1, 0);  // up[1] is unused (root has no up part)

    for (int idx = 0; idx < (int)order.size(); idx++) {
        int u = order[idx];
        int deg = (int)adj[u].size();
        // factor for each neighbor position
        vector<long long> fac(deg);
        for (int k = 0; k < deg; k++) {
            int w = adj[u][k];
            long long g;
            if (w == par[u]) g = up[u];   // the "up" side of u
            else g = down[w];             // a child's down value
            fac[k] = (1 + g) % MOD;
        }
        // prefix[k] = product of fac[0..k-1], suffix[k] = product of fac[k+1..deg-1]
        vector<long long> pref(deg + 1, 1), suf(deg + 1, 1);
        for (int k = 0; k < deg; k++) pref[k + 1] = pref[k] * fac[k] % MOD;
        for (int k = deg - 1; k >= 0; k--) suf[k] = suf[k + 1] * fac[k] % MOD;
        for (int k = 0; k < deg; k++) {
            int c = adj[u][k];
            if (c == par[u]) continue;    // only push to children
            // product over all of u's neighbors except c:
            long long without_c = pref[k] * suf[k + 1] % MOD;
            up[c] = without_c;            // connected subsets above c containing u
        }
    }

    // ans[v] = down[v] * (1 + up[v]); for the root, up is empty so (1 + 0) is wrong
    // -- handle root separately as ans[1] = down[1].
    vector<long long> ans(n + 1, 0);
    for (int v = 1; v <= n; v++) {
        if (v == 1) ans[v] = down[1] % MOD;
        else ans[v] = down[v] * ((1 + up[v]) % MOD) % MOD;
    }

    for (int v = 1; v <= n; v++) {
        cout << ans[v] % MOD;
        if (v < n) cout << ' ';
    }
    cout << "\n";
    return 0;
}
