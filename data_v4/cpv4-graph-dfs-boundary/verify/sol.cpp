#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    if (n <= 0) { printf("0\n"); return 0; }

    vector<int> par(n), p(n);
    vector<vector<int>> ch(n);
    int root = -1;
    for (int i = 0; i < n; i++) {
        int pa;
        if (scanf("%d %d", &pa, &p[i]) != 2) return 0; // parent (-1 for root) and power
        par[i] = pa;
        if (pa == -1) root = i;
        else ch[pa].push_back(i);
    }

    // A station u covers a STRICT descendant v iff depth(v) - depth(u) <= p[u].
    // Carry along the DFS, for each node v, the value
    //   reach = max over PROPER ancestors u of ( depth(u) + p[u] ).
    // Then v is covered  <=>  depth(v) <= reach   (inclusive boundary, excludes self).
    long long covered = 0;
    const long long NEG = LLONG_MIN / 4; // "no ancestor reaches here yet"

    struct Frame { int node; int depth; long long reach; };
    vector<Frame> st;
    st.reserve(n);
    st.push_back({root, 0, NEG}); // root has no proper ancestor

    while (!st.empty()) {
        Frame f = st.back();
        st.pop_back();
        int u = f.node;
        int d = f.depth;
        long long reach = f.reach;

        if ((long long)d <= reach) covered++; // some proper ancestor reaches depth d

        long long reachIncludingU = max(reach, (long long)d + p[u]);
        for (int w : ch[u]) {
            st.push_back({w, d + 1, reachIncludingU});
        }
    }

    printf("%lld\n", covered);
    return 0;
}
