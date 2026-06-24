#include <bits/stdc++.h>
using namespace std;

int parent_[200005];
long long sz[200005];

int find_(int x) {
    while (parent_[x] != x) {
        parent_[x] = parent_[parent_[x]];
        x = parent_[x];
    }
    return x;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    for (int i = 1; i <= n; i++) {
        parent_[i] = i;
        sz[i] = 1;
    }

    long long pairs = 0;     // current number of connected unordered pairs
    long long grand = 0;     // running sum of "pairs" reported after each cable

    string out;
    out.reserve((size_t)m * 8);

    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        int ru = find_(u), rv = find_(v);
        if (ru != rv) {
            // merging two distinct components adds sz[ru]*sz[rv] new connected pairs
            pairs += sz[ru] * sz[rv];
            // union by size
            if (sz[ru] < sz[rv]) swap(ru, rv);
            parent_[rv] = ru;
            sz[ru] += sz[rv];
        }
        // if ru == rv, the cable is redundant; pairs unchanged
        grand += pairs;
        out += to_string(pairs);
        out += '\n';
    }

    out += to_string(grand);
    out += '\n';
    cout << out;
    return 0;
}
