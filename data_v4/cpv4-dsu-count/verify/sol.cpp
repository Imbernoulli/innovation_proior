#include <bits/stdc++.h>
using namespace std;

int par[200005], sz[200005];

int find(int x) {
    while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
    return x;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    for (int i = 1; i <= n; i++) { par[i] = i; sz[i] = 1; }

    long long redundant = 0;        // cables whose endpoints were already connected
    long long samePairs = 0;        // running number of unordered same-component pairs
    long long prefixRedundantSum = 0; // sum over all queries' answers (running total of redundant so far)

    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        int ru = find(u), rv = find(v);
        if (ru == rv) {
            // both endpoints already connected: this cable is redundant, adds 0 new pairs
            redundant++;
        } else {
            // merging two distinct components of sizes sz[ru], sz[rv]
            long long s1 = sz[ru], s2 = sz[rv];
            samePairs += s1 * s2;   // exactly s1*s2 new unordered cross pairs become same-component
            // union by size
            if (sz[ru] < sz[rv]) swap(ru, rv);
            par[rv] = ru;
            sz[ru] += sz[rv];
        }
        prefixRedundantSum += redundant; // after processing cable e, accumulate current redundant count
    }

    cout << redundant << " " << samePairs << " " << prefixRedundantSum << "\n";
    return 0;
}
