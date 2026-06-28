#include <bits/stdc++.h>
using namespace std;

int main() {
    long long k;
    if (!(cin >> k)) return 0;                 // missing input -> nothing to do

    const int NMAX = 1000;                      // vertex budget

    // C2[c] = c*(c-1)/2 = number of edges in a clique of size c (also the number of
    // triangles a new vertex creates when joined to a clique of size c). Precompute up
    // to NMAX so we can pick the largest clique we can afford in O(1) by binary search.
    vector<long long> C2(NMAX + 1);
    for (int c = 0; c <= NMAX; c++) C2[c] = 1LL * c * (c - 1) / 2;

    // Greedy "clique-stacking". Add vertices 0,1,2,... one at a time. Vertex i is joined
    // to the first c_i vertices {0,...,c_i-1}, which we maintain as a clique, so it adds
    // exactly C2[c_i] new triangles. We pick c_i as large as possible without overshooting
    // the remaining budget r (and never more than i, the number of earlier vertices).
    long long r = k;
    vector<pair<int,int>> edges;                // the edge list we emit
    int n = 0;                                  // vertices used so far
    while (r > 0) {
        int i = n;                              // index of the vertex we are about to add
        // largest c with C2[c] <= r, capped at i. Binary search over [0, min(i, NMAX)].
        int lo = 0, hi = min(i, NMAX), c = 0;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (C2[mid] <= r) { c = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        for (int v = 0; v < c; v++) edges.emplace_back(v, i);
        r -= C2[c];
        n = i + 1;
        if (n > NMAX) {                         // ran out of vertices: declare impossible
            cout << -1 << "\n";
            return 0;
        }
    }

    // k == 0 leaves n == 0; emit a single isolated vertex so the output is a valid graph.
    if (n == 0) n = 1;

    cout << n << " " << edges.size() << "\n";
    for (auto &e : edges) cout << (e.first + 1) << " " << (e.second + 1) << "\n";
    return 0;
}
