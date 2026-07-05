// TIER: trivial
// Reproduces the checker's internal baseline exactly: regular bounding-box grid
// tiling of module type 1, capped by its availability -> F == B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int H, W, T;
    if (scanf("%d %d %d", &H, &W, &T) != 3) return 0;
    vector<int> k(T + 1), c(T + 1);
    vector<vector<pair<int,int>>> sh(T + 1);
    for (int t = 1; t <= T; t++) {
        scanf("%d %d", &k[t], &c[t]);
        for (int i = 0; i < k[t]; i++) { int r, cc; scanf("%d %d", &r, &cc); sh[t].push_back({r, cc}); }
    }
    int bh = 0, bw = 0;
    for (auto& p : sh[1]) { bh = max(bh, p.first); bw = max(bw, p.second); }
    bh++; bw++;
    int rows = H / bh, cols = W / bw;
    long long slots = (long long)rows * cols;
    long long copies = min((long long)c[1], slots);
    vector<array<int,4>> out;
    long long cnt = 0;
    for (int i = 0; i < rows && cnt < copies; i++)
        for (int j = 0; j < cols && cnt < copies; j++) { out.push_back({1, 0, i * bh, j * bw}); cnt++; }
    printf("%lld\n", (long long)out.size());
    for (auto& o : out) printf("%d %d %d %d\n", o[0], o[1], o[2], o[3]);
    return 0;
}
