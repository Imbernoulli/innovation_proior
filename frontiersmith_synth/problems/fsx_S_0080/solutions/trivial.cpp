// TIER: trivial
// Reproduce the deterministic hash-scrambled reference balanced partition R.
// Its cut equals the checker's baseline B, so this scores ratio ~ 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long u64;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int e = 0; e < m; e++) { int u, v, w; if (scanf("%d %d %d", &u, &v, &w) != 3) return 0; }

    vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i + 1;
    const u64 MUL = 11400714819323198485ULL; // 0x9E3779B97F4A7C15
    sort(idx.begin(), idx.end(), [&](int a, int b) {
        u64 ka = (u64)a * MUL, kb = (u64)b * MUL;
        if (ka != kb) return ka < kb;
        return a < b;
    });
    vector<int> x(n + 1, 0);
    int half = n / 2;
    for (int r = 0; r < n; r++) x[idx[r]] = (r < half) ? 0 : 1;

    for (int i = 1; i <= n; i++) printf("%d%c", x[i], i == n ? '\n' : ' ');
    return 0;
}
