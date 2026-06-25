#include <bits/stdc++.h>
using namespace std;

const long long MOD = 998244353;

int main() {
    int n, L, R, m;
    if (!(cin >> n >> L >> R >> m)) return 0;

    // feud[i] = bitmask of adventurers that i refuses to share a team with.
    vector<int> feud(n, 0);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;             // 0-indexed
        feud[u] |= (1 << v);
        feud[v] |= (1 << u);
    }

    int full = (1 << n) - 1;

    // valid[S] = 1 iff subset S is a legal team:
    //   size in [L,R] and contains no feuding pair.
    vector<char> valid(1 << n, 0);
    for (int S = 1; S <= full; S++) {
        int sz = __builtin_popcount(S);
        if (sz < L || sz > R) { valid[S] = 0; continue; }
        bool ok = true;
        int t = S;
        while (t) {
            int i = __builtin_ctz(t);
            t &= t - 1;
            if (feud[i] & S) { ok = false; break; }   // i feuds with someone also in S
        }
        valid[S] = ok ? 1 : 0;
    }

    // f[mask] = number of partitions of the set "mask" into legal teams (unlabeled).
    vector<long long> f(1 << n, 0);
    f[0] = 1;
    for (int mask = 1; mask <= full; mask++) {
        int low = mask & (-mask);          // lowest set bit, the anchor element
        long long acc = 0;
        // Enumerate submasks S of mask that CONTAIN the anchor bit "low".
        // This forces each partition to be counted exactly once: the team
        // holding the lowest-indexed element of "mask" is chosen first.
        int rest = mask ^ low;             // bits we may freely add to the anchor team
        for (int sub = rest; ; sub = (sub - 1) & rest) {
            int S = sub | low;             // team containing the anchor
            if (valid[S]) {
                acc += f[mask ^ S];
                if (acc >= MOD) acc -= MOD;
            }
            if (sub == 0) break;
        }
        f[mask] = acc;
    }

    cout << f[full] % MOD << "\n";
    return 0;
}
