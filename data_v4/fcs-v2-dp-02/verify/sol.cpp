#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int B, n, q;
    if (!(cin >> B >> n >> q)) return 0;

    // f[mask] starts as the multiplicity of each item mask, then becomes
    // the number of items whose mask is a SUPERSET of `mask` (item & mask == mask).
    const int SZ = 1 << B;
    vector<long long> f(SZ, 0);

    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        f[x] += 1;
    }

    // Sum-Over-Subsets (zeta transform) in the SUPERSET direction.
    // For each bit b, fold the value of the state with bit b set DOWN into the
    // state with bit b cleared. After processing every bit, f[mask] equals the
    // sum of the original f over all masks t with (t & mask) == mask, i.e. all
    // supersets of `mask`. O(B * 2^B).
    for (int b = 0; b < B; b++) {
        int bit = 1 << b;
        for (int mask = 0; mask < SZ; mask++) {
            if ((mask & bit) == 0) {
                f[mask] += f[mask | bit];
            }
        }
    }

    string out;
    out.reserve((size_t)q * 7);
    for (int i = 0; i < q; i++) {
        int m;
        cin >> m;
        out += to_string(f[m]);
        out += '\n';
    }
    cout << out;
    return 0;
}
