#include <bits/stdc++.h>
using namespace std;

// gcd on non-negative long long, with gcd(0,0)=0.
static long long g2(long long x, long long y) {
    while (y) { long long t = x % y; x = y; y = t; }
    return x; // x is non-negative here; gcd(0,0)=0
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<long long> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    // Sparse table over |a[i]| with the gcd operation (idempotent, so overlap is fine).
    int LOG = 1;
    while ((1 << LOG) < max(n, 1)) LOG++;
    LOG++; // headroom
    vector<vector<long long>> sp(LOG, vector<long long>(max(n, 1), 0));
    for (int i = 0; i < n; i++) sp[0][i] = llabs(a[i]); // strip sign: d | a[i] iff d | |a[i]|
    for (int k = 1; k < LOG; k++) {
        int len = 1 << k;
        for (int i = 0; i + len <= n; i++) {
            sp[k][i] = g2(sp[k - 1][i], sp[k - 1][i + (len >> 1)]);
        }
    }

    // For a query [l, r] (1-indexed, inclusive) return gcd(|a_l|,...,|a_r|), with all-zero -> 0.
    string out;
    out.reserve(q * 12);
    for (int Q = 0; Q < q; Q++) {
        int l, r;
        cin >> l >> r;
        l--; r--; // to 0-indexed
        int len = r - l + 1;
        int k = 31 - __builtin_clz(len);
        long long ans = g2(sp[k][l], sp[k][r - (1 << k) + 1]);
        out += to_string(ans);
        out += '\n';
    }
    cout << out;
    return 0;
}
