#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    // prefix[i] = a[1] + a[2] + ... + a[i], prefix[0] = 0.
    // Values reach n * max|a| = 1e5 * 1e9 = 1e14, far beyond 32-bit range,
    // so prefix sums MUST be 64-bit.
    vector<long long> prefix(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long x;
        cin >> x;
        prefix[i] = prefix[i - 1] + x;
    }

    // Sum over each query window [l, r] is prefix[r] - prefix[l-1].
    // The grand total over up to 5e4 windows of magnitude up to 1e14
    // reaches ~5e18, which still fits in long long but overflows int many times over.
    long long total = 0;
    for (int k = 0; k < q; k++) {
        int l, r;
        cin >> l >> r;
        total += prefix[r] - prefix[l - 1];
    }

    cout << total << "\n";
    return 0;
}
