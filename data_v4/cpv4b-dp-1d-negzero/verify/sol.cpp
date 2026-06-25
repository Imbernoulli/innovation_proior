#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // f0 = best sum of a contiguous non-empty window ending here, no skip used.
    // f1 = best sum ending here with exactly one interior stop skipped (>=1 kept).
    const long long NEG = LLONG_MIN / 4;
    long long f0 = a[0];          // window {0}, nothing skipped
    long long f1 = NEG;           // cannot skip anything yet (need a kept element)
    long long ans = f0;           // best so far must come from a non-empty pick

    for (int i = 1; i < n; i++) {
        long long nf1 = max(f0, f1 + a[i]); // skip i now (extend f0) OR keep i atop f1
        long long nf0 = max(a[i], f0 + a[i]); // start fresh at i, or extend f0
        f0 = nf0;
        f1 = nf1;
        ans = max(ans, max(f0, f1));
    }

    cout << ans << "\n";
    return 0;
}
