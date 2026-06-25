#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;              // n = 0 (or empty input) -> answer 0

    // prefix[i] = a[0] + ... + a[i-1], so prefix[0] = 0 and a block (j, i] has
    // sum prefix[i] - prefix[j]; it is "profitable" iff prefix[i] > prefix[j].
    vector<long long> prefix(n + 1);
    prefix[0] = 0;
    for (int i = 1; i <= n; i++) {
        long long x; cin >> x;
        prefix[i] = prefix[i - 1] + x;
    }

    // Coordinate-compress the n+1 prefix values.
    vector<long long> vals(prefix.begin(), prefix.end());
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int m = (int)vals.size();
    auto cid = [&](long long v) {
        return int(lower_bound(vals.begin(), vals.end(), v) - vals.begin());
    };

    const int NEG = INT_MIN / 4;

    // dp[i] = max profitable blocks in a full partition of prefix[0..i].
    // dp[i] = max( 1 + max_{j<i, prefix[j] <  prefix[i]} dp[j],     // block (j,i] profitable
    //                  max_{j<i, prefix[j] >= prefix[i]} dp[j] ).   // block (j,i] not profitable
    // Two Fenwick trees over the compressed prefix coordinate hold prefix-max of dp:
    //   bitLess : indexed by coordinate, prefix-max query gives best dp over smaller prefix values.
    //   bitGeq  : indexed by REVERSED coordinate, prefix-max query gives best dp over >= values.
    vector<int> bitLess(m + 1, NEG), bitGeq(m + 1, NEG);
    auto upd = [&](vector<int> &t, int i, int v) {       // 1-based index
        for (++i; i <= m; i += i & (-i)) t[i] = max(t[i], v);
    };
    auto qry = [&](vector<int> &t, int i) {              // max over [0..i], 0-based i
        int r = NEG;
        for (++i; i > 0; i -= i & (-i)) r = max(r, t[i]);
        return r;
    };

    // Insert j = 0: dp[0] = 0 at coordinate of prefix[0].
    int c0 = cid(prefix[0]);
    upd(bitLess, c0, 0);
    upd(bitGeq, m - 1 - c0, 0);

    int ans = 0;
    for (int i = 1; i <= n; i++) {
        int ci = cid(prefix[i]);
        int best = NEG;
        // prefix[j] < prefix[i]: coordinates [0 .. ci-1]
        if (ci - 1 >= 0) {
            int a1 = qry(bitLess, ci - 1);
            if (a1 > NEG) best = max(best, a1 + 1);
        }
        // prefix[j] >= prefix[i]: coordinates [ci .. m-1] = reversed prefix [0 .. m-1-ci]
        int a2 = qry(bitGeq, m - 1 - ci);
        if (a2 > NEG) best = max(best, a2);
        int dpi = best;                 // there is always at least one j (j = i-1), so best > NEG
        ans = dpi;
        upd(bitLess, ci, dpi);
        upd(bitGeq, m - 1 - ci, dpi);
    }

    cout << ans << "\n";
    return 0;
}
