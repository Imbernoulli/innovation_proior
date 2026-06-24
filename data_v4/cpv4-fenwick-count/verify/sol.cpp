#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L, R;
    if (!(cin >> n >> L >> R)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Compressed key set = the distinct values that actually appear in a[].
    // We only ever INSERT a[j] values, so these are the only keys the Fenwick
    // can hold; range-query endpoints are resolved against this sorted list by
    // binary search, no need to insert the endpoints themselves.
    vector<long long> ks(a);
    sort(ks.begin(), ks.end());
    ks.erase(unique(ks.begin(), ks.end()), ks.end());
    int m = (int)ks.size();

    // 1-indexed Fenwick over compressed positions 1..m; bit[p] counts how many
    // already-inserted elements have compressed value p.
    vector<int> bit(m + 1, 0);
    auto upd = [&](int p) {                 // p in 1..m
        for (; p <= m; p += p & (-p)) bit[p] += 1;
    };
    auto pref = [&](int p) -> long long {   // count of inserted elems at positions 1..p
        long long s = 0;
        for (; p > 0; p -= p & (-p)) s += bit[p];
        return s;
    };
    // count inserted elements whose real VALUE lies in [lo, hi] (inclusive)
    auto rangeCount = [&](long long lo, long long hi) -> long long {
        if (lo > hi) return 0;
        // keys at positions 1..hiIdx are <= hi  (upper_bound = first key > hi)
        int hiIdx = (int)(upper_bound(ks.begin(), ks.end(), hi) - ks.begin());
        // keys at positions 1..loIdx are < lo   (lower_bound = first key >= lo)
        int loIdx = (int)(lower_bound(ks.begin(), ks.end(), lo) - ks.begin());
        if (hiIdx <= loIdx) return 0;
        return pref(hiIdx) - pref(loIdx);
    };

    long long ans = 0;
    for (int j = 0; j < n; j++) {
        long long v = a[j];
        // earlier i (already inserted) with |a[i] - v| in [L, R]:
        //   lower band  a[i] in [v-R, v-L]
        //   upper band  a[i] in [v+L, v+R]
        // When L==0 both bands include v itself, so their union overlaps exactly
        // at the value v; summing both bands would double-count earlier elements
        // equal to v. Treat L==0 as ONE merged band [v-R, v+R].
        if (L == 0) {
            ans += rangeCount(v - R, v + R);
        } else {
            ans += rangeCount(v - R, v - L);   // lower band
            ans += rangeCount(v + L, v + R);   // upper band (disjoint since L>=1)
        }
        // insert a[j] AFTER querying, so a pair never uses i == j
        int p = (int)(lower_bound(ks.begin(), ks.end(), v) - ks.begin()) + 1; // 1-indexed
        upd(p);
    }

    cout << ans << "\n";
    return 0;
}
