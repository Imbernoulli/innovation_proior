#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> w(n);
    long long maxw = 1;
    for (auto &x : w) { cin >> x; maxw = max(maxw, x); }

    // cuts(p) = total chisel blows needed if each blow can leave pieces of size <= p.
    // A boulder of weight x needs ceil(x/p) final pieces, i.e. ceil(x/p)-1 blows.
    // ceil(x/p)-1 == (x-1)/p  in integer arithmetic (x>=1, p>=1).
    auto cuts = [&](long long p) -> long long {
        long long total = 0;
        for (long long x : w) {
            total += (x - 1) / p;       // ceil(x/p) - 1 blows for this boulder
            if (total > k) return total; // early exit, avoid overflow-ish blowup
        }
        return total;
    };

    // feasible(p): with chisel power p, total blows <= k. Monotone: larger p => fewer blows.
    // We want the MINIMUM p with feasible(p) true. p ranges over [1, maxw]
    // (p = maxw always needs 0 blows per boulder, so it is always feasible).
    long long lo = 1, hi = maxw;        // search space [lo, hi], answer guaranteed in it
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (cuts(mid) <= k) hi = mid;   // mid works -> answer is mid or smaller
        else lo = mid + 1;              // mid fails -> answer strictly larger
    }
    cout << lo << "\n";
    return 0;
}
