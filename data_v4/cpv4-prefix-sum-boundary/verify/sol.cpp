#include <bits/stdc++.h>
using namespace std;

struct Fenwick {
    int n;
    vector<int> bit;
    Fenwick(int n_) : n(n_), bit(n_ + 1, 0) {}
    void add(int i) {                 // i is 1-based index
        for (; i <= n; i += i & (-i)) bit[i] += 1;
    }
    int pref(int i) {                 // count of inserted values at positions 1..i
        int s = 0;
        for (; i > 0; i -= i & (-i)) s += bit[i];
        return s;
    }
    int range(int lo, int hi) {       // count in [lo, hi], 1-based, inclusive
        if (lo > hi) return 0;
        return pref(hi) - pref(lo - 1);
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L, R;
    if (!(cin >> n >> L >> R)) return 0;

    vector<long long> P(n + 1);
    P[0] = 0;
    for (int i = 1; i <= n; i++) {
        long long x;
        cin >> x;
        P[i] = P[i - 1] + x;
    }

    // Coordinate-compress all prefix values P[0..n].
    vector<long long> vals(P.begin(), P.end());
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int m = (int)vals.size();
    auto idx = [&](long long v) -> int { // 1-based position of v in vals (v must exist)
        return (int)(lower_bound(vals.begin(), vals.end(), v) - vals.begin()) + 1;
    };

    Fenwick fw(m);
    long long answer = 0;

    // Sweep r = 1..n. Before querying for r, the set of available j is {0,..,r-1}.
    // We insert P[r-1] just before processing r, so P[0] is in before r=1.
    fw.add(idx(P[0]));
    for (int r = 1; r <= n; r++) {
        // Need P[r]-R <= P[j] <= P[r]-L, j in {0..r-1}.
        long long lo = P[r] - R;
        long long hi = P[r] - L;
        // Map [lo, hi] onto compressed coordinates: count compressed values v with lo <= v <= hi.
        int li = (int)(lower_bound(vals.begin(), vals.end(), lo) - vals.begin()) + 1; // first >= lo
        int hiPos = (int)(upper_bound(vals.begin(), vals.end(), hi) - vals.begin());  // last <= hi (1-based)
        answer += fw.range(li, hiPos);
        if (r < n) fw.add(idx(P[r])); // make P[r] available for future r' > r
    }

    cout << answer << "\n";
    return 0;
}
