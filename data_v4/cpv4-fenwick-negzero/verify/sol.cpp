#include <bits/stdc++.h>
using namespace std;

// Fenwick (Binary Indexed Tree) for prefix counts over compressed values.
struct Fenwick {
    int n;
    vector<long long> t;
    Fenwick(int n) : n(n), t(n + 1, 0) {}
    void add(int i, long long v) {            // i is 1-based
        for (; i <= n; i += i & (-i)) t[i] += v;
    }
    long long sumPrefix(int i) {              // sum over [1..i], i is 1-based
        long long s = 0;
        for (; i > 0; i -= i & (-i)) s += t[i];
        return s;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;               // empty input -> n = 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Prefix sums P[0..n], P[0] = 0, P[k] = a[0] + ... + a[k-1].
    // Subarray [l, r] sum = P[r+1] - P[l]; we want it strictly negative,
    // i.e. count pairs (i, j) with 0 <= i < j <= n and P[j] < P[i].
    vector<long long> P(n + 1);
    P[0] = 0;
    for (int k = 0; k < n; k++) P[k + 1] = P[k] + a[k];

    // Coordinate-compress all n+1 prefix-sum values (negatives/zeros included).
    vector<long long> vals(P.begin(), P.end());
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int m = (int)vals.size();
    auto rankOf = [&](long long x) {          // 1-based rank in sorted unique vals
        return int(lower_bound(vals.begin(), vals.end(), x) - vals.begin()) + 1;
    };

    // Sweep j = 0..n in order. For each P[j], the number of earlier P[i] (i < j)
    // with P[i] > P[j] is (j) - (count of earlier values <= P[j]).
    Fenwick fen(m);
    long long answer = 0;
    for (int j = 0; j <= n; j++) {
        int r = rankOf(P[j]);
        long long inserted = j;               // we have inserted P[0..j-1] so far
        long long leq = fen.sumPrefix(r);     // earlier values with rank <= r (i.e. P[i] <= P[j])
        long long greater = inserted - leq;   // earlier values with P[i] > P[j]
        answer += greater;
        fen.add(r, 1);                        // now insert P[j]
    }

    cout << answer << "\n";
    return 0;
}
