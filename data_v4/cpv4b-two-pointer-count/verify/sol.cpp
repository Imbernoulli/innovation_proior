#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    long long L, R;
    cin >> L >> R;
    vector<long long> f(n);
    for (auto &x : f) cin >> x;

    const long long MOD = 1000000007LL;

    sort(f.begin(), f.end());

    // For an unordered pair {i, j} with sorted values f[i] <= f[j], the gap is
    // f[j] - f[i], and it is "compatible" iff L <= f[j] - f[i] <= R.
    // Fix the LARGER element j (right end). Its valid partners are the indices
    // i < j with  f[j] - R <= f[i] <= f[j] - L.
    // As j increases, both window bounds (f[j]-R and f[j]-L) are nondecreasing,
    // so two pointers lo, hi sweep the prefix [0, j) once. Counting only
    // partners to the LEFT of j makes each unordered pair counted exactly once.
    long long total = 0;       // exact count, fits in long long (<= n*(n-1)/2)
    int lo = 0, hi = 0;        // lo: first i with f[i] >= f[j]-R; hi: first i with f[i] > f[j]-L
    for (int j = 0; j < n; j++) {
        // hi: number of elements among the prefix with value <= f[j]-L.
        while (hi < j && f[hi] <= f[j] - L) hi++;
        // lo: number of elements among the prefix with value <  f[j]-R.
        while (lo < j && f[lo] <  f[j] - R) lo++;
        // valid partners i in [lo, hi): values in [f[j]-R, f[j]-L].
        total += (long long)(hi - lo);
    }

    cout << (total % MOD) << "\n";
    return 0;
}
