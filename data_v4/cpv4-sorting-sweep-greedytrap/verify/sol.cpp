#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> profit 0
    vector<long long> s(n), f(n), p(n);
    for (int i = 0; i < n; i++) cin >> s[i] >> f[i] >> p[i];

    // Sort jobs by finishing time (the sweep order).
    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int x, int y){ return f[x] < f[y]; });
    vector<long long> S(n), F(n), P(n);
    for (int i = 0; i < n; i++) { S[i] = s[idx[i]]; F[i] = f[idx[i]]; P[i] = p[idx[i]]; }

    // best[i] = max profit considering the first i jobs (in finish order).
    // For job i (0-based), find p = largest index j < i with F[j] <= S[i] (compatible),
    // via binary search on the sorted F array.
    vector<long long> best(n + 1, 0);
    for (int i = 0; i < n; i++) {
        // largest k in [0, i) with F[k] <= S[i]; count of such = j, so best[j] is reachable.
        int lo = 0, hi = i; // search in F[0..i-1]
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (F[mid] <= S[i]) lo = mid + 1;
            else hi = mid;
        }
        // lo = number of jobs among first i with F <= S[i]
        long long take = best[lo] + P[i];
        long long skip = best[i];
        best[i + 1] = max(take, skip);
    }

    cout << best[n] << "\n";
    return 0;
}
