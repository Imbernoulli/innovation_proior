#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Track, for the subarray ending at the current position, BOTH the maximum
    // and the minimum product. A negative current element swaps their roles
    // (min*neg can become the new max), so we must carry the minimum too.
    long long curMax = a[0];      // best product of a subarray ending here
    long long curMin = a[0];      // worst product of a subarray ending here
    long long best   = a[0];      // global answer
    for (int i = 1; i < n; i++) {
        long long x = a[i];
        long long c1 = x;             // start fresh at i
        long long c2 = curMax * x;    // extend previous best
        long long c3 = curMin * x;    // extend previous worst (key for negatives)
        curMax = max(c1, max(c2, c3));
        curMin = min(c1, min(c2, c3));
        best = max(best, curMax);
    }

    cout << best << "\n";
    return 0;
}
