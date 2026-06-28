#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // The subarray must be non-empty AFTER any deletion, so we need n >= 1.
    // noDel = best sum of a subarray ending at i with NO element deleted.
    // oneDel = best sum of a subarray ending at i with EXACTLY one element deleted.
    const long long NEG = LLONG_MIN / 4;
    long long noDel = NEG, oneDel = NEG;
    long long best = NEG;
    for (int i = 0; i < n; i++) {
        // oneDel must be computed from the PREVIOUS noDel/oneDel before noDel is updated.
        long long newOneDel = max(noDel,                 // delete a[i]; segment so far = old noDel ending at i-1
                                  oneDel + a[i]);        // deletion already used earlier; extend by a[i]
        long long newNoDel = max(a[i], noDel + a[i]);    // standard Kadane: start fresh or extend
        noDel = newNoDel;
        oneDel = newOneDel;
        best = max(best, noDel);
        best = max(best, oneDel);
    }

    cout << best << "\n";
    return 0;
}
