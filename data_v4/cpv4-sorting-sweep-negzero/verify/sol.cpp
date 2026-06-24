#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long T;
    if (!(cin >> n >> T)) return 0;        // empty input -> nothing to do
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    sort(a.begin(), a.end());

    // Count unordered pairs (i<j) with a[i]+a[j] <= T via a two-pointer sweep.
    // After sorting, for the pair anchored at the high end `hi`, every index in
    // [lo, hi-1] forms a valid pair when a[lo]+a[hi] <= T (monotone in lo).
    long long ans = 0;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        if (a[lo] + a[hi] <= T) {
            ans += (long long)(hi - lo);   // all of lo..hi-1 pair with hi
            lo++;
        } else {
            hi--;
        }
    }

    cout << ans << "\n";
    return 0;
}
