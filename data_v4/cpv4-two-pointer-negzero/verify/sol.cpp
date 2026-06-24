#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;                 // no input / n = 0 -> 0 pairs
    long long T;
    cin >> T;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    if (n < 2) {                               // fewer than two thrusters: no pair exists
        cout << 0 << "\n";
        return 0;
    }

    sort(a.begin(), a.end());

    // Count unordered pairs {i, j}, i < j (in sorted order), with a[i] + a[j] >= T.
    // Two converging pointers on the sorted array.
    long long count = 0;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        if (a[lo] + a[hi] >= T) {
            // a[hi] paired with every index in [lo, hi-1] also satisfies the threshold,
            // because a is sorted ascending: those are (hi - lo) valid pairs.
            count += (long long)(hi - lo);
            hi--;
        } else {
            lo++;                              // smallest element can never reach T with this hi
        }
    }

    cout << count << "\n";
    return 0;
}
