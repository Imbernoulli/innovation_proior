#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;            // n = 0 (or empty input) -> no day, answer 0

    // L is the running level (a prefix sum of the daily deltas), starting at L[0] = 0
    // BEFORE any day. peak is the maximum level seen so far, including day 0.
    // The maximum drawdown is max over j of (peak_up_to_j - L[j]); i = j gives 0,
    // so the answer is never negative even if the level only ever rises.
    long long level = 0;                  // L[0] = 0, the reference before day 1
    long long peak = 0;                   // best level seen so far == L[0]
    long long best = 0;                   // i = j allowed -> drawdown >= 0

    for (int i = 0; i < n; i++) {
        long long d;
        cin >> d;
        level += d;                       // L[i+1] = L[i] + d[i]
        if (peak - level > best) best = peak - level;   // drop from the running peak
        if (level > peak) peak = level;   // update peak AFTER measuring the drop
    }

    cout << best << "\n";
    return 0;
}
