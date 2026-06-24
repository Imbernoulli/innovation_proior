#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;            // n = 0 (or empty input) -> answer 0

    // Level starts at 0 before any day; this initial level is a valid "peak".
    long long prefix = 0;                  // P[-1] = 0
    long long peak = 0;                    // best (max) level seen so far, including the start
    long long answer = 0;                  // i = j gives a decline of 0, so answer >= 0

    for (int i = 0; i < n; i++) {
        long long d;
        cin >> d;
        prefix += d;                       // level after day i
        // decline ending at this day = (highest earlier-or-equal level) - current level
        answer = max(answer, peak - prefix);
        peak = max(peak, prefix);          // update running peak AFTER measuring the decline
    }

    cout << answer << "\n";
    return 0;
}
