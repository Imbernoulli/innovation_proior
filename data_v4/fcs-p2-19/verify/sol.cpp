#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;
    if (n == 0) { cout << 0 << "\n"; return 0; }

    // inc[i] = length of the longest STRICTLY increasing subsequence that ENDS at i.
    // dec[i] = length of the longest STRICTLY decreasing subsequence that STARTS at i.
    // A bitonic subsequence with peak at index i (the peak counted once) has length
    // inc[i] + dec[i] - 1. We require an actual increase before the peak, i.e. inc[i] >= 2,
    // so a purely non-increasing array yields answer 0 (no valid increase-then-decrease).
    vector<int> inc(n, 1), dec(n, 1);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++)
            if (a[j] < a[i])
                inc[i] = max(inc[i], inc[j] + 1);

    for (int i = n - 1; i >= 0; i--)
        for (int j = i + 1; j < n; j++)
            if (a[j] < a[i])
                dec[i] = max(dec[i], dec[j] + 1);

    int best = 0;
    for (int i = 0; i < n; i++)
        if (inc[i] >= 2)                          // peak must be preceded by a real increase
            best = max(best, inc[i] + dec[i] - 1);

    cout << best << "\n";
    return 0;
}
