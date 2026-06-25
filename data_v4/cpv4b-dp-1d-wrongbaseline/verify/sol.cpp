#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;
    if (n == 0) { cout << 0 << "\n"; return 0; }  // contract guarantees n>=1, but be safe

    // Linear (non-wrapping) maximum subarray sum, non-empty (Kadane).
    long long best = a[0], cur = a[0];
    long long total = a[0];
    // Linear minimum subarray sum, non-empty (Kadane on negated logic).
    long long worst = a[0], curMin = a[0];
    for (int i = 1; i < n; i++) {
        cur = max(a[i], cur + a[i]);
        best = max(best, cur);
        curMin = min(a[i], curMin + a[i]);
        worst = min(worst, curMin);
        total += a[i];
    }

    // Wrapping candidate: total minus the minimum interior subarray.
    // If worst == total then every element is in the minimum subarray, i.e.
    // the "complement" wrap would be empty; that is illegal (segment must be
    // non-empty), so we must NOT take the wrap in that case.
    long long answer;
    if (worst == total) {
        // All elements lie in the minimum subarray => array is all non-positive
        // in the sense that the best non-empty pick is just the linear best.
        answer = best;
    } else {
        answer = max(best, total - worst);
    }

    cout << answer << "\n";
    return 0;
}
