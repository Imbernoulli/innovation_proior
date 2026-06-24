#include <bits/stdc++.h>
using namespace std;

// Count contiguous subarrays whose sum lies in [L, R].
// Values a[i] >= 1, so prefix sums are strictly increasing; for a fixed right
// endpoint the set of valid left endpoints is a contiguous window -> two pointers.
// We count atMost(R) - atMost(L-1), where atMost(X) = number of subarrays with sum <= X.

static long long atMost(const vector<long long>& a, long long X) {
    // Number of contiguous subarrays with sum <= X. With a[i] >= 1 and X possibly
    // negative, the window logic must handle X < 0 (answer 0).
    if (X < 0) return 0;
    long long cnt = 0, sum = 0;
    int left = 0;
    int n = (int)a.size();
    for (int right = 0; right < n; right++) {
        sum += a[right];
        while (sum > X) {            // shrink until window sum <= X
            sum -= a[left];
            left++;
        }
        // [left .. right] is the longest window ending at right with sum <= X;
        // every subarray ending at right with start in [left, right] qualifies.
        cnt += (long long)(right - left + 1);
    }
    return cnt;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L, R;
    if (!(cin >> n >> L >> R)) return 0;
    vector<long long> a(n);
    for (auto& x : a) cin >> x;

    long long answer = atMost(a, R) - atMost(a, L - 1);
    cout << answer << "\n";
    return 0;
}
