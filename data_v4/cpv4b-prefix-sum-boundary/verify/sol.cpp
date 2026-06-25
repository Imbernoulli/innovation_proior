#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, w, L, R;
    if (!(cin >> n >> w >> L >> R)) return 0;
    vector<long long> a(n + 1);
    for (long long i = 1; i <= n; i++) cin >> a[i];

    // P[i] = a[1] + ... + a[i], with P[0] = 0 (exclusive prefix).
    vector<long long> P(n + 1, 0);
    for (long long i = 1; i <= n; i++) P[i] = P[i - 1] + a[i];

    // A batch starts at s (1-indexed) and covers s..s+w-1.
    // Valid starts: 1 <= s <= n - w + 1  (inclusive on both ends).
    // Sum of that batch = P[s + w - 1] - P[s - 1].
    long long count = 0;
    if (w >= 1 && w <= n) {
        for (long long s = 1; s <= n - w + 1; s++) {
            long long sum = P[s + w - 1] - P[s - 1];
            if (sum >= L && sum <= R) count++;   // closed band [L, R]
        }
    }

    cout << count << "\n";
    return 0;
}
