#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long K;
    if (!(cin >> n >> K)) return 0;
    vector<unsigned int> a(n);
    for (auto &x : a) cin >> x;

    // Count subarrays whose bitwise-OR is <= K.
    // OR is monotonic: extending a window can only set more bits, so the OR
    // never decreases. For a fixed right end, valid left ends form a suffix;
    // hence two pointers. We CANNOT undo an OR by xor/and-not when shrinking,
    // so we keep per-bit counts: bit b is in the window-OR iff cnt[b] > 0.
    const int BITS = 31; // values fit in 31 bits (< 2^31)
    vector<int> cnt(BITS, 0);

    auto curOr = [&]() -> unsigned int {
        unsigned int o = 0;
        for (int b = 0; b < BITS; b++) if (cnt[b] > 0) o |= (1u << b);
        return o;
    };

    long long ans = 0;
    int left = 0;
    for (int right = 0; right < n; right++) {
        // add a[right]
        for (int b = 0; b < BITS; b++) if (a[right] & (1u << b)) cnt[b]++;
        // shrink from the left while the window OR exceeds K
        while (left <= right && curOr() > K) {
            for (int b = 0; b < BITS; b++) if (a[left] & (1u << b)) cnt[b]--;
            left++;
        }
        // all subarrays ending at `right` with start in [left, right] are valid
        ans += (right - left + 1);
    }

    cout << ans << "\n";
    return 0;
}
