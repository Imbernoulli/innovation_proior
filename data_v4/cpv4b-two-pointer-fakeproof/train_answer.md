**Problem.** Given `n` non-negative 31-bit masks `a[0..n-1]` and a budget `K`, count the contiguous
windows `[l, r]` whose bitwise-OR `a[l] | ... | a[r]` is `<= K`. Read `n`, `K`, then the masks from
stdin; print the count. The answer can reach `n(n+1)/2 ~ 2*10^10`, so accumulate it in `long long`.

**Key idea — two pointers on a monotone OR.** OR-ing in more masks can only set bits, so the window
OR is non-decreasing as the window grows. Hence for each right edge `r` the clean left edges form a
suffix `[l*(r), r]`, and `l*(r)` only moves rightward as `r` grows — the classic two-pointer
frontier. Scan `r` left to right: add `a[r]`, then advance `left` while the window OR exceeds `K`;
every start in `[left, r]` is clean, contributing `r - left + 1`.

**Pitfalls.**
1. *OR is not invertible — do not assert a removal identity.* When `left` advances and mask `x`
   leaves a window with OR `O`, the new OR is **not** `O & ~x`, **not** `O ^ x`, and **not**
   `O ^ (x & O)`. Concrete check: window `{6, 5}` has OR `7`; removing `6` should leave OR `5`, but
   all three formulas give `1`, because bit `2` (value `4`) is shared by both masks and must stay
   set. `(O, x)` simply does not carry "how many remaining masks still set this bit." The fix is
   per-bit counts: `cnt[b]` = number of in-window masks with bit `b` set; bit `b` is in the OR iff
   `cnt[b] > 0`. Add increments, remove decrements, all `O(31)`.
2. *Unguarded shrink walks off the window.* A single mask larger than `K` forces the loop to empty
   the window. Guard the shrink with `while (left <= right && curOr() > K)` so `left` never advances
   past `right` and never indexes a mask outside the window.
3. *Bit width and answer type.* `BITS = 31` covers bits `0..30`, exactly the range of `2^31 - 1`;
   `BITS = 30` would silently drop the top bit. The count needs `long long`, not `int`.

**Edge cases.** `n = 0` -> `0`. Single mask `> K` -> `0`; `<= K` -> `1`. `K = 0` counts only windows
of all-zero masks (handled without special-casing). `K = 2^31 - 1` -> every window clean,
`n(n+1)/2`. Heavily overlapping bits are exactly where the false xor/and-not identity would fail and
where per-bit counts are needed.

**Complexity.** `O(31 n)` time (each mask added and removed once, each OR read `O(31)`), `O(31)`
extra space.

**Code.**

```cpp
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
```
