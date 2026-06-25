**Problem.** A dock conveyor holds `n` containers in a line with non-negative integer weights
`w[0..n-1]`; a crane of capacity `S` can lift any contiguous run `[i..j]` whose total weight is
`<= S`. Count the runs `(i, j)` with `0 <= i <= j <= n-1` and `w[i]+...+w[j] <= S`. Read `n`, `S`,
and the weights from stdin; print the count. Constraints: `n <= 2*10^5`, `w[i] <= 10^9`,
`S <= 10^18`.

**Key idea — two-pointer sliding window.** Because all weights are non-negative, for a fixed right
end `right` the window sum `sum(i..right)` is non-increasing in the start `i`, so the valid starts
form a suffix `{left, ..., right}`. Maintain `[left..right]` and a running `sum`: step `right`
forward adding `w[right]`, and while `sum > S` advance `left` subtracting `w[left]`. Then every start
in `[left..right]` yields a liftable run, contributing `right - left + 1`. The smallest valid start
`left` only moves rightward as `right` grows, so the inner loop is amortized `O(n)`. Sanity check on
`w=[3,1,4,1,5,2], S=7`: contributions `1+2+2+3+2+2 = 12`, which matches enumerating all valid runs.

**Pitfalls — two silent 32-bit overflows, this is the crux.**
1. *The count overflows int.* There are up to `n*(n+1)/2 ~ 2*10^10` runs, past `int`'s `~2.1*10^9`.
   On a large all-fits case (`n=2*10^5`, all `w=10^9`, `S=2*10^14`) the true answer is
   `20000100000`, but an `int count` wraps to `-1474736480` (`= 20000100000 mod 2^32`, re-centered
   into the signed range). The negative output is the tell. Small tests never reveal this — you only
   catch it by tracing a large case. Use `long long count`.
2. *The window sum overflows int.* The running `sum` reaches `~2*10^14`. With an `int sum`, a window
   like `[2*10^9, 2*10^9]` under `S=2*10^9` computes `4*10^9` which wraps to `-294967296`; then
   `sum > S` is false, the window never shrinks, and the over-capacity pair is counted — answer `3`
   instead of `2`. Use `long long sum`. (`S` is read as `long long` too, since it can be `10^18`.)

**Edge cases.** `n = 0`: empty scan, count `0`. `S = 0` with all zeros: every run fits, count
`n*(n+1)/2`. A single container heavier than `S`: the `while` pushes `left` to `right+1`, and the
`left <= right` guard makes the contribution `right - left + 1 = 0` (never negative). Capacity above
the line total: window never shrinks, count `n*(n+1)/2`.

**Complexity.** `O(n)` time (each pointer advances at most `n` times), `O(1)` extra space beyond the
input array.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // Two pointers: for each right end, shrink from the left while the
    // window sum exceeds S. Every valid window ending at `right` with left
    // boundary in [left, right] is admissible, contributing (right-left+1).
    long long count = 0;   // up to ~n*(n+1)/2 = 2*10^10 -> must be 64-bit
    long long sum = 0;     // up to n*max(w) = 2*10^14 -> must be 64-bit
    int left = 0;
    for (int right = 0; right < n; right++) {
        sum += w[right];
        while (left <= right && sum > S) {
            sum -= w[left];
            left++;
        }
        // windows [left..right], [left+1..right], ..., [right..right]
        count += (long long)(right - left + 1);
    }

    cout << count << "\n";
    return 0;
}
```
