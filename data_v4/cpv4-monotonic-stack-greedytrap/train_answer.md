**Problem.** A skyline is a string `s` of `n` digits (`'0'`–`'9'`), tower heights left to right.
Demolish exactly `k` towers (`0 <= k <= n`); the survivors keep their original order and are pushed
together. Output the lexicographically smallest surviving string of length `n - k` (empty when
`k = n`; leading zeros are kept, not stripped). Read `n`, `k`, and `s` from stdin; print the result.

**Why the obvious greedy is wrong.** "Demolish the `k` tallest towers (largest digits)" fails because
lexicographic order is *positional* — an early digit outweighs every later one — so the harm a tower
does depends on its position, not just its height. On `s = "2102"`, `k = 1` the digit `'2'` is
tallest and appears twice; removing the *last* one gives `"210"`, but removing the *earliest* tall
tower gives `"102"`, which is strictly smaller. With two removals the gap widens: on `"12002"`,
`k = 2`, the value greedy yields `"100"` while the true optimum is `"002"`. The value greedy is only
right if its tie-break happens to prefer the earliest max — a coincidence, not a principle. Discard
it.

**Key idea — monotonic (non-decreasing) stack.** Scan left to right, building survivors on a stack
and carrying a `budget = k` of remaining demolitions. When a tower `c` arrives, while `budget > 0`
and the stack top is **strictly taller** than `c`, pop the top and spend one demolition; then push
`c`. Popping a taller predecessor lets a shorter tower occupy its more-significant slot, which is
exactly what lexicographic order rewards. Each tower is pushed once and popped at most once, so the
stack stays non-decreasing bottom-to-top and the whole scan is `O(n)`.

**Pitfalls.**
1. *Strict vs non-strict pop.* Use `>`, not `>=`. Popping an **equal** top spends a demolition for no
   improvement (the same digit refills the slot) and starves a later, genuinely useful removal. Trace:
   `"112"`, `k = 1` with `>=` returns `"12"`; the correct answer is `"11"`.
2. *Leftover budget.* If the skyline is non-decreasing, no pop ever fires and the scan ends with
   `budget > 0`. Those removals must still happen, so drain them from the **tail** (least significant
   towers). Forgetting this returns a string of length `n`, not `n - k`: `"12345"`, `k = 2` would
   print `"12345"` instead of `"123"`.

**Edge cases.** `k = 0` → print `s` unchanged (no pops, no drain). `k = n` → every tower removed,
output an empty line. All-equal (`"00000"`) → no pop fires; the drain trims the tail. Strictly
decreasing → the front (most significant) towers are demolished. The positional case `"2102"`,
`k = 1` → `"102"`, the optimum the value greedy misses.

**Complexity.** `O(n)` time (each character pushed once, popped at most once), `O(n)` space for the
result string. Comfortable for `n = 2*10^5` under 1 second; values are single digits so there is no
overflow concern.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long k;
    if (!(cin >> n >> k)) return 0;
    string s;
    cin >> s;

    // Monotonic (non-decreasing) stack: while we still have removals left and the
    // top of the stack is strictly greater than the current character, pop it.
    // This greedily fixes the earliest position where a smaller character can
    // take a more significant slot.
    string st;
    st.reserve(n);
    long long budget = k;
    for (int i = 0; i < n; i++) {
        char c = s[i];
        while (budget > 0 && !st.empty() && st.back() > c) {
            st.pop_back();
            budget--;
        }
        st.push_back(c);
    }
    // If removals remain (string was non-decreasing), drop from the tail.
    while (budget > 0) {
        st.pop_back();
        budget--;
    }

    cout << st << "\n";
    return 0;
}
```
