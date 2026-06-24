**Problem.** Given a string `S` of length `n` (lowercase letters, `n <= 2*10^5`), for every length `L` from `1` to `n` report `c[L]`, the number of times the length-`L` prefix `S[0..L-1]` occurs as a substring of `S` with overlaps allowed. Print `c[1] c[2] ... c[n]`. Invariants: `c[L] >= 1` (self-match at position `0`) and `c[n] = 1`.

**Why brute force fails.** Comparing each of the `n` prefixes against every start position is `O(n^2)` — on `"aaaa...a"` that is `~2*10^10` comparisons, far over a 1 s limit. Walking the KMP border chain from each end position is also `O(n^2)` in the worst case. A single propagation pass is needed.

**Key idea — failure function plus one push pass.** Build the KMP failure function `pi` (`pi[i]` = longest proper prefix of `S[0..i]` that is also a suffix). The length-`L` prefix occurs *ending at* `i` exactly when `L` lies on the border chain `pi[i], pi[pi[i]-1], ...`. Instead of walking each chain:

1. **Seed.** For each `i`, if `pi[i] > 0`, do `cnt[pi[i]]++` — one count at the top of `i`'s proper border chain.
2. **Push, high to low.** For `L = n` down to `1`, let `b = pi[L-1]`; if `b > 0`, do `cnt[b] += cnt[L]`. Because a proper border is strictly shorter (`b < L`), going high-to-low guarantees `cnt[L]` is final before it donates.
3. **Self-occurrence, last.** Add `cnt[L] += 1` for every `L` (each prefix matches itself at position `0`).

This is `O(n)`: the failure function is amortized linear, and each of the three passes is one loop.

**Pitfalls.**
- *Self `+1` placed too early.* Initializing each `cnt[L]=1` before the push makes the self term propagate down every border chain and get re-counted. On `"abacaba"` this turns the correct `4` into `7` at `c[1]`. The `+1` must come strictly **after** all propagation.
- *Wrong push direction.* Pushing low-to-high reads `cnt[L]` before later donations into it have arrived, losing counts. On `"aaaa"` it gives `3 3 2 1` instead of `4 3 2 1`. Only high-to-low finalizes each cell before it donates.
- *Missing guards.* Without `pi[i] > 0` on the seed and `b > 0` on the push, counts leak into the unused length-`0` cell; both guards are load-bearing.
- *Indexing.* The border of the length-`L` prefix is `pi[L-1]`, not `pi[L]` — an off-by-one here silently shifts every count.

**Edge cases.** `n = 1` -> `1`. All-distinct `"abcde"` -> `1 1 1 1 1` (no seeds, no pushes). Maximal overlap `"a"*n` -> `c[L] = n - L + 1`. Always `c[n] = 1`.

**Complexity.** `O(n)` time, `O(n)` space. Counts fit in `int` (`<= n`), accumulated in `long long` for safety.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) return 0;            // empty / no input -> nothing to print
    int n = (int)s.size();

    // KMP failure function: pi[i] = length of the longest proper prefix of
    // s[0..i] that is also a suffix of s[0..i].  pi[0] = 0.
    vector<int> pi(n, 0);
    for (int i = 1; i < n; i++) {
        int j = pi[i - 1];
        while (j > 0 && s[i] != s[j]) j = pi[j - 1];
        if (s[i] == s[j]) j++;
        pi[i] = j;
    }

    // cnt[L] = number of occurrences (overlaps allowed) of the length-L prefix
    // of s inside s, for L = 1..n.  We index cnt by length, cnt has size n+1.
    //
    // Step 1: every end position i contributes one occurrence of the prefix of
    // length pi[i] (the longest prefix-suffix ending at i).  Length 0 carries no
    // prefix, so we only seed positive lengths.
    vector<long long> cnt(n + 1, 0);
    for (int i = 0; i < n; i++)
        if (pi[i] > 0) cnt[pi[i]]++;

    // Step 2: if the length-L prefix occurs, then every shorter prefix that is a
    // border of it (length pi[L-1], then pi[pi[L-1]-1], ...) also occurs at each
    // of those positions.  Push counts down the failure chain.  Process lengths
    // from long to short so each cnt[L] is final before it is propagated.
    for (int L = n; L >= 1; L--) {
        int b = pi[L - 1];                // longest border length of prefix of length L
        if (b > 0) cnt[b] += cnt[L];
    }

    // Step 3: the length-L prefix also occurs once as the whole prefix itself
    // (the trivial occurrence at position 0), which the failure function never
    // counts.  Add it exactly once per length.
    for (int L = 1; L <= n; L++) cnt[L] += 1;

    for (int L = 1; L <= n; L++) {
        cout << cnt[L];
        cout << (L == n ? '\n' : ' ');
    }
    return 0;
}
```
