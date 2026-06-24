**Problem.** A stamp is engraved with a logo word `s`; each press prints some non-empty *prefix* of `s` (any of `s[0..0] .. s`). Presses are laid end to end with no gaps or overlaps. Given `s` and a banner `t`, reproduce `t` exactly as a concatenation of such prefixes using the fewest presses (blocks), or report `-1` if `t` cannot be tiled this way. Read `s` and `t` from stdin (`1 <= |s|,|t| <= 2*10^5`, lowercase), print the minimum block count or `-1`.

**Why the obvious greedy is wrong.** "At each position press the longest prefix of `s` that matches the banner here, then advance" is tempting because covering more banner per block *looks* like it can only help. It fails because a block boundary is a global commitment — a long block changes where the next block must begin. On `s = "babba"`, `t = "babbabb"` greedy presses `babba | b | b` = 3 blocks, but `bab | babb` = 2 blocks (the slightly shorter first block lands on a position a single `babb` can finish). Worse, greedy can *deadlock*: on `s = "aab"`, `t = "aaaaab"` it eats `aa | aa | a` and gets stuck on the final `b`, falsely reporting `-1`, even though `a | aa | aab` tiles it in 3. So longest-match greedy both miscounts and can declare a feasible banner infeasible. Discarded.

**Key idea — reach + Jump Game II.** A block at position `i` may be any prefix of `s` matching `t` there. Crucially the legal block lengths *nest*: if a length-`L` prefix matches at `i`, every shorter prefix matches too, so the legal lengths form a full interval `1..reach[i]`, where `reach[i]` is the longest prefix of `s` matching `t` at `i`. Compute `reach` with the Z-function of `c = s + sep + t` (separator outside the alphabet): for index `p` in the `t`-part, `min(Z[p], |s|)` is exactly `reach`. Then minimum blocks to advance from `0` to `n`, moving from `i` to any `j` in `[i+1, i+reach[i]]`, is **Jump Game II**, solved optimally by the *level/BFS* greedy: keep the window of indices reachable with the current block count, track the farthest any window member reaches, and spend one block to push the frontier to `farthest` when the window is exhausted.

**Pitfalls.**
1. *Wrong greedy.* The optimal greedy here is the *level* greedy, not "jump maximally far" (`i += reach[i]`). The maximal-jump greedy is the disproved one — it is suboptimal and can deadlock.
2. *Feasibility guards.* `-1` must be detected three ways: an index `i > curEnd` is unreachable; at a window boundary `farthest <= curEnd` means the frontier cannot move; and after the loop `curEnd < n` means the end was never reached. Omitting these turns infeasible `s="ab", t="b"` into a phantom `1`.
3. *Z-offset.* The `t`-part of `c = s + sep + t` begins at index `|s| + 1`, not `|s|`. Using `base = |s|` reads the separator slot and shifts every `reach` value, silently producing wrong `-1`s.

**Edge cases.** Single-char logo `s="a"`: `t="aaaa"` → `4` (only prefix is `a`). Untileable first char `s="abc", t="x"` → `-1`. Self-overlapping `s="aaaa", t="aaaaaaa"` → `2` (`aaaa|aaa`). Whole-logo repeats `s="abc", t="abcabc"` → `2`. Block count is at most `n`, but it is held in `long long` for safety; `i + reach[i] <= 2n` fits in `int`.

**Complexity.** `O(|s| + |t|)` time (Z-function plus one linear sweep), `O(|s| + |t|)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s, t;
    if (!(cin >> s >> t)) return 0;     // missing input -> nothing to do
    int m = (int)s.size();
    int n = (int)t.size();

    // reach[i] = length of the longest prefix of s that matches t starting at position i.
    // Build via the Z-function of  c = s + '\x01' + t  (separator absent from the alphabet).
    // For position p inside the t-part, Z[p] (capped at m) is exactly that longest prefix length,
    // and because every shorter prefix of s is a prefix of that match, the set of block lengths
    // usable at i is precisely {1, 2, ..., reach[i]}.
    string c;
    c.reserve(m + 1 + n);
    c += s;
    c.push_back('\x01');
    c += t;
    int N = (int)c.size();
    vector<int> Z(N, 0);
    int l = 0, r = 0;
    for (int i = 1; i < N; i++) {
        if (i < r) Z[i] = min(r - i, Z[i - l]);
        while (i + Z[i] < N && c[Z[i]] == c[i + Z[i]]) Z[i]++;
        if (i + Z[i] > r) { l = i; r = i + Z[i]; }
    }
    vector<int> reach(n, 0);
    int base = m + 1;                   // index in c where the t-part begins
    for (int i = 0; i < n; i++) reach[i] = min(Z[base + i], m);

    // Minimum number of blocks = minimum jumps to advance from index 0 to index n,
    // where from i you may land on any j in [i+1, i+reach[i]].
    // This is Jump Game II: the LEVEL/BFS greedy (extend the current reachable window to the
    // farthest its members can reach, counting one block per extension) is optimal. The tempting
    // "take the longest prefix each step" (i += reach[i]) greedy is NOT optimal and can deadlock.
    if (n == 0) { cout << 0 << "\n"; return 0; }
    long long blocks = 0;
    int curEnd = 0;                     // farthest index settled with `blocks` jumps
    int farthest = 0;                   // farthest index reachable with one more jump
    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (i > curEnd) { ok = false; break; }   // index i unreachable -> infeasible
        if (i + reach[i] > farthest) farthest = i + reach[i];
        if (i == curEnd) {              // exhausted current window: must spend a block
            if (farthest <= curEnd) { ok = false; break; }  // cannot progress
            blocks++;
            curEnd = farthest;
            if (curEnd >= n) break;
        }
    }
    if (!ok || curEnd < n) cout << -1 << "\n";
    else cout << blocks << "\n";
    return 0;
}
```
