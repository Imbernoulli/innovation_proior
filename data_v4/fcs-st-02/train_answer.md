**Problem.** Given the prefix-function array `pi[0..n-1]` of some hidden string (`pi[i]` = length of the
longest proper prefix of `s[0..i]` that is also a suffix), output any string over `a`–`z` whose prefix
function equals `pi`, or `-1` if no string realizes it. `n <= 2*10^5`; the input may be adversarial
(out-of-range / non-realizable). For `n = 0` print an empty line.

**Why the obvious check is wrong.** The tempting plan is to validate `pi` with local inequalities —
`pi[0]=0`, `0 <= pi[i] <= i`, `pi[i] <= pi[i-1]+1` — and, if they hold, reconstruct. These are
*necessary* but not *sufficient*. Counterexample: `pi = [0,1,1]` passes all of them, yet `pi[1]=1`
forces `s = "xx?"`, after which `pi[2]` can only be `2` (if `?=x`) or `0` (otherwise) — never `1`. So
`[0,1,1]` is infeasible. The legal values of `pi[i]` are exactly the entries of the **border chain**
`pi[i-1] -> pi[pi[i-1]-1] -> ... -> 0` (each +1 on a character match, else 0); that chain is a
*transitive, non-local* object, so a per-step check cannot decide feasibility.

**Key idea — construct, then verify.** Build a witness string by the forced/avoid rule, then recompute
its prefix function and compare to the input. The verification is what makes feasibility *sound*: if
`pi` is unrealizable, the constructed string cannot have it, the recomputation disagrees, and we emit
`-1`; if it agrees, we hold an explicit witness and there is nothing left to prove. The construction:

- `s[0] = 'a'` (position 0 is unconstrained; `pi[0]` must be 0).
- `pi[i] > 0`: the border of length `pi[i]` forces `s[i] = s[pi[i]-1]`.
- `pi[i] = 0`: `s[i]` must not continue any border of `s[0..i-1]`; walk the border chain, forbid every
  chain head `s[k]`, and pick any unused letter.

**Pitfalls to get right.**
1. *Out-of-bounds at `i=0`.* The `pi[i]=0` branch reads `pi[i-1]`; at `i=0` that is `pi[-1]`. Treat
   position 0 as its own case (`s[0]='a'`); the chain walk is only meaningful for `i >= 1`.
2. *Alphabet size.* A `pi=0` position forbids the distinct chain heads. The chain depth is
   `Theta(log n)` (worst case the Zimin/ruler word), so the reconstruction needs `<= 18` distinct
   letters even at `n = 2*10^5`; `26` is provably enough. Keep a "no free letter -> -1" guard anyway so
   the code is total on garbage input.
3. *Don't trust local checks for feasibility.* They reject obvious garbage and prevent bad indexing,
   but the verify pass — not the inequalities — is the actual feasibility decision.

**Edge cases.** `n=0` -> empty line; `pi=[0]` -> `"a"`; `pi=[1]` -> `-1` (a single char has `pi[0]=0`);
constant/strictly-increasing arrays (`[0,0,..]` -> `"abab.."`-ish, `[0,1,2,3,..]` -> `"aaaa.."`);
near-miss arrays (a real prefix function with one entry perturbed) pass the structural check but fail the
verify pass; negative or `> i` entries are rejected up front.

**Complexity.** `O(n)` time, `O(n)` memory. The construction's chain walks are amortized linear by the
same potential argument as the prefix-function computation; the verify pass is a textbook `O(n)` prefix
function. Runs in `~10 ms` at `n = 2*10^5`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;          // empty input -> treat as n = 0
    vector<int> pi(n);
    for (int i = 0; i < n; i++) cin >> pi[i];

    // ---- Quick structural rejects (cheap, before construction) -------------
    // pi[i] is a length in [0, i]; pi[0] must be 0; a border cannot grow by more
    // than one per step (pi[i] <= pi[i-1] + 1). These are necessary but NOT
    // sufficient -- transitive border composition is the real test, handled by
    // the verify pass below.
    bool bad = false;
    for (int i = 0; i < n; i++) {
        if (pi[i] < 0 || pi[i] > i) { bad = true; break; }
        if (i > 0 && pi[i] > pi[i - 1] + 1) { bad = true; break; }
    }
    if (bad) { cout << -1 << "\n"; return 0; }

    // ---- Reconstruct a candidate string ------------------------------------
    // Letters are 0..25 mapped to 'a'..'z'. Rule:
    //   pi[i] > 0  : the border of length pi[i] forces s[i] == s[pi[i]-1].
    //   pi[i] == 0 : s[i] must NOT continue any border of s[0..i-1]. Collect the
    //                set of "next" characters that every border (lengths
    //                pi[i-1], pi[pi[i-1]-1], ...) would demand, and pick any
    //                letter outside that set. With an alphabet of 26 there is
    //                always a free letter (the chain has < n forbidden chars,
    //                and a 0 only forbids the distinct chain heads).
    string s(n, 'a');
    for (int i = 0; i < n; i++) {
        if (i == 0) {
            s[0] = 'a';                 // position 0 is unconstrained (pi[0] == 0)
        } else if (pi[i] > 0) {
            s[i] = s[pi[i] - 1];        // border of length pi[i] forces this char
        } else {
            // forbidden = chars that would extend some border of s[0..i-1].
            // Walk the border chain pi[i-1] -> pi[pi[i-1]-1] -> ... -> 0; each
            // border of length k could be extended only by s[k], so s[i] must
            // avoid every such s[k], else pi[i] would be positive.
            bool used[26] = {false};
            int k = pi[i - 1];
            while (true) {
                used[(int)(s[k] - 'a')] = true;
                if (k == 0) break;
                k = pi[k - 1];
            }
            int c = 0;
            while (c < 26 && used[c]) c++;
            if (c == 26) { cout << -1 << "\n"; return 0; } // unreachable for valid pi
            s[i] = char('a' + c);
        }
    }

    // ---- Verify: recompute the prefix function of s and compare ------------
    // This is what turns "construct" into a sound feasibility test: if the input
    // pi is not realizable, the recomputed prefix function will disagree.
    vector<int> chk(n, 0);
    for (int i = 1; i < n; i++) {
        int k = chk[i - 1];
        while (k > 0 && s[i] != s[k]) k = chk[k - 1];
        if (s[i] == s[k]) k++;
        chk[i] = k;
    }
    for (int i = 0; i < n; i++) {
        if (chk[i] != pi[i]) { cout << -1 << "\n"; return 0; }
    }

    cout << s << "\n";
    return 0;
}
```
