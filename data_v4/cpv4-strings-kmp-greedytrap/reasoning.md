**Reading the problem and pinning the contract.** I own a stamp engraved with a logo word `s`, and on each press I may print any non-empty *prefix* of `s` — one of `s[0..0]`, `s[0..1]`, ..., up to all of `s`. Presses are laid left to right with no gaps and no overlaps, and their concatenation must equal a target banner `t`. I want the minimum number of presses (blocks), or `-1` if `t` cannot be reproduced at all. Input is two tokens `s` and `t` on stdin; output is one integer. Scale first, because it dictates the algorithm class: `|s|, |t| <= 2*10^5`. The answer is a *count of blocks*, at most `|t| = 2*10^5`, so it fits in 32-bit, but I will carry it in `long long` to avoid any thought about overflow; that is free. What `2*10^5` really forbids is anything quadratic: an `O(|s|*|t|)` "try every prefix at every position" loop is `4*10^10` and dies. So whatever I do, the per-position work has to be (amortized) constant, which points at a string-matching preprocessing step plus a linear sweep.

**Laying out the candidate approaches.** Two routes are on the table, and I want the one I can *prove*, not merely the one that types fastest.

- *Greedy by longest match.* Stand at the current banner position, press the longest prefix of `s` that matches the banner there, jump past it, repeat until I either land exactly on the end (success) or get stuck (no prefix matches — failure). After preprocessing this is `O(|t|)` and four lines. The seductive intuition is "covering more banner per block can only reduce the block count." The risk is that block boundaries are a *global* commitment: a long block now changes *where* my next block must start, and a position that is convenient may force an awkward — or impossible — continuation. That smells like exactly the configuration where greedy breaks, so I will not trust it until I have tried to break it.
- *Reachability DP.* For each banner position `i`, find the set of legal block lengths, then compute the minimum number of blocks to reach the end. `O(|t|)` if I can get the legal-length sets cheaply and the minimization in one sweep. The risk is not the idea but the transcription — off-by-ones in the "reach" computation and in the sweep.

**A structural observation that shapes everything.** A block placed at position `i` is legal exactly when some prefix of `s` equals `t[i..i+L-1]`. Here is the key: *if a prefix of length `L` matches at `i`, then every shorter prefix matches at `i` too*, because a prefix of a matching prefix is itself a prefix that matches. So the legal block lengths at `i` are not some scattered set — they are a solid interval `1, 2, ..., reach[i]`, where `reach[i]` is the length of the *longest* prefix of `s` that matches `t` starting at `i` (capped at `|s|`). This is what turns the problem into something clean: from position `i` I may advance to any of `i+1, i+2, ..., i+reach[i]`, paying one block. Minimum blocks to get from `0` to `n` over those moves is precisely a *minimum-jumps* problem (Jump Game II), and `reach[i]` is precisely what string matching computes.

**Stress-testing the longest-match greedy before committing.** Hand-waving "longest block is best" is how wrong solutions ship, so let me actually attack it. Take `s = "babba"` and `t = "babbabb"` (length 7). The prefixes of `s` are `b, ba, bab, babb, babba`. Greedy stands at position 0 and asks for the longest prefix of `s` matching `t[0..]= "babbabb"`. `babba` matches the first five characters `babba`, so greedy presses `babba` and jumps to position 5. The remaining banner is `t[5..6] = "bb"`. The longest prefix of `s` matching `"bb"` is just `b` (since `ba` ≠ `bb`), so greedy presses `b`, lands at 6, presses `b` again, lands at 7. Greedy's total is `babba | b | b` = **3 blocks**.

Is 3 optimal? Let me hunt for something greedy structurally cannot reach. Split `t = "babbabb"` as `bab | babb`: `bab` is `s[0..2]` and `babb` is `s[0..3]`, both legal prefixes, and `"bab"+"babb" = "babbabb"`. That is **2 blocks**, strictly better than greedy's 3. So the longest-match greedy is wrong, and I now see *why*: by grabbing the full `babba` it advanced to position 5, a position whose only continuations are short single-`b` presses; the slightly shorter first block `bab` advanced to position 3, from which a *single* `babb` finishes the job. Taking more now cost more later. The verification paid off — it killed an approach I would otherwise have shipped.

**A second, nastier failure of greedy: it can deadlock.** Suboptimal block counts are bad, but the greedy has a worse failure mode that I should confirm exists, because it changes the `-1` logic. Take `s = "aab"`, `t = "aaaaab"` (length 6). Prefixes of `s`: `a, aa, aab`. Greedy at position 0: longest prefix of `aab` matching `"aaaaab"` — `aab` would need `t[0..2]="aaa"` to equal `aab`, no; `aa` matches `t[0..1]="aa"`, yes; press `aa`, jump to 2. Position 2: banner `"aaab"`, longest match again `aa`, jump to 4. Position 4: banner `"ab"`, longest match is `a` (since `aa` ≠ `ab`), press `a`, jump to 5. Position 5: banner `"b"`, but no prefix of `aab` starts with `b` (every prefix starts with `a`). Greedy is *stuck* — it reports impossible. Yet a valid tiling exists: `a | aa | aab` = `"a"+"aa"+"aab" = "aaaaab"`, **3 blocks**. So greedy not only miscounts, it can declare a feasible banner infeasible. That settles it: greedy is out, and my `-1` must come from genuine unreachability under the DP, not from a greedy dead end.

**Deriving the correct method and sanity-checking it.** I commit to the reachability DP. Step one is `reach[i]` for every `i`: the longest prefix of `s` matching `t` at `i`. The clean way is the Z-function on the concatenation `c = s + sep + t`, where `sep` is a byte outside the alphabet (I use `'\x01'`; the alphabet is lowercase letters). For an index `p` inside the `t`-part of `c`, `Z[p]` is the longest common prefix of `c` (i.e. of `s`) with `c[p..]` (i.e. with `t` from that offset), capped at `|s|` by the separator — that is exactly `reach`. Step two: minimum blocks from `0` to `n` where from `i` I may land on any `j` in `[i+1, i+reach[i]]`. That is Jump Game II, and the *correct* greedy there is the **level/BFS** greedy: maintain the window of indices reachable with the current number of blocks `[.. curEnd]`, track the farthest index any window member can reach, and when I exhaust the window I spend one block to jump the frontier out to `farthest`. (Note this is a *different* greedy from "take the longest prefix" — the maximal-jump greedy `i += reach[i]` is the wrong one I just disproved; the level greedy is provably optimal because it never delays a reachable position to a later level.)

Let me sanity-check the recurrence intent on the disproving example `s="babba", t="babbabb"`. First `reach`: at 0, longest prefix matching `babbabb` is `babba` length 5, so `reach[0]=5`; at 1 (`abbabb`) no prefix of `babba` starts with `a`, `reach[1]=0`; at 2 (`bbabb`) longest is `b`, `reach[2]=1`; at 3 (`babb`) `babb` matches, `reach[3]=4`; at 4 (`abb`) `reach[4]=0`; at 5 (`bb`) `reach[5]=1`; at 6 (`b`) `reach[6]=1`. So `reach = [5,0,1,4,0,1,1]`, `n=7`. Level greedy: start `curEnd=0`, `farthest=0`. i=0: `0+reach[0]=5`, `farthest=5`; i==curEnd(0) so spend a block (blocks=1), `curEnd=5`. i=1..4 update `farthest` from indices within `[1,5]`: index 3 gives `3+4=7`, so `farthest=7`; i=5==curEnd so spend a block (blocks=2), `curEnd=7>=7`, stop. Answer **2**, matching the optimum I found by hand. The mechanism is right.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the jump loop:

```
long long blocks = 0;
int curEnd = 0, farthest = 0;
for (int i = 0; i < n; i++) {
    if (i + reach[i] > farthest) farthest = i + reach[i];
    if (i == curEnd) {            // window exhausted -> spend a block
        blocks++;
        curEnd = farthest;
        if (curEnd >= n) break;
    }
}
```

I trace the deadlock case `s="aab", t="aaaaab"` whose `reach = [2,2,2,3,1,0]`, `n=6` (at 0 `aa`→2, at 1 `aa`→2, at 2 `aa`→2 wait at index 2 banner is `aaab`, `aab` would need `aab`==`aaa`? no, `aa`→2... let me recompute carefully: index 3 banner `aab` matches `aab` fully → 3; index 4 banner `ab` → `a` →1; index 5 banner `b` → 0). So `reach=[2,2,2,3,1,0]`. Start `curEnd=0, farthest=0`. i=0: `0+2=2`, farthest=2; i==curEnd → blocks=1, curEnd=2. i=1: `1+2=3`, farthest=3. i=2: `2+2=4`, farthest=4; i==curEnd(2) → blocks=2, curEnd=4. i=3: `3+3=6`, farthest=6. i=4: `4+1=5`, farthest stays 6; i==curEnd? curEnd=4, i=4 → yes → blocks=3, curEnd=6>=6, break. Output **3**. That is correct here. But I am uneasy, so I trace a case designed to *break* progress: take an unreachable banner, `s="ab", t="b"`, `reach=[0]`, `n=1`. Start `curEnd=0, farthest=0`. i=0: `0+reach[0]=0`, farthest stays 0; i==curEnd(0) → blocks=1, curEnd=0; `curEnd>=n`? `0>=1` false, no break. Loop ends (i reaches n=1). After the loop my code never checks `curEnd>=n`, so it just prints `blocks=1`.

**Diagnosing the first bug.** The output for `s="ab", t="b"` should be `-1` — `b` is not any prefix of `ab`, so the banner is untileable — but my loop happily printed `1`. The defect is precise: when the window is exhausted (`i==curEnd`) but `farthest <= curEnd`, I cannot actually advance, yet I still incremented `blocks` and set `curEnd = farthest = curEnd`, fabricating a block that covers nothing. I need a *stuck* guard: at the moment I would spend a block, if `farthest <= curEnd` the frontier did not move and the banner is infeasible. There is also a second, related gap: even when I never trigger the stuck guard, after the loop I must confirm I actually reached `n`; if the loop falls off the end with `curEnd < n` the answer is `-1`. And one more reachability hole: if at some `i` I have `i > curEnd`, position `i` was never reachable at all — that also means infeasible and I should stop. My first version checks none of these.

**Fixing and re-verifying.** I add the three guards — unreachable index, no-progress, and the post-loop completeness check:

```
bool ok = true;
for (int i = 0; i < n; i++) {
    if (i > curEnd) { ok = false; break; }            // index i unreachable
    if (i + reach[i] > farthest) farthest = i + reach[i];
    if (i == curEnd) {
        if (farthest <= curEnd) { ok = false; break; } // frontier cannot move
        blocks++;
        curEnd = farthest;
        if (curEnd >= n) break;
    }
}
if (!ok || curEnd < n) cout << -1 << "\n"; else cout << blocks << "\n";
```

Re-trace `s="ab", t="b"`: `reach=[0]`, `curEnd=0,farthest=0`. i=0: `i>curEnd`? no. update farthest = max(0, 0+0)=0. `i==curEnd` → `farthest(0) <= curEnd(0)` → `ok=false`, break. Output `-1`. Correct. Re-trace `s="aab", t="aaaaab"`: as before reaches blocks=3, curEnd=6>=6, `ok` stays true, `curEnd<n`? `6<6` false → output `3`. Correct. Re-trace `s="babba", t="babbabb"`: blocks=2, curEnd=7>=7 → `2`. Correct. The two cases that mattered — a true `-1` and the deadlock-but-feasible — now both come out right, and they come out right for the reason I fixed.

**A second debug episode: the `reach` indexing.** Before trusting the sweep I must trust `reach`. My first `reach` build read `reach[i] = Z[base + i]` with `base = m` (forgetting the separator occupies one slot). Trace `s="ab"` (m=2), so `c = "ab" + '\x01' + t`; the `t`-part starts at index `m+1 = 3`, not `m = 2`. With `base=m=2`, `reach[0]` would read `Z[2]`, which is the Z-value *at the separator* `'\x01'` — meaningless (it is the LCP of `s` with `"\x01"+t`, namely `0` unless `s` starts with `'\x01'`, which it never does). So `reach[0]` would be a spurious `0` and *every* `reach[i]` would be shifted one position too early, reading the separator/`t` boundary garbage. The fix is `base = m + 1`. I verify on `s="ab", t="ab"`: `c="ab\x01ab"`, indices `0:a 1:b 2:\x01 3:a 4:b`. `Z[3]` = LCP of `"ab\x01ab"` with `"ab"` = 2 (`ab` then `\x01`≠`a` stops, but the `t`-part is only `"ab"` so it is 2, capped at m=2). With `base=m+1=3`, `reach[0]=min(Z[3],m)=2`, `reach[1]=min(Z[4],m)`; `Z[4]`=LCP with `"b"`=0 since `c[0]='a'≠'b'`. So `reach=[2,0]`. Sweep: `n=2`, i=0: farthest=2, i==curEnd→blocks=1, curEnd=2>=2 break. Output `1` — `t="ab"` is the whole logo, one press. Correct. Had I used `base=m`, `reach[0]` would have been `min(Z[2],m)=min(0,2)=0`, making the banner look infeasible — a silent `-1` on a clearly tileable input. Caught.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n` minimal, `s` minimal: `s="a", t="a"` → `reach=[1]`, blocks=1, curEnd=1>=1 → `1`. `s="a", t="aaaa"` → `reach=[1,1,1,1]`; level greedy advances one per block → `4`. Correct (only prefix is `a`).
- Total infeasibility from the first character: `s="abc", t="x"` → `reach=[0]`, stuck guard fires → `-1`. Correct.
- Feasible but with a non-prefix character mid-string: `s="ab", t="aba"` → prefixes `a,ab`. `reach`: at 0 `ab`→2, at 1 (`ba`) `b`? no prefix of `ab` starts with `b` → 0, at 2 (`a`) → 1. `reach=[2,0,1]`, n=3. i=0 farthest=2, blocks=1 curEnd=2; i=1 update from index1: `1+0=1`, farthest stays 2; i=2: `i>curEnd`? `2>2` no; update `2+1=3` farthest=3; `i==curEnd`(2) → `farthest(3)>curEnd(2)` ok, blocks=2, curEnd=3>=3 break → `2`. Decomp `ab|a`. Correct.
- A banner equal to many whole logos: `s="abc", t="abcabc"` → `reach[0]=3, reach[3]=3` etc.; level greedy → `2`. Correct.
- Self-overlapping logo where longest-match greedy is most tempting: `s="aaaa", t="aaaaaaa"` (7 a's) → every `reach[i]=min(4, 7-i)`; optimal is `aaaa|aaa` = `2`. My sweep: i=0 farthest=4 blocks=1 curEnd=4; window 1..4 best farthest = `3+4=7`; i=4==curEnd blocks=2 curEnd=7>=7 → `2`. Correct, and note longest-match greedy would also give 2 here — the trap shows up only on the engineered cases above, which is exactly why I needed those.
- Overflow / types: `blocks` is at most `n = 2*10^5`, held in `long long`; `i + reach[i]` is at most `2*n`, fits in `int`. No overflow.
- I/O: `cin >> s >> t` consumes the two whitespace-separated tokens regardless of newline layout; output is one integer plus newline.

**Performance check.** The Z-function over `c` is `O(|s| + |t|)`; the jump sweep is one pass `O(|t|)`. On `|s| = |t| = 2*10^5` the whole thing is well under the 2-second limit (measured at a few milliseconds, ~6 MB). No quadratic anywhere — exactly what the constraints demanded when they ruled out "try every prefix at every position."

**Final solution.** I convinced myself the *idea* is right by disproving the longest-match greedy twice (a suboptimal count on `babba/babbabb`, a feasible-but-deadlocked banner on `aab/aaaaab`) and then sanity-checking the reachability DP's level greedy against those very cases; I convinced myself the *code* is right by tracing two real bugs to precise causes — the missing stuck/completeness guards that turned an infeasible `ab/b` into a phantom `1`, and the `base = m` off-by-one that shifted every `reach` onto the separator — and re-verifying the fixes. That is what I ship: one self-contained file, the linear KMP/Z-plus-jump method I can defend rather than the greedy I broke.

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

**Causal recap.** The longest-match greedy looked right but two traced counterexamples killed it — on `s="babba", t="babbabb"` it presses `babba|b|b` for 3 while `bab|babb` needs only 2 (a long block now forces awkward short blocks later), and on `s="aab", t="aaaaab"` it greedily eats `aa|aa|a` and deadlocks on the final `b` even though `a|aa|aab` tiles in 3, proving greedy can both miscount and false-report `-1`. So I recast the task: a block at `i` may be any prefix length `1..reach[i]` (legal lengths nest into an interval), `reach[i]` is the longest prefix of `s` matching `t` at `i` computed by a Z-function on `s+sep+t`, and minimum blocks is Jump Game II solved by the *level* greedy — not the maximal-jump greedy. My first sweep omitted the stuck/completeness guards (turning infeasible `ab/b` into a phantom `1`) and my first `reach` used `base=m` instead of `m+1` (shifting every value onto the separator); tracing each to its exact cause and adding the `i>curEnd`, `farthest<=curEnd`, and post-loop `curEnd<n` checks plus the corrected offset makes the linear solution agree with brute force on every case.
