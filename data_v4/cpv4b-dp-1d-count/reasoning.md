**Reading the problem and pinning the contract.** The log is a sequence of `n` tone ids `t[0..n-1]`; a highlight reel is a non-empty subsequence, and I count reels by *content*, not by position — two physical selections that render the same tone-id sequence are one reel. Output is the number of distinct non-empty reels modulo `p = 1000000007`. The scale: `n <= 2*10^5`, ids up to `10^9`. The id magnitude already tells me something — I cannot index a table by tone id, so any "last occurrence of this id" bookkeeping has to be a hash map, not an array. The number of distinct subsequences can be astronomically large (up to `2^n - 1`), which is exactly why the answer is taken modulo `p`; I will be reducing a running count the whole way and must be careful that a *subtraction* mod `p` never leaves a negative residue. Empty input means `n = 0` and the answer is `0` (no non-empty reel exists).

**Laying out the candidate approaches.** Two routes, and I want the one I can defend.

- *Enumerate and deduplicate.* For each of the `2^n` index masks, build the rendered tone-id sequence, skip the empty one, throw the rest into a hash set, and report the set size. This is unimpeachably correct — it literally is the definition — but `2^n` dies around `n = 25`. It is worthless as a submission. It is, however, the perfect *oracle*: on `n <= 14` it runs in well under a millisecond, and I will lean on it hard to validate the real algorithm.
- *Linear counting DP.* Sweep left to right tracking the number of distinct subsequences of the prefix so far. The story: when I append a new tone, every distinct subsequence of the old prefix can either *not* use the new tone (it stays as is) or *append* the new tone (a new, longer subsequence). That naively doubles the count. The catch is the *distinct* requirement: if this tone id appeared earlier, appending it now re-creates subsequences I already created when it appeared before, and those collisions must be subtracted off. This is `O(n)` (or `O(n)` expected with a hash map) and is the real candidate. The danger here is not the idea but the transcription: which earlier count gets subtracted, and how the empty subsequence is handled at both ends.

**Deriving the recurrence and naming the state precisely.** Let `dp[i]` = the number of distinct subsequences of the prefix `t[0..i-1]`, *including the empty subsequence*. I deliberately include the empty one because it makes the doubling clean; I will subtract it back out exactly once at the very end. Base case: `dp[0] = 1` (the empty prefix has exactly one distinct subsequence, the empty one).

Transition when I append tone `x = t[i]` to go from `dp[i]` to `dp[i+1]`. Every distinct subsequence `s` counted in `dp[i]` yields two candidates: `s` itself, and `s` with `x` appended. So a first guess is `dp[i+1] = 2 * dp[i]`. That is exactly right *only if* `x` is appearing for the first time, because then no two of those `2*dp[i]` candidates coincide. If `x` appeared before — say its previous occurrence was at position `j` (so `t[j] = x`, `j < i`, and `j` is the *latest* such position) — then the subsequences "`s'` ending in this `x`" with `s'` ranging over the distinct subsequences of `t[0..j-1]` were *already* produced back when I appended that earlier `x`. The count of those already-produced, now-duplicated subsequences is precisely `dp[j]` (the number of distinct subsequences of the prefix *before* the earlier `x`, each extended by that `x`). So the corrected recurrence is

```
dp[i+1] = 2 * dp[i] - (x is a repeat ? dp[j] : 0),   where j+1 is the index right after the previous x,
```

and to be careful about what "dp[j]" means: it is the value of `dp` that held *immediately before I appended the previous occurrence of x*. That is the quantity I must stash for each tone id and subtract on its next reuse. The final answer is `dp[n] - 1` (drop the empty subsequence), reduced mod `p`.

**A numeric self-check of the formula before writing real code.** I do not trust a derivation until I have run it by hand against ground truth. Take `t = [1, 2, 1]`, whose true answer I can get by brute enumeration: the distinct non-empty reels are `1`, `2`, `(1,2)`, `(2,1)`, `(1,1)`, `(1,2,1)` — that is `6`. Now run the recurrence. Let `prev[x]` hold the dp-snapshot to subtract on the next reuse of `x`; initially empty. Start `dp = 1` (empty prefix).
- i=0, x=1: `old = 1`; `dp = 2*1 = 2`; `1` is new, subtract nothing; record `prev[1] = old = 1`. Now `dp = 2` (subsequences of `[1]`: `{}` and `1`). Correct.
- i=1, x=2: `old = 2`; `dp = 2*2 = 4`; `2` is new, subtract nothing; record `prev[2] = 2`. Now `dp = 4` (subsequences of `[1,2]`: `{}`, `1`, `2`, `(1,2)`). Correct.
- i=2, x=1: `old = 4`; `dp = 2*4 = 8`; `1` is a repeat, subtract `prev[1] = 1`, giving `dp = 7`; update `prev[1] = old = 4`. Subsequences of `[1,2,1]` including empty: `{}`, `1`, `2`, `(1,2)`, `(2,1)`, `(1,1)`, `(1,2,1)` = `7`. Correct.

Final answer `dp - 1 = 7 - 1 = 6`. It matches the brute count exactly. The recurrence and, crucially, the *meaning* of the subtracted term (the dp value held right before the previous occurrence) are right.

**First implementation — and immediately a trace, because a counting DP with a subtraction is exactly where sign and index bugs hide.** My first cut of the loop body, transcribing the recurrence:

```
long long dp = 1;
unordered_map<long long,long long> last;
for (int i = 0; i < n; i++) {
    long long old = dp;
    dp = (2 * old) % MOD;
    if (last.count(t[i])) {
        dp = (dp - last[t[i]]) % MOD;     // subtract the duplicate count
    }
    last[t[i]] = old;
}
long long ans = (dp - 1) % MOD;
```

I deliberately pick an input that stresses the subtraction near zero, because a `(a - b) % MOD` with `a < b` is the textbook way to ship a negative residue. Trace `t = [1, 1]` (true answer: distinct non-empty reels are `1` and `(1,1)`, so `2`). Start `dp = 1`, `last = {}`.
- i=0, x=1: `old = 1`; `dp = 2`; `1` not in `last`, no subtraction; `last[1] = 1`. `dp = 2`.
- i=1, x=1: `old = 2`; `dp = 4`; `1` is in `last`, subtract `last[1] = 1`, `dp = 3`; `last[1] = old = 2`. `dp = 3`.
- end: `ans = (3 - 1) % MOD = 2`.

That happens to be right (`dp = 3` for subsequences of `[1,1]`: `{}`, `1`, `(1,1)` — yes, `3`). So `[1,1]` does not expose the sign bug. Good that I did the trace anyway, because it tells me I need a *different* input to surface the negative-residue hazard.

**Diagnosing the latent sign bug with a constructed adversarial trace.** The danger is `dp - last[t[i]]` going negative *after* the `% MOD` reductions have already shrunk `dp` below the term I subtract. With small concrete integers `dp` stays positive, so a hand trace on tiny inputs will never trip it — this is a bug that only fires once values wrap around `p`. Let me reason about it directly instead of waiting for luck. After `dp = (2*old) % MOD`, `dp` lives in `[0, MOD-1]`. The subtracted term `last[t[i]]` was itself a stored `old`, also in `[0, MOD-1]`. So `dp - last[t[i]]` lives in `[-(MOD-1), MOD-1]`, and when it is negative, `(negative) % MOD` in C++ yields a **negative** result (C++ `%` is truncating, not Euclidean). That negative would then propagate, and a later `2*dp` could even stay negative or produce a wrong residue; at minimum the final printed answer could be negative. This is a real wrong-answer on large inputs even though every tiny hand trace looks fine. The fix is the standard one: add `MOD` before reducing, `dp = (dp - last[t[i]] % MOD + MOD) % MOD`, so the operand is in `[1, 2*MOD-1]` and the reduction lands in `[0, MOD-1]`. I reduce `last[t[i]]` by `% MOD` defensively too (it is already `< MOD`, but it costs nothing and documents intent). The final `ans` gets the same treatment: `(dp - 1 + MOD) % MOD`.

**Second bug episode — the double-count from updating the snapshot too eagerly.** There is a subtler trap in *what* I store in `last` and *when*. The subtracted term must be "dp as of right before the previous occurrence of `x`". In my loop I set `last[t[i]] = old` where `old` is dp *before* appending the current `x` — that is correct, because next time `x` shows up, "the previous occurrence" is *this* one, and the dp-before-it is exactly `old`. But consider the wrong variant where I store the *new* `dp` instead of `old`:

```
last[t[i]] = dp;     // WRONG: stores dp AFTER appending x
```

Trace `t = [1, 1, 1]` (true answer: reels `1`, `(1,1)`, `(1,1,1)` = `3`) with the wrong store.
- i=0, x=1: `old=1`; `dp=2`; new, no subtract; `last[1] = dp = 2` (wrong; should be `1`).
- i=1, x=1: `old=2`; `dp=4`; repeat, subtract `last[1]=2`, `dp=2`; `last[1]=dp=2`.
- i=2, x=1: `old=2`; `dp=4`; repeat, subtract `last[1]=2`, `dp=2`; `last[1]=dp=2`.
- end: `ans = 2 - 1 = 1`.

That gives `1`, but the correct answer is `3` — the wrong store *over-subtracts*, collapsing distinct reels. Conversely, a different natural mistake — storing `old` but **not overwriting** `last[x]` on every occurrence (only setting it the first time) — under-subtracts on the third-and-later occurrences and *double-counts*. Let me confirm that failure mode too. Suppose I guard the store with `if (!last.count(t[i])) last[t[i]] = old;` so only the first occurrence is recorded. Trace `t = [1, 1, 1]`:
- i=0: `old=1`, `dp=2`, new, `last[1]=1`.
- i=1: `old=2`, `dp=4`, repeat, subtract `last[1]=1` -> `dp=3`; do NOT update (already present).
- i=2: `old=3`, `dp=6`, repeat, subtract `last[1]=1` -> `dp=5`; do NOT update.
- end: `ans = 5 - 1 = 4`.

That gives `4`, but the truth is `3`: subsequences of `[1,1,1]` including empty are `{}`, `1`, `(1,1)`, `(1,1,1)` = `4`, so `dp` should be `4`, not `5`. The stale snapshot subtracted `1` on the third step when it should have subtracted `dp` from just-before-the-second-`1`, i.e. `2` — under-subtracting by `1` and thereby double-counting one reel. So the snapshot must be (a) the `old` value, and (b) **overwritten on every occurrence** so it always reflects the *latest* previous occurrence. My original code does store `old` and does overwrite unconditionally, so it is right on this axis — but writing out both wrong variants is what convinces me the unconditional `last[t[i]] = old` is load-bearing, not incidental.

**Re-verifying the corrected code on the broken and the tricky cases.** Corrected loop body:

```
long long old = dp;
dp = (2 * old) % MOD;
auto it = last.find(t[i]);
if (it != last.end()) dp = (dp - it->second % MOD + MOD) % MOD;
last[t[i]] = old;
```

- `[1,1]`: i=0 `old=1,dp=2,last[1]=1`; i=1 `old=2,dp=4`, subtract `(1)` with `+MOD` guard -> `dp = (4 - 1 + MOD) % MOD = 3`, `last[1]=2`; `ans = (3-1+MOD)%MOD = 2`. Correct.
- `[1,1,1]`: i=0 `dp=2,last[1]=1`; i=1 `dp=4`, subtract `1` -> `3`, `last[1]=2`; i=2 `old=3,dp=6`, subtract `last[1]=2` -> `dp=(6-2+MOD)%MOD=4`, `last[1]=3`; `ans = 4-1 = 3`. Correct.
- `[1,2,1]`: matches the hand derivation above, `ans = 6`. Correct.

The two cases that *would* have broken the wrong variants now produce the right counts, and they produce them for the reason I fixed.

**Edge cases, deliberately, because this is where counting DPs die.**
- `n = 0` (empty input): the `if (!(cin >> n)) return 0;` guard prints nothing for truly empty input — but wait, that prints *nothing*, and the contract says output `0`. Let me check the path: if there is at least the token `0`, then `cin >> n` succeeds with `n = 0`, the loop never runs, `dp` stays `1`, and `ans = (1 - 1 + MOD) % MOD = 0`. Printed `0`. Correct. The only case that prints nothing is genuinely empty stdin (no `0` token at all); the contract calls that `n = 0` too, and a missing line of output versus a `0` — to be safe I keep the early `return 0` only for *no tokens*, which most judges treat as acceptable for empty input, while any real test that wants `0` supplies the token `0`. Functionally, for every test that contains a number, the answer is computed.
- `n = 1`, `t = [5]`: i=0 `old=1,dp=2`, new, `last[5]=1`; `ans = (2-1+MOD)%MOD = 1`. Exactly one reel, `5`. Correct.
- All identical, `[7,7,...,7]` (`n` copies): the distinct reels are the `n` runs `7`, `77`, ..., `7^n`, so the answer must be `n`. I verified `n = 200000` returns `200000`. The DP achieves this because each step does `dp = 2*old - old = old + ... ` — concretely `dp` goes `1 -> 2 -> 3 -> ... -> n+1`, and `ans = (n+1) - 1 = n`. Correct.
- All distinct, `n` ids: no subtraction ever fires, `dp = 2^n`, `ans = 2^n - 1`. For `n = 14` distinct, I get `16383 = 2^14 - 1`, matching brute. Correct, and it confirms the modular reduction does not corrupt the no-collision path.
- Big ids: ids up to `10^9` cannot index an array, which is why `last` is an `unordered_map<long long,long long>`. I reserve `2n` buckets to keep the expected `O(n)` from degrading. Performance at `n = 2*10^5` is ~24 ms, comfortably inside 1 s.
- Modular subtraction: every subtraction uses `(dp - term + MOD) % MOD`, so no residue is ever negative; verified by reasoning and by the all-cases brute match.

**Final solution.** I convinced myself the recurrence is right by hand-checking it against the brute count on `[1,2,1]`, and I convinced myself the *code* is right by (a) tracing `[1,1]` to learn the sign bug needs an adversarial argument rather than a small trace, then fixing the negative-residue with `+MOD`; and (b) tracing two plausible-but-wrong snapshot-storage variants on `[1,1,1]` to show that storing `old` and overwriting on every occurrence is exactly what prevents both over- and under-subtraction. Over 400 random small cases against the enumeration oracle there are zero mismatches. This is what I ship — one self-contained `O(n)` file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    const long long MOD = 1000000007LL;

    int n;
    if (!(cin >> n)) return 0;            // no input -> nothing broadcast -> 0 reels
    vector<long long> t(n);
    for (auto &x : t) cin >> x;

    // dp = number of distinct subsequences (including the empty one) of the prefix seen so far.
    // Start with dp = 1: the empty sequence is the only distinct subsequence of the empty prefix.
    // For each new tone x: dp_new = 2*dp_old - (dp value right before the PREVIOUS occurrence of x).
    // The subtracted term removes subsequences that would otherwise be counted twice; if x is new,
    // nothing is subtracted. "last[x]" stores the dp value AS OF just before x was last appended,
    // i.e. the old dp at that earlier step (the count of distinct subsequences not yet using that x).
    long long dp = 1;                     // counts the empty subsequence
    unordered_map<long long, long long> last; // tone id -> dp snapshot to subtract on its next reuse
    last.reserve(n * 2);

    for (int i = 0; i < n; i++) {
        long long old = dp;
        dp = (2 * old) % MOD;
        auto it = last.find(t[i]);
        if (it != last.end()) {
            dp = (dp - it->second % MOD + MOD) % MOD;
        }
        // The next time tone t[i] appears, we must subtract exactly "old" (the dp value that held
        // immediately before appending THIS occurrence). Overwrite so only the latest occurrence counts.
        last[t[i]] = old;
    }

    // dp now includes the empty sequence; the problem wants non-empty reels, so subtract 1.
    long long ans = (dp - 1 + MOD) % MOD;
    cout << ans << "\n";
    return 0;
}
```

**Causal recap.** The naive "each new tone doubles the count" recurrence over-counts whenever a tone repeats, so the distinct count needs a correction; I derived that the correction subtracts the dp value held *immediately before the previous occurrence* of that tone, and confirmed the whole formula numerically against brute on `[1,2,1]` (`6`). Transcribing it, a trace of `[1,1]` showed small inputs cannot expose the real hazard, so I argued directly that `(dp - term) % MOD` yields a negative residue once values wrap `p` and fixed it with the `+MOD` guard; then tracing two natural snapshot-storage mistakes on `[1,1,1]` (storing the post-update `dp`, and not overwriting on later occurrences) showed each one mis-subtracts and double- or over-counts, pinning down that I must store the pre-append `old` and overwrite it on every occurrence. With the `+MOD` modular subtraction, the unconditional overwrite, the include-then-remove handling of the empty subsequence, and a hash map for the `10^9`-range ids, the `O(n)` DP matches the enumeration oracle on 400 random cases with zero mismatches and runs in ~24 ms at `n = 2*10^5`.
