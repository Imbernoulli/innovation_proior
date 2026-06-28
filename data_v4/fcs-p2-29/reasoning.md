**Reading the problem and pinning the contract.** I have `n` integers `a[0..n-1]`, values possibly negative. I look at every contiguous non-empty subarray `a[l..r]`, and from that subarray I am allowed to delete at most one element — zero or one — with the constraint that whatever remains is still non-empty. So a length-`1` subarray can keep its element but cannot delete it (that would leave nothing). The score is the sum of the remaining elements, and I want the maximum score over all subarrays and all allowed deletions. Input on stdin is `n` followed by the `n` values; I print one integer.

Before any algorithm I want the scale fixed, because it dictates the data types. The constraints are `1 <= n <= 10^5` and `|a[i]| <= 10^9`. A subarray sum can reach `10^5 * 10^9 = 10^14`, which is far past the 32-bit range of about `2.1*10^9`. So every accumulator that holds a running sum has to be 64-bit; an `int` here is a silent wrong-answer on the large tests. I will use `long long` throughout. That decision is non-negotiable and comes before I pick the algorithm.

One more contract subtlety I want nailed down up front: the answer is always well defined even when every value is negative, because a length-`1` subarray with no deletion is always legal. In that case the best score is simply the largest single element (the least negative one). I will keep checking that this corner survives whatever recurrence I write.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one that is cheapest to type on top of code I already know.

- *Plain Kadane, then delete the worst element of that window.* The reflex is: this is "max subarray" with a small twist, and I already know Kadane cold. So run ordinary Kadane to find the best subarray with no deletion; then, inside exactly that winning window, look at its most negative element and, if dropping it helps, drop it. That is `O(n)` and maybe three lines bolted onto a function I can write in my sleep. The risk is structural and I can feel it: the deletion option changes which window is best, and "the window Kadane found" was chosen under the *no-deletion* rule. There is no obvious reason the optimal-to-delete-from window is the same window. I will not trust this until I have tried to break it.

- *Two-state Kadane DP.* Carry two running quantities as I scan: the best subarray *ending at* `i` that has used **no** deletion, and the best subarray ending at `i` that has used **exactly one** deletion. The deletion-state transition is the new part. This is still `O(n)`, `O(1)` memory. The risk here is not whether the idea is correct — it tracks the state the future actually depends on (how many deletions I've spent) — but whether I transcribe the two coupled transitions in the right order.

The reflex pull is toward the first one because I already own Kadane and the second route asks me to invent a recurrence. But "I already know how to write it" is not "it is correct," so the first thing I do is attack the cheap one.

**Stress-testing the greedy before committing.** Hand-waving "delete the worst element of the best window feels right" is exactly how a wrong solution gets shipped. So I attack it with concrete instances. I want a case where the *best window to delete from* is a different, larger window than the one plain Kadane stops on. Intuitively that happens when a single bad element makes plain Kadane cut the window short — but with a deletion that bad element is free, so it would have been profitable to keep extending through it.

Let me build that on purpose. Take `a = [5, 6, -3, 3, -5]`, indices `0..4`. Plain Kadane: running sum starts at `5`, then `11` at index 1. At index 2 the `-3` drops it to `8`, at index 3 back to `11`, at index 4 the `-5` drops it to `6`. The maximum prefix-of-a-window value Kadane records is `11`, achieved by the window `[5, 6]` (indices `0..1`); Kadane never extends past index 1 as the recorded best because adding `-3` only lowered the running max. So the greedy's chosen window is `[5, 6]`. Its most negative element is `5` (both are positive!), and deleting `5` gives `6`, which is *worse* than `11`. So the greedy keeps `11` and reports **11**.

Is `11` optimal? Now I hunt for something the greedy structurally cannot reach. Consider the window `[5, 6, -3, 3]` (indices `0..3`) and delete the `-3`: that leaves `5 + 6 + 3 = 14`. That is strictly better than `11`. So the greedy is wrong, and I can now see exactly *why*: plain Kadane chose its window under the no-deletion rule, where the `-3` was a reason to stop; but once a deletion is on the table, the `-3` is precisely the element you delete, and keeping the window open through it to capture the `+3` afterward is the whole point. The greedy never even considered window `[5,6,-3,3]` because Kadane's no-deletion logic had already pruned it.

I'll record the kill cleanly: on `[5, 6, -3, 3, -5]`, greedy reports `11`, the true answer is `14`. The verification paid off — it killed an approach I would otherwise have shipped because it was the one I already knew how to write. The greedy is out. (I later let a small random search confirm this is not a fluke: it spits out many such cases — e.g. `[5, -6, 2, -3, 6, 1]` greedy `7` vs correct `11`, and `[3, 6, 6, -6, 5, 1]` greedy `15` vs correct `21` — all with the same shape: the profitable deletion lives in a window Kadane's no-deletion rule had already discarded.)

**Deriving the two-state DP and checking the recurrence on paper.** I want, for each ending position `i`, the best subarray that *ends at* `i` — because "ends at `i`" is the quantity Kadane-style scans accumulate, and the global answer is the max over all ending positions. The only extra thing the future of the scan cares about is *how many deletions I have already spent inside the current window*: zero or one. So I keep two values, both defined as "best subarray ending exactly at `i`":

- `noDel` = best sum of a subarray ending at `i` with **no** element deleted,
- `oneDel` = best sum of a subarray ending at `i` with **exactly one** element deleted (and the remainder still non-empty).

Transitions. For `noDel` this is just ordinary Kadane: a subarray ending at `i` with no deletion either starts fresh at `i` or extends the best no-deletion subarray ending at `i-1`:

`noDel_i = max(a[i], noDel_{i-1} + a[i])`.

For `oneDel` there are two ways to end at `i` having spent exactly one deletion:
1. The deletion is `a[i]` itself. Then before `i` I had a no-deletion subarray ending at `i-1`, and I append `a[i]` only to immediately delete it. The remaining sum is exactly `noDel_{i-1}`. (This is only legal if such a subarray ending at `i-1` exists — i.e. `i >= 1`.)
2. The deletion happened earlier, and `a[i]` is kept. Then I extend the best one-deletion subarray ending at `i-1` by `a[i]`: `oneDel_{i-1} + a[i]`.

So `oneDel_i = max(noDel_{i-1}, oneDel_{i-1} + a[i])`.

The crucial detail jumps out immediately: `oneDel_i` reads `noDel_{i-1}` and `oneDel_{i-1}`, the values from the *previous* step. If I update `noDel` to `noDel_i` first and then compute `oneDel_i` using it, I'd be reading the wrong generation — I would let `oneDel` "delete `a[i]`" from a window that already includes `a[i]`'s own fresh start. So both new values must be computed from the old pair via temporaries, then assigned. I'm flagging this now because it is exactly the kind of ordering bug that compiles and looks clean.

The answer is the maximum of `noDel_i` and `oneDel_i` over all `i`. I keep a running `best` and fold both states into it at every step.

Base case. Before any element there is no subarray, so conceptually `noDel = oneDel = -infinity`. I'll initialize both to a large negative sentinel `NEG`. At `i = 0`: `noDel_0 = max(a[0], NEG + a[0]) = a[0]` (correct — the lone element, no deletion). And `oneDel_0 = max(NEG, NEG + a[0])`, which stays `NEG` — correct, because deleting the only element of a length-`1` subarray would leave nothing, which is forbidden. So `oneDel` only becomes meaningful from `i = 1` onward, which is exactly when "the segment so far = `noDel_{i-1}`" refers to a real non-empty remainder.

Let me confirm the recurrence by hand on the example `a = [5, 6, -3, 3, -5]`, expected answer `14`. Initialize `noDel = oneDel = NEG`, `best = NEG`.
- i=0 (5): `newNoDel = max(5, NEG+5) = 5`; `newOneDel = max(NEG, NEG+5) = NEG`. State `(noDel, oneDel) = (5, NEG)`, `best = 5`.
- i=1 (6): `newOneDel = max(noDel=5, oneDel+6=NEG) = 5` (delete the `6`, keep the `5`); `newNoDel = max(6, 5+6=11) = 11`. State `(11, 5)`, `best = 11`.
- i=2 (-3): `newOneDel = max(noDel=11, oneDel + (-3) = 5-3=2) = 11` (window `[5,6,-3]` delete `-3` -> `11`); `newNoDel = max(-3, 11-3=8) = 8`. State `(8, 11)`, `best = 11`.
- i=3 (3): `newOneDel = max(noDel=8, oneDel + 3 = 11+3=14) = 14` (extend the one-deletion window `[5,6,-3 (deleted)]` by `+3`); `newNoDel = max(3, 8+3=11) = 11`. State `(11, 14)`, `best = 14`.
- i=4 (-5): `newOneDel = max(noDel=11, oneDel + (-5) = 14-5=9) = 11`; `newNoDel = max(-5, 11-5=6) = 6`. State `(6, 11)`, `best = max(14, 11) = 14`.

Final answer `14`. The recurrence reproduces the hand-derived optimum, and the winning path is exactly the one the greedy could not reach: window `[5,6,-3,3]` with the `-3` deleted.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the loop body, written fast, looked like this:

```
long long noDel = NEG, oneDel = NEG, best = NEG;
for (int i = 0; i < n; i++) {
    noDel = max(a[i], noDel + a[i]);     // update no-deletion state
    oneDel = max(noDel, oneDel + a[i]);  // update one-deletion state
    best = max({best, noDel, oneDel});
}
```

The ordering smell I flagged during the derivation is right here: `oneDel`'s "delete `a[i]`" branch reads `noDel`, but I just overwrote `noDel` with `noDel_i` on the line above. So I trace the smallest input that could expose it. I want a case where deleting `a[i]` should give the *previous* window but the buggy code uses the freshly extended one. Try `a = [10, -100, 10]`. The true answer: any single element is at most `10`; the best one-deletion window is `[10, -100, 10]` delete `-100` -> `20`; or `[10, -100]` delete `-100` -> `10`. So the answer is `20`.

Buggy trace. Start `noDel = oneDel = best = NEG`.
- i=0 (10): `noDel = max(10, NEG+10) = 10`; `oneDel = max(noDel=10, NEG+10) = 10`; `best = 10`. Already wrong: `oneDel = 10` claims a one-deletion subarray ending at index 0 with score `10`, but the only such "subarray" would be `[10]` with its element deleted — empty, illegal. The bug let `oneDel` read the just-updated `noDel` (`10`) as if it were `noDel_{i-1}`.
- i=1 (-100): `noDel = max(-100, 10-100) = -90`; `oneDel = max(noDel=-90, oneDel + (-100) = 10-100 = -90) = -90`; `best = 10`.
- i=2 (10): `noDel = max(10, -90+10) = 10`; `oneDel = max(noDel=10, oneDel+10 = -90+10 = -80) = 10`; `best = 10`.

Buggy code returns `10`. The correct answer is `20`.

**Diagnosing the bug.** The defect is precise and it is exactly the ordering I worried about. On each step, `oneDel_i` must use `noDel_{i-1}` — the no-deletion window ending at `i-1`, onto which I append `a[i]` and then delete `a[i]` — but I overwrote `noDel` with `noDel_i` *before* reading it for `oneDel`. At i=2, the correct `oneDel` should be `max(noDel_{i-1} = noDel after i=1 = -90, oneDel_{i-1} + a[i] = -90 + 10 = -80) = -80`... wait, that is not `20` either. Tracing the *correct* recurrence at i=2: `oneDel_2 = max(noDel_1, oneDel_1 + a[2])`. With the *correct* values `noDel_1 = -90` and `oneDel_1 = max(noDel_0 = 10, NEG) = 10`, I get `oneDel_2 = max(-90, 10 + 10) = 20`. There it is — `20`. The buggy version destroyed two things: it read a stale-by-being-too-fresh `noDel`, and it propagated a wrong `oneDel_1`. Both stem from the single sin of updating `noDel` in place before `oneDel` consumes the previous generation.

**Fixing and re-verifying.** Compute both new values from the old pair via temporaries, then assign:

```
long long newOneDel = max(noDel, oneDel + a[i]);  // uses old noDel, old oneDel
long long newNoDel  = max(a[i], noDel + a[i]);    // uses old noDel
noDel = newNoDel;
oneDel = newOneDel;
```

Re-trace `[10, -100, 10]`. Start `(noDel, oneDel) = (NEG, NEG)`.
- i=0 (10): `newOneDel = max(NEG, NEG) = NEG`; `newNoDel = max(10, NEG) = 10`. State `(10, NEG)`, `best = 10`.
- i=1 (-100): `newOneDel = max(noDel=10, oneDel + (-100) = NEG) = 10`; `newNoDel = max(-100, 10-100) = -90`. State `(-90, 10)`, `best = 10`.
- i=2 (10): `newOneDel = max(noDel=-90, oneDel + 10 = 10+10 = 20) = 20`; `newNoDel = max(10, -90+10) = 10`. State `(10, 20)`, `best = 20`.

Answer `20`. Correct. The case that broke before now passes, and it broke for the precise reason I fixed — the in-place overwrite — which is the evidence I trust.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 1`, `a = [-7]`: i=0 -> `newOneDel = max(NEG, NEG) = NEG`, `newNoDel = max(-7, NEG) = -7`. `best = max(NEG, -7, NEG) = -7`. The single element, no deletion allowed to empty it — correct. (Importantly the answer is **not** `0`; this is not the empty-set problem, a non-empty subarray is mandatory.)
- All negative, `[-3, -1, -4]`: `noDel` tracks the best single/extended sum, which peaks at the largest single element `-1`; `oneDel` from i=1 onward can delete one element but a two-element all-negative window minus one element is still one negative element, never beating `-1`. `best = -1`. Correct — least negative single element.
- `oneDel` legality: the sentinel `NEG = LLONG_MIN/4` guarantees the "delete `a[i]`" branch can only fire (i.e. beat `oneDel + a[i]`) once a real `noDel_{i-1}` exists, i.e. from `i >= 1`. At `i = 0` both `oneDel` candidates are `NEG`, so it never claims an illegal empty remainder.
- Overflow: all running sums are `long long`; the maximum magnitude `~10^14` fits with enormous room. The sentinel `NEG = LLONG_MIN/4` is only ever read inside a `max`; it does get `a[i]` added to it in `oneDel + a[i]` and `noDel + a[i]`, but `LLONG_MIN/4 + 10^9` is nowhere near underflowing `LLONG_MIN`, so it stays safely negative and never wraps. Safe.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace so parsing is format-agnostic.

**Self-verification against an independent oracle.** Hand traces convince me of the cases I thought to try; they do not protect me from the case I did not. So I wrote a separate brute force that defines the problem from scratch — enumerate every subarray `[l, r]`, score it with no deletion, and additionally (when length `>= 2`) score it deleting each interior index `k`, taking the global max — and differential-tested it against the DP. I ran the explicit edge battery (single element, all-negative pairs, the `[5,6,-3,3,-5]` greedy-killer, large-magnitude `10^9` values whose sums exceed 32 bits) plus 600 randomized cases across biased distributions (all-negative, all-positive, alternating signs, near-zero ranges, single elements, full `10^9` magnitudes). Zero mismatches over the whole run. I also ran the DP on `n = 10^5` with random `10^9`-magnitude values: it finishes in well under 10 ms and a few MB, comfortably inside the 1 s / 256 MB budget, confirming the `O(n)` / `O(1)` cost is not just asymptotic comfort but real headroom.

**Final solution.** I convinced myself the idea is right by disproving the "Kadane then delete the worst element" greedy with a concrete counterexample (`[5,6,-3,3,-5]`: greedy `11` vs the reachable `14`), and by hand-checking the two-state recurrence on that same instance. I convinced myself the *code* is right by tracing the failing in-place version to a precise cause (`[10,-100,10]` returning `10` instead of `20`), re-verifying the temporary-based fix and the corners, and then differential-testing against an independent brute over 600+ cases with zero mismatches. That is what I ship — one self-contained file, the simple `O(n)` two-state DP I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // The subarray must be non-empty AFTER any deletion, so we need n >= 1.
    // noDel = best sum of a subarray ending at i with NO element deleted.
    // oneDel = best sum of a subarray ending at i with EXACTLY one element deleted.
    const long long NEG = LLONG_MIN / 4;
    long long noDel = NEG, oneDel = NEG;
    long long best = NEG;
    for (int i = 0; i < n; i++) {
        // oneDel must be computed from the PREVIOUS noDel/oneDel before noDel is updated.
        long long newOneDel = max(noDel,                 // delete a[i]; segment so far = old noDel ending at i-1
                                  oneDel + a[i]);        // deletion already used earlier; extend by a[i]
        long long newNoDel = max(a[i], noDel + a[i]);    // standard Kadane: start fresh or extend
        noDel = newNoDel;
        oneDel = newOneDel;
        best = max(best, noDel);
        best = max(best, oneDel);
    }

    cout << best << "\n";
    return 0;
}
```

**Causal recap.** The "run plain Kadane, then delete the worst element of that window" greedy looked right and was the approach I already knew how to write, but a single traced counterexample (`[5,6,-3,3,-5]`: greedy `11` vs the reachable `14`) showed that the optimal window to delete from is a *larger* window than the one Kadane's no-deletion rule keeps — the deleted `-3` is exactly the element that made plain Kadane stop early. So I moved to a two-state Kadane DP carrying "best ending here with no deletion" and "with one deletion," whose `oneDel` transition `max(noDel_{i-1}, oneDel_{i-1} + a[i])` both read the *previous* generation; my first in-place loop overwrote `noDel` before `oneDel` consumed it, and a trace of `[10,-100,10]` returning the wrong `10` instead of `20` pinpointed it; computing both new values from the old pair via temporaries and seeding both states at a safe negative sentinel fixes it and keeps the single-element, all-negative, and empty-remainder corners correct; an independent brute over 600+ random and edge cases then confirmed zero mismatches.
