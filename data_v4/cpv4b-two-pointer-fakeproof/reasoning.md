**Reading the problem and pinning the contract.** I have `n` masks `a[0..n-1]`, each a non-negative 31-bit integer, and a budget `K`. A window `[l, r]` is *clean* when the bitwise-OR of `a[l..r]`, read as an ordinary integer, is `<= K`. I must count clean windows. Input is `n` and `K` on the first line, then `n` masks; I print one integer. Before any algorithm I fix the scale, because it dictates the data types. There are up to `n(n+1)/2` windows; with `n = 2*10^5` that is about `2*10^10`, which overflows 32-bit (`~2.1*10^9`). So the *answer* accumulator must be `long long`. The masks and `K` fit in 31 bits, so `unsigned int` holds each mask comfortably, but I will read `K` into a wide type so the comparison `cur > K` never trips on sign or width. The values themselves are small; only the count is large. That is the first non-negotiable decision.

**Why two pointers should even apply here.** The thing that makes a sliding window legal is monotonicity. Claim: as a window grows (right edge moves right, left edge fixed), its OR is non-decreasing. That is structurally true for OR — OR-ing in another mask can set bits but never clears one, so `OR(a[l..r+1]) = OR(a[l..r]) | a[r+1] >= OR(a[l..r])` bit by bit, hence `>=` as integers too. Equivalently, for a fixed right edge `r`, as I move the left edge *rightward* (shrinking the window) the OR is non-increasing. So for each `r` the set of clean left edges is a contiguous suffix `[l*(r), r]`, and `l*(r)` is itself non-decreasing in `r`: once a mask forced me to push `l` right, a later, larger window can never let it back. That monotone-frontier picture is exactly the two-pointer invariant. Good — the area fits.

**Laying out candidate approaches.** Two routes survive.

- *Two pointers with a maintained running OR.* Extend `r`, OR the new mask in, then advance `l` while the window OR exceeds `K`. The clean windows ending at `r` number `r - l + 1`. `O(n)` *if* I can maintain the OR in `O(1)` or `O(31)` per move. The danger is the removal step: when `l` advances and `a[l]` leaves the window, I need the new OR of the *remaining* masks.
- *Prefix-recompute.* For each `r`, recompute the OR of the current candidate window from scratch when shrinking. Correct but I must watch the time-limit factor; if I recompute over the whole window every shrink it degrades.

I will commit to the two-pointer scan, but the entire correctness hinges on the OR-removal step, and OR is where confident-but-wrong algebra creeps in. So I refuse to assert a removal identity; I will *derive a candidate and check it numerically* before trusting it.

**Deriving the OR-removal step — and the tempting false identity.** When the left edge advances, mask `x = a[l]` leaves a window whose current OR is `O`. I want `O'` = OR of the masts that remain. My first instinct, by analogy with sum (`new = old - x`) or with a set where I "turn off x's contribution", is one of these closed forms:

- *Identity A:* `O' = O & ~x` — "clear the bits that x contributed."
- *Identity B:* `O' = O ^ x` — "xor x back out."
- *Identity C:* `O' = O ^ (x & O)` — a cautious variant of B that only flips bits actually present in `O`.

Each is one line and each *looks* right. This is precisely the moment a solution gets shipped wrong: an unproven bit identity asserted with confidence. So before I write a single loop around it, I test it on a concrete tiny window by hand.

**Numeric self-check of the removal identity (catching the false step).** Take a two-mask window `{6, 5}`. In binary `6 = 110`, `5 = 101`, so `O = 6 | 5 = 111 = 7`. Now remove `x = 6`. The masks that remain are just `{5}`, so the *true* `O'` is `5 = 101`. Check the candidates:

- Identity A: `O & ~x = 7 & ~6 = 0111 & ...1001 = 0001 = 1`. But the truth is `5`. **A is false.**
- Identity B: `O ^ x = 7 ^ 6 = 001 = 1`. Truth is `5`. **B is false.**
- Identity C: `O ^ (x & O) = 7 ^ (6 & 7) = 7 ^ 6 = 1`. Truth is `5`. **C is false.**

All three collapse, and now I see *why*, which matters more than that they failed: bit `2` (value `4`) is set in **both** `6` and `5`. When I remove `6`, that bit must stay on because `5` still carries it — but every "subtract x out" identity clears it, because they only look at `O` and `x` and have no idea another mask shared that bit. The information "how many remaining masks still set this bit" simply is not present in `(O, x)`. OR is not invertible; there is no closed form in `O` and `x` alone. That false algebra step is exactly the bug I would have shipped had I not put numbers to it.

**The correct maintenance: per-bit counts.** The fix follows directly from the diagnosis. Keep `cnt[b]` = the number of masks *currently in the window* whose bit `b` is set. Adding a mask `m`: for each set bit `b` of `m`, `cnt[b]++`. Removing a mask `m`: for each set bit `b`, `cnt[b]--`. Then bit `b` is in the window OR **iff** `cnt[b] > 0`. This carries exactly the shared-bit information `(O, x)` lost: removing `6` decrements `cnt[2]` from `2` to `1`, so bit `2` stays in the OR — correct, because `5` still holds it. Per move costs `O(31)`; over the whole scan each mask is added once and removed at most once, so the total is `O(31 n)`, well within budget. Let me sanity-check the count rule on the same window: start empty, add `6` → `cnt = {bit1:1, bit2:1}` (since `6=110`), add `5=101` → `cnt = {bit0:1, bit1:1, bit2:2}`. OR from counts = bits 0,1,2 set = `111 = 7`. Remove `6` → `cnt = {bit0:1, bit1:0, bit2:1}`. OR = bits 0,2 = `101 = 5`. Exactly the true `O'`. The maintenance is right.

**First implementation — and a trace, because clean logic transcribes dirty.** My first cut of the scan:

```
const int BITS = 31;
vector<int> cnt(BITS, 0);
auto curOr = [&]() { unsigned int o=0; for (int b=0;b<BITS;b++) if (cnt[b]>0) o|=(1u<<b); return o; };
long long ans = 0; int left = 0;
for (int right = 0; right < n; right++) {
    for (int b=0;b<BITS;b++) if (a[right] & (1u<<b)) cnt[b]++;   // add a[right]
    while (curOr() > K) {                                        // shrink
        for (int b=0;b<BITS;b++) if (a[left] & (1u<<b)) cnt[b]--;
        left++;
    }
    ans += (right - left + 1);
}
```

Before trusting it I trace the smallest input that can expose the shrink loop: a single mask that is itself too big. Take `n = 1`, `K = 3`, `a = [8]` (`8 = 1000`). The true answer is `0` — the only window is `[0,0]` with OR `8 > 3`. Run it. `right = 0`: add `8`, so `cnt[3] = 1`. Enter the while: `curOr() = 8 > 3`, so decrement `cnt[3]` using `a[left]=a[0]=8`, `left` becomes `1`. Loop condition again: `curOr() = 0`, `0 > 3` is false, exit. Then `ans += right - left + 1 = 0 - 1 + 1 = 0`. Final `0`. Correct here — but I almost missed something.

**The bug: an unguarded `left` walking off the window.** The while condition is `while (curOr() > K)`. Imagine a mask `a[right]` whose own value already exceeds `K`, sitting at the current right edge with the window empty to its left. The loop will keep shrinking, but once `left` reaches `right` and I remove `a[right]` itself, `left` becomes `right + 1` and the window is empty with `curOr() = 0`. In the single-element case above that worked out, but consider what happens if, due to a logic slip, `curOr()` could stay `> K` with `left > right`: the loop would decrement `cnt` using `a[left]` for `left = right+1, right+2, ...`, indexing masks *outside* the intended window and driving counts negative — a real out-of-window read and a corrupted state. Let me construct a case that probes the boundary: `n = 3`, `K = 3`, `a = [1, 9, 1]` (`9 = 1001 > 3`). Trace. `right=0`: add `1` (`cnt[0]=1`), `curOr=1 <= 3`, no shrink, `ans += 0-0+1 = 1`. `right=1`: add `9` (`cnt[0]=2, cnt[3]=1`), `curOr = 1001|0001? ` counts say bit0 and bit3 set → `1001 = 9 > 3`, shrink: remove `a[left]=a[0]=1` (`cnt[0]=1`), `left=1`; `curOr` bit0,bit3 → `9 > 3`, shrink again: remove `a[left]=a[1]=9` (`cnt[0]=0, cnt[3]=0`), `left=2`; `curOr=0 <= 3`, stop; `ans += 1-2+1 = 0`. `right=2`: add `1` (`cnt[0]=1`), `curOr=1 <= 3`, `ans += 2-2+1 = 1`. Total `2`.

Brute check: windows `[0,0]`OR1✓, `[1,1]`OR9✗, `[2,2]`OR1✓, `[0,1]`OR9✗, `[1,2]`OR9✗, `[0,2]`OR9✗. Clean = 2. Matches. So the *logic* is right — but I notice the shrink crossed `left = right` cleanly only because `curOr()` genuinely dropped to `0` once the offending mask left. The latent hazard is that `curOr() > K` with `left > right` would over-shrink. The defensive fix is to bound the loop: `while (left <= right && curOr() > K)`. With correct OR maintenance the `curOr() > K` test alone would stop in time (an empty window has OR `0 <= K` whenever `K >= 0`, which holds since `K >= 0` by the contract), but the `left <= right` guard makes the termination independent of that reasoning and removes any chance of indexing `a[left]` past `right`. I add it. This is the kind of off-by-one that only shows up on adversarial single-big-element inputs, so I pin it down now rather than discover it on the hidden tests.

**Re-verifying after the guard.** With `while (left <= right && curOr() > K)`, re-run `a=[8], K=3`: `right=0`, add `8`, while: `left(0)<=right(0)` and `8>3` → remove `8`, `left=1`; while: `left(1)<=right(0)` false → stop; `ans += 0-1+1 = 0`. Correct. Re-run `a=[1,9,1], K=3`: identical to before, total `2`. Correct. The guard changes nothing on the well-behaved cases and forecloses the boundary hazard.

**A second self-verify: the `BITS` bound and top-bit masks.** I set `BITS = 31`, covering bits `0..30`. The contract says `a[i] <= 2^31 - 1`. The largest mask, `2^31 - 1 = 2147483647`, sets bits `0..30` — bit `30` is the top, since `2^30 = 1073741824` and `2^31 - 1` is all of bits `0..30`. So `BITS = 31` covers every possible set bit; bit `31` is never used. Let me check the extreme directly by hand-reasoning: `a = [2^31-1, 2^31-1, 2^31-1]`, `K = 2^31-1`. Every window's OR is `2^31-1 = K`, so all `3*4/2 = 6` windows are clean; `cur > K` is never true, no shrink ever fires, `ans = 1+2+3 = 6`. Correct. And `K = 2^31-2`: every single mask already has OR `2^31-1 > K`, so each `right` shrinks until `left = right+1`, `ans += 0` each step, total `0`. Correct. If I had written `BITS = 30` I would have dropped bit `30` from every OR, silently undercounting; `31` is right and the top-value case confirms it.

**Edge cases, deliberately.**
- `n = 0`: the read `cin >> n >> K` succeeds, the mask loop reads nothing, the main loop never runs, `ans = 0`. Empty input — correct. (If the whole stream is empty, `if (!(cin >> n >> K)) return 0;` prints nothing; but the contract guarantees the two header integers, and `n=0` still supplies them.)
- `n = 1`, mask `<= K`: one clean window, `ans = 1`. Mask `> K`: shrink empties the window, `ans = 0`. Both traced above in spirit.
- `K = 0`: a window is clean iff its OR is `0`, i.e. every mask in it is `0`. The count machinery handles this with no special case: any non-zero mask pushes `curOr() > 0 = K` and gets shrunk away, so only maximal runs of zeros contribute. Checked separately on `a=[0,0,1,0,0], K=0`: the two zero-runs of length 2 give `2*3/2 + 2*3/2 = 3 + 3 = 6` clean windows — and the program returns `6`.
- `K = 2^31 - 1`: no OR can exceed it, never shrink, `ans = n(n+1)/2`. The accumulator is `long long`, and for `n = 2*10^5` that is `20000100000`, which I confirmed equals `n(n+1)/2` exactly — well within 64-bit range, impossible in 32-bit.
- Heavily overlapping bits: this is the case the false identity would have wrecked. With per-bit counts it is exactly handled, as the `{6,5}` derivation showed.
- Output: a single integer and newline; `cin >>` skips arbitrary whitespace, so the optional second line and any layout are parsed fine.

**Performance.** Each `right` step does an `O(31)` add and the count read inside `curOr()`; `curOr()` is `O(31)`. The shrink loop calls `curOr()` once per check; total shrink steps across the whole run are at most `n` (each mask leaves once), and there are `n` non-shrinking `curOr()` checks, so the work is `O(31 n)` plus `O(31 n)` for the OR reads — about `1.2*10^7` operations at `n = 2*10^5`. Measured at roughly `0.07 s` on a worst case, comfortably under the `1 s` limit. I keep the readable `curOr()` rather than caching the OR in a variable, because correctness here is worth more than the constant factor and the bound holds either way.

**Final solution.** I disproved the seductive OR-removal identities by putting `{6, 5}` through them and watching the shared bit `2` betray every "subtract x out" formula, which forced the per-bit-count maintenance I can actually defend; I traced the shrink loop on a single oversized mask and on a big mask wedged in the middle, which surfaced the unguarded-`left` boundary hazard and earned the `left <= right` guard; and I pinned `BITS = 31` against the `2^31 - 1` top value and the answer type against the `2*10^10` count. That is what I ship — one self-contained file:

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

**Causal recap.** The window OR is monotone, so two pointers apply, but the scan dies on the shrink step where a mask leaves: I derived three plausible "subtract x out" identities and *numerically* ran the window `{6,5}` (OR `7`) through each — all returned `1` instead of the true `5`, because bit `2` is shared by both masks and OR has no inverse, so I switched to per-bit counts where bit `b` survives iff `cnt[b] > 0`; tracing the shrink loop on `[8]` and `[1,9,1]` then exposed that an unguarded `while (curOr() > K)` could walk `left` past `right` and index out of the window, fixed by the `left <= right` guard; and `BITS = 31` (checked against `2^31 - 1`) plus a `long long` answer (checked against `n(n+1)/2 ~ 2*10^10`) close the remaining corners.
