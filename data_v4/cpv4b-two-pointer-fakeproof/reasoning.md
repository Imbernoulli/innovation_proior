Counting windows `[l, r]` whose bitwise-OR `a[l] | ... | a[r]` is `<= K`, with up to `n = 2*10^5` masks of 31 bits each, two facts pin themselves before any algorithm. First the count: there are up to `n(n+1)/2 ~ 2*10^10` windows, past the 32-bit ceiling of `~2.1*10^9`, so the answer accumulator is `long long`; the masks and `K` are only 31 bits and fit `unsigned int`, and I read `K` into a wide type so `cur > K` never trips on width or sign. Second, the difficulty itself: the queried quantity is a bitwise-OR, and OR is *not* invertible — exactly where a sliding-window solution ships a confident, wrong bit identity the moment it shrinks the window.

**Why two pointers is legal.** OR-ing in another mask sets bits and clears none, so `OR(a[l..r+1]) = OR(a[l..r]) | a[r+1] >= OR(a[l..r])` bit by bit, hence as integers. So for a fixed right edge `r` the clean left edges form a contiguous suffix `[l*(r), r]`, and `l*(r)` only moves rightward as `r` grows: once some mask forced `l` right, a larger window never lets it back. That monotone frontier is the two-pointer invariant. Extend `r`, OR the new mask in, advance `left` while the window OR exceeds `K`; every start in `[left, r]` is clean, contributing `r - left + 1`.

**The removal step, and the tempting false identity.** Adding on the right is `cur |= a[r]`. The trap is removal: when `left` advances and `x = a[l]` leaves a window whose OR is `O`, what is the OR of the remaining masks? By analogy with sums (`new = old - x`) or with turning off `x`'s contribution, the one-line closed forms that suggest themselves are `O & ~x`, `O ^ x`, and the cautious `O ^ (x & O)`. Each looks right. I run the window `{6, 5}` through all three: `6 = 110`, `5 = 101`, `O = 111 = 7`; remove `6` and the truth is just `{5} = 101 = 5`. But `7 & ~6 = 1`, `7 ^ 6 = 1`, `7 ^ (6 & 7) = 1` — all three return `1`, not `5`. The reason is structural: bit `2` (value `4`) is set in **both** masks, so removing `6` must leave it on because `5` still carries it, yet every "subtract `x` out" form sees only `(O, x)` and cannot know another mask shared that bit. `(O, x)` does not carry "how many remaining masks still set this bit." OR has no inverse; there is no closed form in `O` and `x` alone.

**Per-bit counts.** The fix follows from the diagnosis: keep `cnt[b]` = the number of in-window masks with bit `b` set. Adding a mask increments the counts of its set bits, removing one decrements them, and bit `b` is in the window OR iff `cnt[b] > 0`. That carries exactly the shared-bit information `(O, x)` lost — removing `6` drops `cnt[2]` from `2` to `1`, so bit `2` correctly stays in the OR. Each move is `O(31)`, and each mask is added once and removed at most once, so the whole scan is `O(31 n)`.

**First cut of the scan.**

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

**The boundary the constraints invite.** A single oversized mask is where the shrink bites. If `a[right]` alone exceeds `K`, the loop must empty the window: for `n=1, K=3, a=[8]`, adding `8` gives `curOr=8>3`, removing it leaves `left=1`, `curOr=0`, and `ans += 0` — correct. An empty window has OR `0 <= K` (the contract gives `K >= 0`), so `while (curOr() > K)` on its own terminates before `left` passes `right`. I still fold in a `left <= right` guard so termination does not *depend* on the maintenance being flawless and `a[left]` can never be indexed past `right`: `while (left <= right && curOr() > K)`.

**Bit width.** `BITS = 31` covers bits `0..30`, and `2^31 - 1` sets exactly bits `0..30`, so 31 is the right span; `BITS = 30` would silently drop the top bit and undercount every OR that reaches it.

**Edge cases.** `n = 0`: the main loop never runs, `ans = 0`. `K = 0`: a window is clean iff its OR is `0`, i.e. all its masks are zero — no special case is needed, since any nonzero mask forces a shrink and only maximal zero-runs survive. `K = 2^31 - 1`: no OR can exceed it, the shrink never fires, `ans = n(n+1)/2`, which for `n = 2*10^5` is `2.00001*10^10` — the reason the accumulator is `long long`.

**Performance.** Each `right` step is an `O(31)` add plus `O(31)` OR reads, and total shrink steps across the run are at most `n`, so the work is `O(31 n)` — on the order of `10^7` operations at `n = 2*10^5`, comfortably under the 1 s limit. I keep the readable `curOr()` rather than caching the OR in a variable; the bound holds either way and correctness on the OR maintenance is worth more than the constant.

The finished scan, with the `left <= right` guard folded in, is the single self-contained C++ file in the answer.
