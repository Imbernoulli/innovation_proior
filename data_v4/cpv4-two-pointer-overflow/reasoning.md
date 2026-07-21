I have `n` hourly releases `a[0..n-1]`, each strictly positive, and a cap `B` on the total discharged over any single contiguous run of hours. I want the contiguous block `a[l..r]` with the largest sum that still stays `<= B`; the empty block (discharge `0`) is always legal, so the answer never drops below `0`. Input is `n` and `B`, then the `n` values; I print one integer.

The first thing that shapes the solution here is not the algorithm but the arithmetic. `n` reaches `2*10^5` and each `a[i]` reaches `10^9`, so a full block sums to `2*10^5 * 10^9 = 2*10^14`, and `B` is allowed up to that same `2*10^14` precisely so that the "whole array fits" regime is reachable. Both dwarf the 32-bit signed ceiling near `2.147*10^9`. Every quantity that holds a sum — the running window total, the cap, the answer — has to be 64-bit. `int` everywhere compiles, passes the small samples, and then silently produces a wrapped answer on the large hidden tests; that failure is the whole reason this problem exists, and I will make it concrete once the algorithm is down.

I want the heaviest contiguous block with sum `<= B`. The brute-force route — for each start `l`, accumulate rightward and keep the largest partial sum still `<= B` — is unarguably correct but `O(n^2)`, which at `n = 2*10^5` is `4*10^10` operations, hopeless under a one-second limit. It survives only as a reference oracle to test against. The route I want is a two-pointer sliding window: keep a window `[left..right]` with running sum `cur`, step `right` forward one element at a time, and whenever `cur` exceeds `B`, advance `left` — dropping elements off the front — until `cur <= B` again. Each pointer only ever moves forward, so the whole sweep is `O(n)`.

That speed rests on two correctness claims: that the widest admissible window ending at `right` is also the *heaviest*, and that `left` never has to move backward. Both follow from positivity, `a[i] >= 1`.

Fix the right end at `right`. Sliding the left end leftward only adds positive terms, so `sum(a[left..right])` strictly increases as `left` decreases. Among blocks ending at `right` with sum `<= B`, the widest — the one with the smallest admissible `left` — therefore has the largest sum. So finding, for each `right`, the smallest `left` that keeps the sum `<= B` gives the best block ending there, and ranging `right` over all endpoints covers every contiguous block. And the left pointer is monotone: moving from `right` to `right+1` adds `a[right+1] > 0`, which only tightens the budget — any `left` that was inadmissible for `right` (sum `> B`) stays inadmissible for `right+1`, so the smallest admissible `left` never decreases. That is why `left` carries forward across iterations and the total work stays linear. The whole argument leans on `a[i] >= 1`; with zeros or negatives the monotonicity breaks and a forward-only window would be wrong, so I must not later "generalize" the code past its positivity assumption.

The implementation is a direct translation:

```
long long cur = 0, best = 0;
int left = 0;
for (int right = 0; right < n; right++) {
    cur += a[right];
    while (cur > B) {
        cur -= a[left];
        left++;
    }
    if (cur > best) best = cur;     // cur = sum(a[left..right]) <= B here
}
```

`best` starts at `0` so that if no nonempty block ever fits (because `B` is below every single release), the answer is the empty block's `0`; `cur` starts at `0` for the empty window.

Sample check: `n = 7`, `B = 15`, `a = [4, 2, 7, 3, 1, 6, 5]`, expected `15`. Running the recurrence by hand:

- `right=0`: `cur = 4`, window `[0..0]`, `best = 4`.
- `right=1`: `cur = 6`, window `[0..1]`, `best = 6`.
- `right=2`: `cur = 13`, window `[0..2]`, `best = 13`.
- `right=3`: `cur = 16 > 15`. Subtract `a[0]=4` -> `cur = 12`, `left = 1`, window `[1..3]`, `best = 13`.
- `right=4`: `cur = 13`, window `[1..4]`, `best = 13`.
- `right=5`: `cur = 19 > 15`. Subtract `a[1]=2` -> `17`, still `> 15`; subtract `a[2]=7` -> `10`, `left = 3`, window `[3..5]`, `best = 13`.
- `right=6`: `cur = 15 <= 15`, window `[3..6]`, `best = 15`.

Final `15`, from the block `[3,1,6,5]` at hours `3..6`, and `left` only ever climbs (`0 -> 1 -> 3`), exactly as the monotonicity argument requires.

Now the overflow, made concrete, since it is the trap this problem is built around. Take `int cur` — each `a[i] <= 10^9` fits an `int`, so "a sum is just an int" is the reflex — and run the regime that triggers it: `n = 5`, `B = 10^11`, `a = [10^9]*5`. The true answer is the whole block, `5*10^9`, comfortably `<= 10^11`, so nothing should ever shrink. But `cur` is 32-bit: at `right=2` the accumulation reaches `3*10^9`, past `INT_MAX = 2147483647`, and wraps to `3000000000 - 2^32 = -1294967296`. Now `while (cur > B)` compares a negative `int` against `B = 10^11` — false — so no shrink fires and `best` takes the garbage. Compiled and run, the all-`int` version prints `2000000000` where the 64-bit version prints `5000000000`: not a near-miss, a different number, and one that passes every sample with sums under `2.1*10^9`, detonating only on the large hidden tests. The fix is to hold `B`, the array, `cur`, and `best` in `long long`, where the running sum tops out near `2*10^14`, far inside the `~9.2*10^18` signed 64-bit ceiling. `left` and `right` stay `int`, bounded by `n <= 2*10^5`; only the sums need widening.

One corner in the shrink loop still needs closing: what happens when a single release exceeds `B` and `left` catches up to, or passes, `right`? Trace `n = 3`, `B = 1`, `a = [5, 9, 2]` — every release exceeds `1`, so the answer must be `0`. At `right=0`, `cur = 5 > 1`, subtract `a[0]` -> `cur = 0`, `left = 1`; the window `[1..0]` is empty and `cur = 0 <= 1`, so `best = 0`. At `right=1`, `cur = 9 > 1`, subtract -> `cur = 0`, `left = 2`. At `right=2`, `cur = 2 > 1`, subtract -> `cur = 0`, `left = 3`. Final `0`. The invariant is that I only subtract while `cur > B`, with `cur = sum(a[left..right])`; when `left == right` and `cur = a[right] > B`, subtracting sets `left = right+1` and `cur = 0`, and since `B >= 0` the condition `cur > B` is now false, so the loop stops. `left` never advances past `right+1` and never reads `a[left]` outside the valid range. So even in the all-over-budget case the window empties safely and the answer stays at the empty block's `0`.

The remaining edges fall out of the same code. `n = 0` runs both loops zero times and prints `0`. `n = 1` with the release over `B` shrinks to the empty window and prints `0`; with the release exactly at `B`, `cur = B <= B` and it prints `B`. `B` at least the whole-array total never triggers a shrink and prints the full sum — `2*10^14` at the extreme, the headline overflow case. `cin >>` skips arbitrary whitespace, so the one-line-vs-two-line input layout parses either way, and the output is a single integer plus newline.

For evidence beyond the hand traces I ran the two-pointer against the `O(n^2)` brute force on a thousand-plus randomized small cases — `n` including `0` and `1`, value caps varied, `B` swept across `[0, total + slack]` so the empty-window, exact-fit, and whole-array regimes all appear, plus small-`n` batches with values near `10^9` to exercise the 64-bit arithmetic. The brute force re-sums every interval rather than sliding a window, so it is an independent oracle; zero mismatches.

The full self-contained module — the `O(n)` window plus the stdin/stdout wrapper — is in the answer.
