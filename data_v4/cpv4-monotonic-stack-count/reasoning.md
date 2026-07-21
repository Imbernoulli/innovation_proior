Every subarray `a[l..r]` gets exactly one owner — the index of its minimum, ties going to the **leftmost** such position — and I want `S = sum_i i*c[i] mod (10^9+7)`, where `c[i]` counts the subarrays owned by `i` and `i` is 0-based. Two features of this problem decide everything before I pick an algorithm: the tie-break, and the scale.

Scale first, because it fixes my types. With `n <= 2*10^5` there are `n(n+1)/2 ~ 2*10^10` subarrays, and one index can own almost all of them, so a single `c[i]` is already past 32-bit. The weighted term `i*c[i]` reaches `~2*10^5 * 2*10^10 = 4*10^15` — comfortably inside `long long` (`~9.2*10^18`), nowhere near `int`. So counts live in 64-bit, and the modulus belongs on the final weighted sum, not on the raw counts I use to build the geometry.

One free oracle comes with the ownership definition: since every subarray has exactly one owner, `sum_i c[i] = n(n+1)/2` for any input. A `<`/`<=` convention that double-counts overshoots it, one that drops subarrays undershoots — so the classic monotonic-stack tie-break bug becomes visible instead of silent, and I lean on it at every step.

Brute force implements the definition directly: for each left endpoint `l`, sweep `r`, track the running minimum and its leftmost owner (move the owner only on a *strictly* smaller value), increment. `O(n^2)` — correct, but `~2*10^10` steps, so it survives only as a reference oracle. For speed I use the per-index rectangle: for a fixed `i` the subarrays it owns are `l in (L, i]`, `r in [i, R)`, giving `c[i] = (i-L)(R-i)` with `L`, `R` from nearest-smaller scans. The entire risk lives in the exact `<` vs `<=` on each side.

Fix `i`. For `a[i]` to own `[l, r]` (`l <= i <= r`) under leftmost-minimum:

1. nothing strictly less than `a[i]` in `[l, r]` — otherwise `a[i]` is not even a minimum there;
2. nothing equal to `a[i]` strictly *left* of `i` inside `[l, r]` — an equal element at `j < i` is a more-left minimum, so it owns, not `i`;
3. equal elements to the *right* of `i` are fine — `i` is still the leftmost occurrence of the minimum.

Going left I must stop at the first element `<= a[i]` (condition 1 forbids strictly-smaller, condition 2 forbids equal), so `L` is the nearest `j < i` with `a[j] <= a[i]` and `l in (L, i]` gives `i - L` choices. Going right I stop only at a strictly smaller element (condition 3 lets equals through), so `R` is the nearest `j > i` with `a[j] < a[i]` and `r in [i, R)` gives `R - i` choices. The convention is asymmetric: `<=` on the left, `<` on the right — precisely what credits each subarray spanning equal minima to its single leftmost owner. If no barrier exists, `L = -1` or `R = n`.

The tempting shortcut is the symmetric convention — nearest *strictly* smaller on both sides — and `[2,2]` shows in one line why it fails here. With both barriers strict, `L = [-1,-1]` and `R = [2,2]`, giving `c = [2,2]`, sum `4`. But the truth is `c = [2,1]`, sum `3 = 2*3/2`: the subarray `[0,1]` (min value 2, leftmost owner index 0) also gets claimed by index 1, because an equal element that blocks neither side lets both minima reach across it. The asymmetric `<=`-left barrier stops index 1's leftward rectangle at index 0, and the invariant is restored.

So: two stack passes. Left pass pops while `a[top] > a[i]` (it stops at `<=`), recording the survivor as `L` or `-1`; right pass, scanning down, pops while `a[top] >= a[i]` (it stops at `<`), recording `R` or `n`. Trace the sample `a = [2,1,2,1,3]` through them. Left pass gives `L = [-1,-1,1,1,3]` — note `i=3` (`a=1`) does *not* pop the equal `1` at index 1, so `L[3]=1` rather than `-1`, which is the `<=` barrier at work. Right pass gives `R = [1,5,3,5,5]` — at `i=1` the equal `1` at index 3 *is* popped, so `R[1]=5`. Then `c = [1, 8, 1, 4, 1]`, sum `15 = 5*6/2`, and the two `1`s split cleanly: index 1 takes 8, index 3 takes 4, no overlap. `S = 0*1 + 1*8 + 2*1 + 3*4 + 4*1 = 26`, matching the sample. The `0*1` term being zero is what pins the 0-based weight — an `i+1` weighting or a `1..n` loop would silently shift the whole answer.

Forming `S`, the accumulator must be `long long` reduced every iteration, not `int`. An `int ans` would evaluate `ans + i*cnt` before the `% MOD` and wrap: on the increasing array of length `2*10^5`, `c[i] = n - i` so `S = sum_i i(n-i) ~ n^3/6 ~ 1.3*10^15`, and even partial sums blow past `int`. I reduce `cnt` mod `p` first (`< 10^9`), multiply by `i % MOD` (`< 10^9`) for a product `< 10^18`, and reduce the running sum each step.

Edge cases:

- `n = 0`: the loop never runs, output `0` (zero subarrays).
- `n = 1`: the lone subarray sits at index 0, weight 0, so `S = 0`.
- all-equal `[5]*6`, the maximum-tie stress case: only index 0 reaches `L = -1`; every other `i` has `L = i-1` (the equal neighbor is a `<=` barrier) and `R = n` (no value below 5), so `c = [6,5,4,3,2,1]`, sum `21 = 6*7/2`, and `S = 0+5+8+9+8+5 = 35`, matching brute.
- strictly decreasing `[3,2,1]`: everything to the left is larger, so there is no `<=` barrier and `L[i] = -1`; `R[i] = i+1`; `c = [1,2,3]`, sum `6`.

The `sum c[i] = n(n+1)/2` invariant holds in each. The full module is in the answer.
