The task is the maximum-sum non-empty contiguous run on a closed loop of `n` slots, where a run may wrap past the seam from slot `n-1` back to slot `0`. Two things decide the shape of the code before I even pick an algorithm. First the scale: `n <= 2*10^5` and `|a[i]| <= 10^9`, so a whole-loop sum reaches `2*10^5 * 10^9 = 2*10^14`, an order of magnitude past the 32-bit ceiling of `~2.1*10^9`. Every accumulator — running Kadane value, total, minimum — must be 64-bit `long long`; an `int` here is a silent wrong answer on the large tests. Second the wrap: it splits every run into exactly two shapes, and the boundary between them is where circular-subarray solutions usually die.

A non-empty circular run either does *not* cross the seam — an ordinary subarray `a[l..r]` — or it *does*, occupying a suffix `a[k..n-1]` glued to a prefix `a[0..m]`. The answer is the max of (best non-wrapping run) and (best wrapping run). That case split is the whole game.

For the non-wrapping half, plain Kadane: `cur = max(a[i], cur + a[i])`, `best = max(best, cur)`. For the wrapping half there is the standard trick: a wrapping run is the whole loop minus a contiguous chunk left out in the middle, and that dropped chunk is itself a non-wrapping subarray. So maximizing `total - gap` means minimizing `gap` over non-wrapping subarrays — compute `worst`, the minimum non-empty subarray sum (Kadane with min and `+` flipped), and the wrapping candidate is `total - worst`. The advertised answer is `max(best, total - worst)`, `O(n)` in two interleaved passes. The one thing I will not take on faith is whether `total - worst` is always a *legal* non-empty run under this contract.

On the sample `a = [5, -3, 5]` the wrap genuinely wins, so it exercises the mechanism. Kadane gives `best = 7` (the whole loop, read as `a[0..2]`), `total = 7`, and the minimum subarray is the single slot `{1} = -3`, so `worst = -3` and `total - worst = 10`. That is dropping slot 1 and shipping `{2,0} = 5+5 = 10`, the wrapping run; `max(7,10)=10`. The formula is sound when a real interior gap exists.

But `total - worst` is exactly the reduction that can misfire, and the suspicious family is the all-negative belt, where "leave out the worst chunk" may want to leave out *everything*. Take `a = [-3, -1, -4]`. Every run is negative; the largest is the single slot `-1` (any longer run is more negative: `{0,1}=-4`, `{1,2}=-5`, whole loop `-8`), so the true answer is `-1`. Now the formula: Kadane `best = -1`; `total = -8`; the minimum subarray is the *entire array* `{0,1,2} = -8`, so `worst = -8` and `total - worst = -8 - (-8) = 0`. The bare formula returns `max(-1, 0) = 0` — wrong.

The defect is exact. `worst = -8` is the whole belt, so `total - worst` means "ship the loop minus the whole loop" — the empty run, which sums to `0`. The empty selection is illegal here; the contract demands length `1..n`. This is the trap: whenever the minimum subarray swallows the entire belt, the wrap candidate silently degenerates into the empty set, and that happens precisely on all-negative (all-non-positive) belts.

The wrap candidate is legal only when the dropped gap is a proper, non-empty subarray, leaving at least one slot in the wrap. The exact signal that the gap is the whole belt is `worst == total`: the minimum non-empty subarray sum equals the sum of everything. When that holds I must drop the wrap option and take `best`, which — being plain Kadane on a non-empty array — is always a legal non-empty run (the least-negative single element survives). So:

```
if (worst == total) answer = best;                 // wrap would be empty -> illegal
else                answer = max(best, total - worst);
```

`worst == total` is the *direct* statement of the failure, not a proxy like "all elements negative"; it fires exactly when the entire belt is the minimum subarray, and nowhere else. On the broken belt `worst = -8 == total` sends `answer = best = -1`; on `[5,-3,5]` `worst = -3 != total = 7` leaves `10` unchanged.

Edge cases, where circular runs like to break:

- `n = 1`: wrapping and non-wrapping coincide. For `a = [-7]` the loop from `i=1` never runs, so `best = total = worst = -7`, the guard fires, and `answer = -7`. For `a = [9]`, `best = total = worst = 9`, the guard fires, `answer = 9` — right, because at `n=1` the wrap candidate `total - worst = 0` is the empty run and `best` is the lone slot. The guard's meaning ("don't let the wrap go empty") holds at `n=1` without any special-casing.
- All-positive, `a = [1,2,3]`: `best = total = 6`; `worst` is the smallest single element `1`, so `total - worst = 5` — dropping a positive chunk always shrinks below `total`, and `best` already holds the whole loop, so `max(6,5)=6`. The wrap can never exceed the full loop. Zeros collapse to the same case (`best=total=worst=0`, guard fires, `answer=0`).
- Overflow: the extreme `|total|` is `2*10^14`, and `total - worst` is at most `2*total = 4*10^14`, both far under the `long long` ceiling `~9.2*10^18`.
- I also guard `n == 0` (print `0`) so `a[0]` is never indexed out of bounds, though the contract guarantees `n >= 1`.

Two interleaved Kadane scans in one pass: `O(n)` time, `O(1)` extra state beyond the input array. At `n = 2*10^5` this is comfortably inside the 1-second limit.

So I ship the guarded circular Kadane rather than the bare textbook formula I just broke — one 64-bit pass for `best`, `worst`, `total`, then the `worst == total` guard around `max(best, total - worst)`. The full self-contained C++ program is the answer.

