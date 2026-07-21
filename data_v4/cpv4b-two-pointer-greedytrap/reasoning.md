Two cranes clear a row of `n` containers: a left crane peels off a prefix of `i` containers, a right
crane peels off a suffix of `j`, the two ranges cannot overlap (`i + j <= n`), and — the part that
makes this more than two independent problems — they burn from *one* shared tank, so
`pref[i] + suf[j] <= B`, where `pref[i]` is the sum of the first `i` weights and `suf[j]` the sum of
the last `j`. I want to maximize the count `i + j`. The coupling is the entire difficulty: every unit
the left crane spends is a unit the right crane can no longer have.

The sizes settle the data types before any algorithm does. `n <= 2*10^5` and `w[i] <= 10^9` put a
prefix sum as high as `2*10^5 * 10^9 = 2*10^14`, and `B` itself reaches `10^18` — both far past
32-bit's `~2.1*10^9`. So every weight, every prefix/suffix sum, and `B` are `long long`. The answer is
a count `<= n` and fits `int`, but I keep it in `long long` until the final print so nothing narrows
mid-arithmetic. I will only ever form `B - pref[i]` after checking `pref[i] <= B`, which keeps that
difference nonnegative and rules out underflow.

The tempting solution is a cheaper-end greedy: two pointers at the ends, repeatedly lift whichever
end's next container is lighter while it fits, stop when neither does. To maximize *count* you want to
spend the least fuel per container, so grab the cheapest available — three lines, `O(n)`. But the tank
is shared and global while this rule decides locally, exactly the shape where a greedy count-maximizer
can fail, so I look for a counterexample. Take `w = [3, 4, 6, 6, 3, 4]`, `B = 10`. The greedy
lifts `w[0] = 3`, then (a tie against the right's `4`) `w[1] = 4`, spending `7`; now both ends cost
more than the leftover `3`, so it halts at **2** containers. But leaving `w[1]` alone, the left takes
only `w[0] = 3` and the right takes the last two, `w[5] = 4` and `w[4] = 3`, for `3 + 7 = 10 <= B` and
**3** containers. Burning `4` on `w[1]` starved the tank of exactly the cheap `w[4] = 3` on the far
side — the locally cheapest step is not the cheapest path to a large count when one tank feeds both
ends. Greedy is out.

The handle that works: fix the prefix length `i`. The prefix is then forced to be `w[0..i-1]` with
cost `pref[i]`, and the only remaining freedom is the suffix length `j`, which I want as large as
possible subject to fuel `suf[j] <= B - pref[i]` and no-overlap `j <= n - i`. Since weights are
positive, both `pref` and `suf` are strictly increasing, so for a fixed `i` the feasible `j` form a
prefix `0..J(i)` and the answer is `max_i (i + J(i))`. Binary-searching each `J(i)` is `O(n log n)`; I
want `O(n)`, and that rests on `J(i)` being **non-increasing** in `i`: raising `i` grows `pref[i]`
(shrinking the fuel cap) and shrinks `n - i` (tightening the overlap cap), and a tighter cap can only
lower the largest feasible `j`, never raise it. So one pointer that starts at `j = n` and only ever
slides inward covers the whole sweep, with total movement `<= n`.

On the sample `w = [1, 7, 7, 8, 2, 8]`, `B = 11`, this reads out concretely: `pref = [0, 1, 8, 15,
23, 25, 33]`, `suf = [0, 8, 10, 18, 25, 32, 33]` (last-to-first `8, 2, 8, 7, 7, 1`). For `i = 0`,
leftover `11`, largest `j` with `suf[j] <= 11` is `2`; for `i = 1`, leftover `10`, `suf[2] = 10` still
fits so `2`; for `i = 2`, leftover `3`, even `suf[1] = 8` is too big so `0`. `J = 2, 2, 0` is
non-increasing, and `max(i + J(i)) = 1 + 2 = 3` — the sample answer.

The math is settled; the pointer direction is the easy place to get backwards. My first cut grew the
suffix pointer from `0`:

```
int j = 0;
for (int i = 0; i <= n; i++) {
    if (pref[i] > B) break;
    while (j < n - i && suf[j + 1] <= B - pref[i]) j++;
    if (j > n - i) j = n - i;
    best = max(best, (long long)i + j);
}
```

Growing `j` while the feasible `j` is meant to *shrink* with `i` is backwards, and the smallest case
exposes it. On `w = [2, 5]`, `B = 6` (`pref = [0, 2, 7]`, `suf = [0, 5, 7]`; the true answer is `1`,
since one container of weight `5` or the single `2` is all `B` affords): at `i = 0` the loop grows `j`
to `1` (`suf[1] = 5 <= 6`) and records `best = 1`. At `i = 1` the leftover fuel has dropped to `4`, so
the feasible suffix length should drop to `0` — but a grow-only pointer has no way to walk back down;
it keeps the stale `j = 1`, the overlap check `1 > n - i = 1` is false so nothing clamps it, and it
records `1 + 1 = 2` for the pair `pref[1] + suf[1] = 2 + 5 = 7 > 6`, which is infeasible. A grow-only
pointer is valid only when the feasible bound is non-decreasing; here it is non-increasing, so the
pointer must run the other way. I flip it: start `j = n`, clamp to the overlap bound first, then shrink
while the fuel cap is violated.

```
int j = n;
for (int i = 0; i <= n; i++) {
    if (pref[i] > B) break;
    if (j > n - i) j = n - i;                    // no-overlap bound first
    while (j > 0 && suf[j] > B - pref[i]) j--;    // shrink until the suffix fits
    best = max(best, (long long)i + j);
}
```

Re-tracing `[2, 5]`, `B = 6`: `i = 0` shrinks `j` from `2` to `1` (`suf[2] = 7 > 6`), `best = 1`;
`i = 1` clamps `j` to `1`, then shrinks to `0` since `suf[1] = 5 > 4`, `best` stays `1`; `i = 2`
breaks on `pref[2] = 7 > 6`. Answer `1`, and the pointer moved `2 -> 1 -> 0`, only downward — the
amortized `O(n)` holds.

The overlap clamp on that middle line is the sole enforcer of `i + j <= n`. Drop it and a loose budget
lets the prefix and suffix each grab everything. On
`w = [1, 1, 1, 1]`, `B = 100` the fuel never bites, so without the clamp every `i` keeps `j = 4` and
`i = 4` reports `4 + 4 = 8` — each container counted once as prefix and again as suffix. With the
clamp, `i = 1` first sets `j = min(4, 3) = 3`, every `i` yields exactly `4`, and the answer is `4`.
That is why the clamp comes *before* the fuel shrink.

The corners the sweep has to survive:
- `n = 0`: only `i = 0` runs, `j` clamps to `0`, `best = 0`.
- `n = 1`, `w = [3]`: with `B = 5`, `i = 0` gives `j = 1`, `best = 1` (one container, one crane —
  the prefix-vs-suffix tie is harmless since the count is the same); with `B = 2 < 3` the container is
  unaffordable and the answer is `0`.
- `B = 0`: every `pref[i]` for `i >= 1` is at least `w[0] >= 1 > 0`, so the loop breaks after `i = 0`
  with `j = 0`; answer `0`.
- `B >=` total weight: the budget never bites, only the overlap cap does, and every `i` reports
  `i + (n - i) = n` — remove everything (the `[1,1,1,1]` case).
- Optimum all on one side: an unaffordable-right end forces every `J(i) = 0`, so the answer is the
  largest affordable prefix length, found at that `i` with `j = 0`; symmetrically `i = 0` with a large
  `j` covers an all-suffix optimum. Both extremes live in the sweep's range.

The full program is in the answer.
