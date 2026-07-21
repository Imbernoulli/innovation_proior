The reduction here is textbook once I see it, so the whole difficulty lives in three corners the
constraints are deliberately built around: the sign mix (values negative, zero, or positive), the
*strict* `< 0` boundary, and the sheer scale. Fix the scale first, because it dictates the types.
`n <= 2*10^5` gives up to `n(n+1)/2 ~ 2*10^10` subarrays, and the *count of qualifying ones* can be
on that same order — well past 32-bit (`~2.1*10^9`), so the answer accumulator must be 64-bit. And
`|a[i]| <= 10^9` over `2*10^5` terms lets a prefix sum reach `2*10^14`, also 64-bit. An `int`
anywhere on the count or the prefix sums is a silent wrong-answer on the large tests; I use
`long long` throughout.

**Reducing to a counting-inversions shape.** The subarray sum is awkward directly, so I move to
prefix sums. Define `P[0] = 0` and `P[k] = a[0] + ... + a[k-1]` for `1 <= k <= n`. Then
`sum(l, r) = P[r+1] - P[l]`, and `sum(l, r) < 0  <=>  P[r+1] < P[l]`. As `(l, r)` ranges over
`0 <= l <= r < n`, the pair `(i, j) = (l, r+1)` ranges over `0 <= i < j <= n`. So the answer is
exactly the number of pairs `(i, j)` with `i < j` and `P[j] < P[i]` — the count of *strict
inversions* of the `(n+1)`-element prefix array. Index `0`, `P[0] = 0`, is a real participant here:
a subarray starting at `l = 0` is the pair `i = 0`, so `P[0]` must be one of the "earlier" values
counted against. Dropping it silently loses every left-anchored subarray.

**Choosing the algorithm.** Brute force — for each `l`, run a running sum rightward and tally each
dip below `0` — is `O(n^2)`, three lines, correct by construction, but `~2*10^10` operations blows
the 1-second limit; it survives only as an oracle. So: a Fenwick over compressed prefix sums. Sweep
`j` from `0` to `n` in index order; just before inserting `P[j]`, count the already-inserted `P[i]`
(with `i < j`) that are strictly greater than `P[j]`. Summed over `j`, that is the strict-inversion
count, and each query is `O(log n)`, so `O(n log n)` overall.

Two spots need care, and both are exactly what the negatives/zeros corner exposes. A Fenwick
naturally answers "how many inserted values have rank `<= r`" — a prefix sum of frequencies — but I
need "strictly greater than `P[j]`." With `r = rank(P[j])` (1-based rank among sorted-unique
values), "rank `<= r`" counts values `<= P[j]` (equal values share a rank), so
`greater = (number inserted so far) - (count with rank <= r)`. The first spot is seeding `P[0] = 0`
correctly; the second is the equality boundary, `<= P[j]` versus `< P[j]`, which flips on every pair
of equal prefix sums — and the all-zero array is nothing but equal prefix sums.

**Checking the pipeline on the sample.** For `a = [3, -4, 1, -2]` (answer `7`), prefix sums are
`P = [0, 3, -1, 0, -2]`; sorted unique `[-2, -1, 0, 3]` give ranks `-2 -> 1`, `-1 -> 2`, `0 -> 3`,
`3 -> 4`. Sweeping `j = 0..4`, each step querying "earlier values strictly greater than `P[j]`" then
inserting `P[j]`:

- `j=0`, `P=0`, rank 3. Inserted 0. `greater = 0`. Answer 0. Insert rank 3.
- `j=1`, `P=3`, rank 4. Inserted 1. Count `<= 3` is 1; `greater = 1 - 1 = 0`. Answer 0. Insert rank 4.
- `j=2`, `P=-1`, rank 2. Inserted 2. Count `<= -1` is 0; `greater = 2 - 0 = 2`. Answer 2. Insert rank 2.
- `j=3`, `P=0`, rank 3. Inserted 3. Count `<= 0` is 2 (the `0` and the `-1`); `greater = 3 - 2 = 1`.
  Answer 3. Insert rank 3.
- `j=4`, `P=-2`, rank 1. Inserted 4. Count `<= -2` is 0; `greater = 4 - 0 = 4`. Answer 7. Insert rank 1.

Final `7`, matching.

**First cut, and the base case it gets wrong.** My first sweep:

```
Fenwick fen(m);
long long answer = 0;
for (int j = 1; j <= n; j++) {            // start at first real prefix
    int r = rankOf(P[j]);
    long long greater = (j - 1) - fen.sumPrefix(r);
    answer += greater;
    fen.add(r, 1);
}
```

I started at `j = 1` reasoning "the subarray's right end `r+1` is at least `1`, so the first
interesting prefix is `P[1]`," with `inserted = j - 1`. That conflates the two endpoints. Trace the
input where left-anchored subarrays dominate: `a = [-2, -5, -1]`, every subarray negative, so the
answer must be `n(n+1)/2 = 6`. Prefix sums `P = [0, -2, -7, -8]`, sorted unique `[-8, -7, -2, 0]`,
ranks `-8 -> 1`, `-7 -> 2`, `-2 -> 3`, `0 -> 4`:

- `j=1`, `P=-2`, rank 3. `inserted = 0`, tree empty, `greater = 0`. Answer 0. Insert rank 3.
- `j=2`, `P=-7`, rank 2. `inserted = 1`, `sumPrefix(2) = 0`, `greater = 1`. Answer 1. Insert rank 2.
- `j=3`, `P=-8`, rank 1. `inserted = 2`, `sumPrefix(1) = 0`, `greater = 2`. Answer 3. Insert rank 1.

Returns `3`, truth is `6` — exactly half missing, and the missing half is the three subarrays
starting at `l = 0`: `[0,0]=-2`, `[0,1]=-7`, `[0,2]=-8`. Those are inversions against `i = 0`, i.e.
against `P[0] = 0`, but the loop began at `j = 1` and never inserted `P[0]`. By the time I query
`P[1..3]`, the value `0` — strictly greater than all of them, worth `+1` each — simply is not in the
tree. This is the missing-base-case bug the reduction warned me about, and all-negative data is
precisely the input that makes it halve the answer rather than error loudly.

**The fix.** Start the sweep at `j = 0` with `inserted = j`. Then `P[0]` is queried (no earlier
values, contributes `0`) and, crucially, inserted so every later prefix sees it. Re-tracing
`[-2, -5, -1]`: `j=0` inserts rank 4 (greater 0); `j=1` inserted 1, greater 1, answer 1; `j=2`
inserted 2, greater 2, answer 3; `j=3` inserted 3, greater 3, answer 6. The left-anchored subarrays
are back.

**The strict boundary.** With the base case seeded, the equality question decides strict-vs-nonstrict.
`greater = inserted - sumPrefix(r)` with `r = rank(P[j])` subtracts the count of earlier values
`<= P[j]`, leaving those strictly `> P[j]` — the strict inversion I want. The tempting wrong form
queries `sumPrefix(r-1)` (earlier values `< P[j]`), which would leave `>= P[j]`, i.e. subarrays with
sum `<= 0`, wrongly counting the zero-sum ones. The pure-equality input separates them: for
`a = [0, 0, 0]` (answer `0`), all prefix sums are `0` with rank `1`. The correct form gives
`sumPrefix(1) = inserted` at every step, so `greater = 0` throughout — answer `0`. The `sumPrefix(r-1)`
form would give `inserted - 0 = 0, 1, 2, 3`, summing to `6`, counting all six zero-sum subarrays as
negative. My query uses `sumPrefix(r)`, so zeros stay out.

**Edge cases where this kind of code dies.**
- `n = 0`: input is just `0`, the value loop never runs, `P = [0]`. The sweep runs one iteration
  `j=0` with `greater = 0`, answer `0` — no subarrays. Truly empty stdin (no `n` token) exits via
  `if (!(cin >> n)) return 0;`.
- `n = 1`, `a = [-5]`: `P = [0, -5]`; at `j=1` the earlier `0` is strictly greater, `greater = 1`,
  answer `1`. `n = 1`, `a = [0]`: `P = [0, 0]`; `0` is not strictly greater than `0`, answer `0`.
- All-positive: prefix sums strictly increase, no earlier value is ever greater, answer `0`.
- Overflow: at `n = 2*10^5` with values near `10^9`, the count can exceed 32-bit and a prefix sum
  reaches `~2*10^14` — both live in the `long long` accumulators. The Fenwick holds frequencies
  (`<= n`), so `int` would suffice there, but I keep it `long long` to avoid mixed-type surprises.
  The `O(n log n)` sweep is comfortably under the 1-second limit.

The final program is the Fenwick sweep over compressed prefix sums, seeded from `P[0]`, querying
`sumPrefix(r)` for the strict boundary, with 64-bit accumulators throughout.

