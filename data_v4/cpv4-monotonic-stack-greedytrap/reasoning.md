The skyline is a string of `n` single digits and I must demolish **exactly** `k` towers so the
`n - k` survivors, kept in order and squeezed together, read lexicographically smallest. Two features
of the contract shape everything before I pick an algorithm. First, heights are single characters
`'0'`–`'9'`, so there is no arithmetic and no overflow anywhere — the entire difficulty lives in
*which* towers to remove, a pure selection problem. Second, and unlike the textbook "remove `k`
digits to form the smallest number", **leading zeros are printed, not stripped**, and the removal
count is *exact*: I emit a length-`(n - k)` string verbatim, a bare newline when `k = n`. That kills
one special case the classic phrasing carries, and it means I must always spend the full budget of
`k`. With `n <= 2*10^5`, a quadratic scan is `4*10^10` operations against a 1-second limit, so I need
linear work. I read `k` as `long long` since it costs nothing, though the contract bounds it by `n`.

**What lexicographic-smallest rewards.** Comparison is positional: the first surviving character
dominates the second, which dominates the third. So the objective decomposes greedily — make the
earliest surviving slot as small as possible, then the next, and so on. A demolition spent on an
early tall tower can promote a short tower into a very significant slot, worth far more than removing
a taller tower that sits late. That already makes me suspicious of ranking towers by height alone.

**Ruling out the value greedy.** The tempting `O(n log n)` route is "demolish the `k` tallest digits".
But "tallest" is a property of the digit while the *damage* a tower does is a property of the digit
*and its position*, and when those disagree the greedy breaks. Take `s = "2102"`, `k = 1`: the
tallest digit `'2'` sits at index 0 and index 3. Removing the last `'2'` gives `"210"`; removing the
earliest gives `"102"`, strictly smaller because a smaller digit reaches the leading slot. So
correctness hinges entirely on a tie-break — prefer the earliest max — that the greedy does not
naturally carry. And with two removals even that variant fails: on `"12002"`, `k = 2` the optimum is
`"002"` (drop the leading `1` and one `2` to expose both zeros) while "remove the two tallest" yields
`"100"`, far larger. The elimination hands me the right primitive: demolish the *earliest tower a
shorter successor can improve upon*, not the globally tallest.

**The monotonic stack.** Scan left to right, holding the survivors-so-far on a stack `st` and a
`budget` of remaining demolitions (initially `k`). When tower `c` arrives, while `budget > 0` and the
stack top is taller than `c`, pop the top and spend a demolition; then push `c`. Optimality: consider
the earliest slot the answer will fill. If a shorter tower `c` arrives while a taller `t` sits on top
and I can still afford the removal, deleting `t` strictly improves the answer at `t`'s significant
position and harms nothing below — everything under `t` is already `<= t` by the invariant. The stack
stays non-decreasing bottom-to-top throughout, which is exactly what makes the local pop globally
correct, and each tower is pushed once and popped at most once: `O(n)`.

**Strictness of the pop is not free to get wrong.** If the top *equals* `c`, popping it spends a
demolition and refills the slot with the same digit — no improvement, and that budget is gone for a
later real gain. So the condition must be strict `>`, not `>=`. The gap is concrete: on `"112"`,
`k = 1` the only useful removal is the trailing `'2'`, giving `"11"`. Under `>=`, the second `'1'`
pops the first (equal top), burns the single demolition, and when `'2'` finally arrives there is no
budget left — output `"12"`, wrong. Under strict `>` the two ones survive and only the `'2'` is a pop
candidate.

**Leftover budget.** A non-decreasing skyline never triggers a pop, so the scan ends with
`budget > 0`. Those demolitions must still happen — exactly `k` — and on a non-decreasing remainder
the smallest result drops from the *end*, the least significant towers. So after the scan I drain the
tail `budget` more times. This is also what finishes `"112"` above: strict `>` leaves the scan at
`"112"` with one unspent removal, and the tail drain trims the last character for `"11"`. Skip the
drain and I print `n` characters instead of `n - k` — `"12345"`, `k = 2` would emit `"12345"` rather
than `"123"`.

**Checking the given example.** `s = "1432219"`, `k = 3`, expected `"1219"`. Push `'1'`; `'4'` has a
smaller top, push → `"14"`; `'3'` pops `'4'` (budget 2) → `"13"`; `'2'` pops `'3'` (budget 1) →
`"12"`; the second `'2'` is *not* `> '2'` so it does not pop →
`"122"`; `'1'` pops the top `'2'` (budget 0) → `"121"`; `'9'` pushes → `"1219"`. Budget reaches 0
mid-scan so the drain is idle. Matches the stated answer.

**Corners.** `k = 0`: budget 0, nothing pops or drains, I print `s`. `k = n`: everything goes —
`"9081"`, `k = 4` collapses mid-scan to `"01"`, then the drain empties it and I print a bare newline
(length `n - k = 0`). All-equal `"00000"`, `k = 2`: no `'0'` is ever `> '0'`, so the drain trims the
tail to `"000"`. Strictly decreasing `"9531"`, `k = 2`: each new tower pops its taller predecessor,
demolishing the front → `"31"`. And the case that sank the value greedy, `"2102"`, `k = 1`: `'1'`
pops the leading `'2'`, the scan finishes `"102"` — the positional optimum, reached without any
tie-break bookkeeping.

**Cost.** Each character is pushed once and popped at most once (the drain only removes characters
that were pushed), so the whole run is `O(n)`; I `reserve(n)` on the result string to avoid
reallocation churn. At `n = 2*10^5` that is comfortably inside a second — no sort, no rescan.

So the whole solution is one left-to-right scan with a strict-`>` pop, then a tail drain for any
unspent budget. The strict comparison keeps equal towers from wasting a demolition, and the drain
forces leftover removals onto the least significant tail. Wrapped in the stdin/stdout contract — read
`n`, `k`, `s`; print the result string then a newline — that is the program.
