A single row of `n` lamps, each working (`'.'`) or broken (`'x'`); I count the on/off displays that keep every broken lamp dark and never light two neighbours at once, all-off included. The I/O is small — read `n`, then a length-`n` mask if `n > 0`, print one integer — and `n <= 90` looks harmless. But this is a *counting* problem, and in a counting problem the dangerous number is the answer, not `n`. On an all-working strip the valid displays are exactly the `11`-avoiding binary strings, counted by the Fibonacci number `F(n+2)`, which grows exponentially. At `n = 90` that is `F(92) ~ 7.5 * 10^18` — comfortably inside signed 64-bit (`9.22 * 10^18`) but about three and a half billion times past the signed 32-bit ceiling `2.15 * 10^9`. So an `int` accumulator is a guaranteed wrong answer, and the type decision is settled before I write a line: every accumulator is `long long`.

The bound is `90` and not `91` for a reason. `F(92) ~ 7.5 * 10^18` fits under the signed-64 ceiling, while `F(93) ~ 1.22 * 10^19` would overflow it — so the constraint is tuned to make `long long` both necessary (against `int`) and sufficient. No modulus (the contract asks for the *exact* count, so reducing mod a prime would answer a different problem) and no big integers (overkill for a value that fits in 64 bits): `long long` is the exactly-right width.

**Two routes to the count.** For an all-working strip the answer is the closed form `F(n+2)`, `O(1)`. But a broken lamp forces an off, which severs the line: the runs of consecutive working lamps between the `x`s become independent and their per-run counts `F(len+2)` multiply. That is correct but fiddly to assemble by hand — finding run lengths, multiplying, and getting the empty-run and all-broken corners right are all off-by-one traps. The alternative is a linear DP that scans the lamps left to right, carrying for each prefix how many valid displays end with the last lamp **off** versus **on**. A broken lamp just contributes `0` to the "ends on" count, so the mask needs no special-casing at all — `O(n)`, `O(1)` memory, short transitions. That is the one I can transcribe without subtle errors, so that is the one I commit to.

**Deriving the DP.** Over the processed prefix `s[0..i]`, let `off` be the number of valid displays whose last lamp is off and `on` the number whose last lamp is on. The next lamp only cares whether lamp `i` is on — that is all the adjacency rule constrains — so this pair is a sufficient summary. Appending lamp `i`:

- lamp `i` **off**: the previous lamp may be anything, so `off_i = off_{i-1} + on_{i-1}`;
- lamp `i` **on**: allowed only if the lamp works *and* the previous lamp was off, so `on_i = (s[i] == '.') ? off_{i-1} : 0`.

The base case is the empty prefix: one empty display, which I initialize as `off = 1`, `on = 0`, treating "no lamp yet" as ending off so the first lamp sees a valid previous-off. The answer over the whole strip is `off + on`. On the worked sample `n = 4`, `s = "...x"` (expected `5`): `(1,0) -> (1,1) -> (2,1) -> (3,2) -> (5,0)`, sum `5`, with the broken final lamp correctly zeroing `on`. And `n = 0` runs no iterations and returns `1`, the vacuous empty display.

**The transcription hazard.** Both transitions read the *previous* pair `(off_{i-1}, on_{i-1})`, so updating `off` in place before computing `on` is wrong: `on = (s[i]=='.') ? off : 0` would then read the freshly-written `off_i` instead of `off_{i-1}`. On `n = 2`, `s = ".."` (valid displays `..`, `#.`, `.#` — not `##`) that in-place version returns `4`: at `i = 1` it sets `off = 1 + 1 = 2`, then `on = off = 2`, counting the displays where lamp 0 was on — exactly the adjacency the problem forbids. Snapshotting both new values from the old pair before committing either fixes it:

```
long long off = 1, on = 0;
for (int i = 0; i < n; i++) {
    long long noff = off + on;                  // lamp i off
    long long non  = (s[i] == '.') ? off : 0;   // lamp i on: reads OLD off
    off = noff;
    on  = non;
}
```

Now `".."` traces `(1,0) -> (1,1) -> (2,1)`, sum `3`. Correct.

**Where `int` actually breaks.** On an all-working strip the accumulators track consecutive Fibonacci numbers, so a 32-bit answer crosses the ceiling `2147483647` at `n = 45`: `F(46) = 1836311903` (the `n = 44` answer) still fits, but `F(47) = 2971215073` wraps to `2971215073 - 2^32 = -1323752223` — a negative count on 45 lamps. Worse, `n = 46` wraps `F(48) = 4807526976` to a *positive*, plausible-looking `512559680` that a judge silently rejects. The overflow bites at half the cap, not just at the extreme.

**Two independent checks on the recurrence itself.** On an all-working strip it must reproduce `F(n+2)`: for `n = 5`, `s = "....."`, `(1,0) -> (1,1) -> (2,1) -> (3,2) -> (5,3) -> (8,5)`, sum `13 = F(7)`. On a split mask `".x..x."` the DP gives `12 = F(3)*F(4)*F(3)` over runs of lengths `1, 2, 1` — so a broken lamp cleanly severs its neighbours (the `x` forces `on = 0`, and the next lamp sees a guaranteed previous-off).

The remaining corners all fall out of the base case `(off, on) = (1, 0)`: a single working lamp gives `2`, a single broken lamp `1`, and an all-broken strip keeps `on = 0` throughout so it prints `1`. Reading is guarded by `if (n > 0) cin >> s`, so `n = 0` never reaches for a second token that isn't there.

The solution is an `O(n)`, `O(1)`-space scan over the two states with 64-bit accumulators and temporaries so each step reads the previous pair; the full program follows.
