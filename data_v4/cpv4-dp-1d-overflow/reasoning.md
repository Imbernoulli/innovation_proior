**Reading the problem and pinning the contract.** I have a single row of `n` lamps. Each lamp is working (`'.'`) or broken (`'x'`). I count *displays* — assignments of on/off to the lamps — that obey two rules: a broken lamp is never on, and no two adjacent lamps are both on. The all-off display always counts. Input is `n` then, if `n > 0`, a length-`n` string `s`; output is the single count. Before I touch an algorithm I fix the scale, because in a *counting* problem the dangerous number is not `n` but the answer itself. Here `n <= 90`, which sounds tiny, yet the count of `11`-avoiding strings of length `n` is a Fibonacci number `F(n+2)`, and Fibonacci grows exponentially. Let me get a concrete sense of how big: `F(n+2)` for `n = 90` is `F(92)`. I know `F(92)` is on the order of `7.5 * 10^18`. That is the size of the *answer*, and it sits just under the signed 64-bit ceiling `9.22 * 10^18` but is roughly *three and a half billion times* larger than the signed 32-bit ceiling `2.15 * 10^9`. So the moment `n` passes the mid-forties the answer overflows a 32-bit `int` silently. Conclusion locked in before any code: every accumulator is `long long`, and the problem author has clearly chosen the bound `n <= 90` precisely so the answer fits in `long long` but blows past `int`. An `int` here is not a near miss — it is a guaranteed wrong answer on a large fraction of tests.

**Why the bound is exactly 90, and a sanity check that I am not about to overflow even `long long`.** I should make sure my own data type survives the worst case, the all-working strip at `n = 90`. If `F(92) ~ 7.5 * 10^18 < 9.22 * 10^18`, I am safe; and `F(93) ~ 1.22 * 10^19` would exceed signed `long long`, which is exactly why the author stops at `90` rather than `91`. So `long long` is both necessary (to beat `int`) and sufficient (it does not itself overflow within the stated bound). Good — the type is pinned down on both sides, and I will revisit this as an explicit overflow edge case at the end.

**Laying out the candidate approaches.** Two routes, and I want the one I can transcribe without subtle errors, not merely the asymptotically cutest.

- *Closed form via Fibonacci.* For an all-working strip the answer is `F(n+2)`, computable in `O(1)` (or `O(n)` by iterating the recurrence). The trouble is the broken-lamp mask. A broken lamp forces an off at that position, which severs the line: the runs of consecutive working lamps between broken lamps become independent, and the total count is the *product* of the per-run counts `F(len+2)`. That is correct but fiddly — I have to find run lengths, multiply Fibonacci numbers, and handle the empty-run and all-broken corners. Every one of those is a place to be off by one. I distrust it not because it is wrong in principle but because it is hard to get right by hand under time pressure.

- *Linear DP over the lamps.* Scan left to right, carrying for each prefix two counts: how many valid partial displays end with the last lamp **off**, and how many end with it **on**. A broken lamp simply cannot be on, so it contributes `0` to the "ends on" count and the rest of the machinery is untouched. This handles the mask uniformly with zero special-casing of runs, is `O(n)` and `O(1)` memory, and the transitions are short. This is the one I commit to.

**Deriving the DP and checking the recurrence on paper.** Define, over the processed prefix `s[0..i]`:

- `off` = number of valid displays of the prefix whose last lamp `i` is **off**,
- `on`  = number of valid displays of the prefix whose last lamp `i` is **on**.

The only thing the next lamp cares about is whether lamp `i` is on (that is what the adjacency rule constrains), so these two states are a sufficient summary. Transitions when I append lamp `i`:

- Lamp `i` **off**: the previous lamp may have been anything, so `off_i = off_{i-1} + on_{i-1}`.
- Lamp `i` **on**: this is allowed only if lamp `i` is working *and* the previous lamp was off, so
  `on_i = (s[i] == '.') ? off_{i-1} : 0`.

The base case is the empty prefix before any lamp. There is exactly one empty display, and I need it to feed the first lamp's transition as "the previous lamp was off" (an absent lamp imposes no adjacency conflict). So I initialize `off = 1` (the one empty display, treated as ending "off" by convention) and `on = 0`. The final answer over the whole strip is `off + on`.

Let me confirm the recurrence by hand on the worked sample `n = 4`, `s = "...x"`, expected answer `5`. Start `(off, on) = (1, 0)`.
- i=0 (`.`): `off = 1 + 0 = 1`; `on = off_prev = 1`. -> `(1, 1)`.
- i=1 (`.`): `off = 1 + 1 = 2`; `on = off_prev = 1`. -> `(2, 1)`.
- i=2 (`.`): `off = 2 + 1 = 3`; `on = off_prev = 2`. -> `(3, 2)`.
- i=3 (`x`): `off = 3 + 2 = 5`; `on = 0` (broken). -> `(5, 0)`.

Answer `off + on = 5 + 0 = 5`. Matches, and the broken lamp at the end correctly zeroed the "on" state. The recurrence is right. Let me also check `n = 0`: the loop never runs, the answer is `off + on = 1 + 0 = 1`, the vacuous empty display — correct, and it confirms the base case is doing real work, not just priming the loop.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the loop body, mirroring the two transitions:

```
long long off = 1, on = 0;
for (int i = 0; i < n; i++) {
    off = off + on;                       // lamp i off: previous either
    on  = (s[i] == '.') ? off : 0;        // lamp i on: needs working + previous off
}
```

The shape is right but something nags: I assign `off` and then immediately read `off` again on the same line's successor when computing `on`. Both transitions are supposed to read the *previous* pair. Let me trace the smallest input that could expose it: `n = 2`, `s = ".."`. By hand the valid displays of two adjacent working lamps are `..`, `#.`, `.#` — three of them (not `##`, which is the forbidden adjacency). So the answer must be `3`. Run my code. Start `(off, on) = (1, 0)`.
- i=0 (`.`): `off = 1 + 0 = 1`; then `on = (working) ? off : 0 = off = 1`. -> `(1, 1)`.
- i=1 (`.`): `off = 1 + 1 = 2`; then `on = off = 2`. -> `(2, 2)`.

Answer `off + on = 2 + 2 = 4`. Wrong — it should be `3`.

**Diagnosing the first bug (in-place clobber).** The defect is precise. On `i=1` I overwrote `off` with `off + on = 2` *first*, and then computed `on = off = 2` reading the *already-updated* `off`. But `on_i` is supposed to equal `off_{i-1}`, the previous off-count, which was `1`, not the new `2`. By reading the freshly-written `off`, my "lamp 1 is on" count silently includes displays where lamp 0 was on — exactly the adjacency I forbade. The two transitions both depend on the *old* pair `(off_{i-1}, on_{i-1})`, and I destroyed `off` before reading it for the `on` transition. The fix is to compute both new values from the old pair via temporaries before committing either:

```
long long off = 1, on = 0;
for (int i = 0; i < n; i++) {
    long long noff = off + on;                  // lamp i off
    long long non  = (s[i] == '.') ? off : 0;   // lamp i on: reads OLD off
    off = noff;
    on  = non;
}
```

Re-trace `n = 2`, `s = ".."`. Start `(1, 0)`.
- i=0 (`.`): `noff = 1 + 0 = 1`, `non = off_old = 1`. Commit -> `(1, 1)`.
- i=1 (`.`): `noff = 1 + 1 = 2`, `non = off_old = 1`. Commit -> `(2, 1)`.

Answer `2 + 1 = 3`. Correct. And the case broke for exactly the reason I fixed — the temporaries now preserve the old `off` for the `on` transition — which is the evidence I trust over "it looks right." Let me re-confirm the original sample too: `n = 4`, `s = "...x"`. With temporaries: `(1,0) -> (1,1) -> (2,1) -> (3,2) -> (5,0)`, answer `5`. Still correct.

**Second debug episode — the int-overflow the twist demands.** The logic is right; now I hunt for the silent-data-type failure, because this whole problem is engineered around it. Suppose, in a moment of carelessness, I had written the accumulators as `int` instead of `long long` (a very natural slip — `n <= 90` *feels* small, so why would anything overflow?). Let me trace what happens on the worst case, the all-working strip, as `n` climbs. The pair `(off, on)` follows consecutive Fibonacci numbers: after processing `i` working lamps, `off + on = F(i+3)` and each component is itself a Fibonacci number. So the running total tracks `F`. The signed 32-bit ceiling is `2147483647`. I need the first `n` where the answer `F(n+2)` exceeds that.

- `n = 44`: `F(46) = 1836311903`. Still below `2147483647`. An `int` survives — `1836311903`.
- `n = 45`: `F(47) = 2971215073`. This exceeds `2147483647`. An `int` accumulator wraps: `2971215073 - 2^32 = 2971215073 - 4294967296 = -1323752223`. So the `int` program would print `-1323752223` — a *negative count*, visibly nonsensical, on an input as small as 45 lamps.

That is the smoking gun: at `n = 45` the correct answer is `2971215073` but an `int` prints `-1323752223`. And it gets worse — by `n = 46` the wrap lands on `512559680`, a *positive* but wrong number (`F(48) = 4807526976`, and `4807526976 - 4294967296 = 512559680`), which is the truly dangerous case because it is not obviously bogus; it just silently disagrees with the judge. So the overflow is not a theoretical worry confined to the very largest input; it triggers at less than half the maximum `n`, and for `n` in the high forties it produces a *plausible-looking* wrong positive integer. The cure is the type decision I locked in at the start: `long long`. With `long long`, `n = 45` gives `2971215073` and `n = 90` gives `7540113804746346429`, both exact. I keep the accumulators 64-bit.

**A note on why I do *not* need modular arithmetic or big integers.** A reflex on "the count is huge" problems is to take everything mod some prime, but here the contract asks for the *exact* count and the bound `n <= 90` was chosen so the exact answer fits in signed `long long` (worst case `F(92) ~ 7.5 * 10^18 < 9.22 * 10^18`). Introducing a modulus would be a *different, wrong* problem. And big integers are unnecessary overhead. So `long long` is exactly the right width: not too small (`int` overflows), not needlessly large (`__int128`/bignum). This is the whole pedagogical edge of the problem — recognising that "small `n`" does not imply "small answer."

**Sanity-checking the derivation itself against an independent count.** Beyond tracing the code, I want to know the *recurrence* is the right one, independent of implementation. For an all-working strip the valid displays are binary strings of length `n` with no two adjacent `1`s, and that count is the well-known Fibonacci `F(n+2)`: `n=1 -> 2`, `n=2 -> 3`, `n=3 -> 5`, `n=4 -> 8`, `n=5 -> 13`. My DP on `n=5`, `s="....."`: `(1,0) -> (1,1) -> (2,1) -> (3,2) -> (5,3) -> (8,5)`, answer `8 + 5 = 13`. Matches `F(7) = 13`. The DP reproduces the closed form on the masks where the closed form applies, which is strong evidence the recurrence is correct, while the DP additionally handles the masked cases the closed form would force me to assemble by hand.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: the loop never runs; answer `off + on = 1 + 0 = 1`. The one vacuous empty display — correct.
- `n = 1`, `s = "."` (single working lamp): `(1,0) -> noff = 1, non = 1 -> (1,1)`, answer `2` (off or on). Correct.
- `n = 1`, `s = "x"` (single broken lamp): `(1,0) -> noff = 1, non = 0 -> (1,0)`, answer `1` (it can only be off). Correct.
- All broken, e.g. `s = "xxxxxxx"`: every lamp forces `non = 0`, so `on` stays `0` forever and `off` stays `1` (since `off = off + on = off + 0`), answer `1`. The only display is all-off — correct.
- Two adjacent working lamps `".."`: answer `3` (traced above), correctly excluding `##`.
- A mask that splits the row into runs, `n = 6`, `s = ".x..x."`: by hand the three runs are lengths `1, 2, 1`, contributing `F(3)*F(4)*F(3) = 2*3*2 = 12`. My DP: `(1,0)` -> i0 `.`: `(1,1)` -> i1 `x`: `noff=2, non=0 -> (2,0)` -> i2 `.`: `noff=2, non=2 -> (2,2)` -> i3 `.`: `noff=4, non=2 -> (4,2)` -> i4 `x`: `noff=6, non=0 -> (6,0)` -> i5 `.`: `noff=6, non=6 -> (6,6)`, answer `12`. Matches the run-product, confirming the broken lamp correctly severs adjacency between its neighbours (the `x` resets `on` to `0`, so the lamp after it sees a clean "previous off").
- Overflow: accumulators are `long long`; worst case `F(92) ~ 7.5 * 10^18` fits with room under `9.22 * 10^18`. An `int` would have failed at `n = 45`. The sentinel values are never negative and only ever summed, so there is no underflow path. Safe.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace, so the `n`-then-string format is parsed robustly, and the `if (n > 0) cin >> s` guard means I never read a non-existent second token when `n = 0`.

**Final solution.** I convinced myself the *idea* is right by deriving the two-state recurrence, hand-checking it on the sample, and matching the all-working case to the independent Fibonacci closed form; I convinced myself the *code* is right by tracing the failing `".."` case to a precise in-place-clobber cause and re-verifying the temporary-based fix, and by tracing the `int` accumulator to the exact lamp count (`n = 45`) where it silently goes wrong, which fixes the data type. That is what I ship — one self-contained `O(n)` file with 64-bit accumulators:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;            // n = 0 -> exactly one pattern (all lamps off, vacuously valid)
    string s;
    if (n > 0) cin >> s;                  // s has length n: '.' = working lamp, 'x' = broken (forced off)

    // off = number of valid patterns for the processed prefix whose LAST lamp is OFF
    // on  = number of valid patterns for the processed prefix whose LAST lamp is ON
    // A lamp may be ON only if it is working ('.') and the previous lamp is OFF.
    long long off = 1, on = 0;            // empty prefix: 1 pattern, ends "off" by convention
    for (int i = 0; i < n; i++) {
        long long noff = off + on;        // this lamp OFF: previous could be either
        long long non = (s[i] == '.') ? off : 0; // this lamp ON: needs working lamp + previous OFF
        off = noff;
        on = non;
    }

    cout << (off + on) << "\n";           // total valid patterns over the whole strip
    return 0;
}
```

**Causal recap.** Recognising that a *counting* problem's danger is the answer's magnitude, not `n`, I noted the count is a Fibonacci number reaching `~7.5 * 10^18` at `n = 90`, which fixed `long long` as both necessary (an `int` wraps to `-1323752223` at `n = 45`, where the true answer is `2971215073`) and sufficient (it stays under the signed-64 ceiling, which is why the author caps `n` at `90`). I committed to a two-state prefix DP — counts ending with the last lamp off versus on — because it absorbs the broken-lamp mask uniformly (a broken lamp just zeroes the "on" state) where a Fibonacci closed form would force fragile run-product bookkeeping; both transitions read the previous `(off, on)` pair, which my first in-place loop destroyed by overwriting `off` before reading it for `on` (a trace of `".."` returning `4` instead of `3` pinned it), and computing both new values from the old pair via temporaries fixes it; cross-checking the DP against the Fibonacci closed form on all-working strips and against a run-product on `".x..x."` confirmed the recurrence, and the `n = 0`, single-lamp, and all-broken corners fall straight out of the base case `(off, on) = (1, 0)`.
