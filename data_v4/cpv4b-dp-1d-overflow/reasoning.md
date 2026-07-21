I'm counting length-`n` score sequences over `0..m` in which no two adjacent rounds both hit the peak value `m`, reported modulo `1000000007`. Two numbers in the constraints fix the shape of everything: `n <= 2*10^5` says the algorithm has to be linear, and `m <= 10^9` says the arithmetic is where I'll get hurt. The raw count is exponential in `n` — that's why a modulus is handed to me — so after reducing, every stored count sits in `0..10^9+6`. The per-round multiplier is `m`, also up to `10^9`. So the characteristic product I keep forming is (count)·(m) ≈ `10^9 · 10^9 = 10^18`: snug inside signed 64-bit (ceiling `~9.2*10^18`) but nine-plus orders past the 32-bit ceiling of `~2.1*10^9`. I keep that flag up through the whole derivation — any multiply left in `int` is a silent wrong answer here.

Structure first. The adjacency rule cares only about peaks, and only about the immediately preceding round, so the one bit of history the future needs is whether the round I just placed was a peak. Two running counts over a length-`i` prefix suffice:

- `lo` — sequences ending in a **non-peak** (score one of the `m` values `0..m-1`);
- `hi` — sequences ending in a **peak** (the single value `m`).

Extending by one round: a non-peak may follow anything and has `m` score choices, so `lo' = (lo + hi)·m`; a peak has one choice but may not follow a peak, so `hi' = lo`. Base case `i = 1`: `lo = m`, `hi = 1`. The answer for `n >= 1` is `lo + hi`. Each step needs only the previous pair, so this is `O(n)` time and `O(1)` memory — a rolling two-variable scan.

Run the given sample `n = 3, m = 2` (expect `22`): `(2,1) -> ((3)·2, 2) = (6,2) -> ((8)·2, 6) = (16,6)`, total `22`. The intermediate `i = 2` total `8` independently equals the brute count — `9` length-2 strings minus the one banned pair `(2,2)` — so the transitions are right on more than a single endpoint.

The same `n = 2` case gives me an overflow target for free. For general `m` the recurrence gives `(m+1)·m + m = m^2 + 2m`, while a direct count is `(m+1)^2 - 1 = m^2 + 2m` (all length-2 strings minus the single `(m,m)`). Identical for every `m`. At `m = 10^9` that is `10^18 + 2*10^9`, which reduces to `35` modulo `10^9+7` — an exact answer I can check the arithmetic against.

Now the hazard concretely. The tempting first cut accumulates in `int`: `nlo = (lo + hi) * m % MOD`. On the sample it prints `22` — small numbers never overflow, so the sample certifies nothing about large inputs. But at `n = 2, m = 10^9` the inner expression is `(10^9 + 1) * 10^9 ≈ 10^18`, and with `int` operands that product is formed in 32-bit and wraps modulo `2^32` *before* `% MOD` ever runs; the modulo is then reducing already-corrupted bits. The answer is `35`; the int build disagrees. And on the stress case `n = 2*10^5, m = 10^9` the int accumulation comes out **negative** — a count of sequences can never be negative, so that alone flags signed 32-bit overflow. The recurrence is right; the type is wrong.

The fix is to carry every sum, product, and modulo in `long long`, reducing before each multiply so both operands entering the product are `< MOD`. Then the largest intermediate is `(lo + hi)` before its reduction, at most `~2*10^9`, and the product of two sub-`MOD` factors is `< (10^9+7)^2 ≈ 1.0*10^18` — both comfortably inside 64-bit, no wrap anywhere. Re-running: `n = 3, m = 2` still prints `22`; `n = 2, m = 10^9` now prints `35`, matching the target for exactly the reason I fixed; and `n = 2*10^5, m = 10^9` produces a valid non-negative residue below `MOD` instead of the earlier negative garbage.

Edge cases, where counting DPs usually die:

- `n = 0`: the empty sequence vacuously obeys the rule, so the answer is `1`. My loop starts at `i = 2` off a base built for `i = 1`, so `n = 0` needs an explicit early return printing `1`.
- `n = 1`: the loop never runs and the base pair gives `lo + hi = m + 1` — every one of the `0..m` scores is allowed. Correct.
- `m = 0`: the only score is `0`, which *is* the peak, so every round peaks. Base is `lo = 0, hi = 1` (one valid single round); for `n >= 2` the recurrence gives `lo[2] = (0+1)·0 = 0`, `hi[2] = lo[1] = 0`, and it stays `0` forever — you cannot place two rounds without two adjacent peaks. Correct.

I read `n` and `m` as `long long` and guard `if (!(cin >> n >> m)) return 0;` against empty input. That lands the two-track `O(n)` scan with all arithmetic in 64-bit; the full program is in the answer.
