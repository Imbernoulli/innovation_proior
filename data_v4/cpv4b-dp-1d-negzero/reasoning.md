I have `n` signed charges `a[0..n-1]` and must pick one contiguous block `[l..r]`, with the option to re-route around at most one interior stop, maximizing the net charge of the stops actually flown. Two features set this apart from textbook maximum-subarray, and both live in the corners. First, the block must fly at least one stop — no empty mission — so this is *not* the "empty subset allowed, answer >= 0" family: if every charge is negative the answer is the least-negative single stop, a negative number. A best-so-far seeded at `0` is a silent wrong sign exactly here. Second, the scale fixes the types: `n <= 2*10^5`, `|a[i]| <= 10^9`, so a block sum reaches `2*10^14`, two hundred times past the 32-bit ceiling. Every accumulator is `long long`; an `int` here is a silent wrong answer on the big tests, not a crash.

The "skip one stop" allowance is one bit of state — have I spent my re-route yet — so I scan left to right carrying, for the block *ending at the current stop*, the best net charge with the skip unused (`f0`) and with it already spent (`f1`). That is `O(n)` time, `O(1)` memory. (Brute over all blocks with an inner min is `O(n^2)` ~ `4*10^10` at the largest `n`, hopeless as the solution, but I keep it as a checker.)

Deriving the transitions. `f0[i]`, the best block ending exactly at `i` with no skip, is either the singleton `{i}` or an extension of the block ending at `i-1`, both flying `i`:

`f0[i] = max(a[i], f0[i-1] + a[i])`.

The `max(a[i], ...)` — never a standalone `max(0, ...)` — is the non-empty guard: I can restart at `i`, but there is no empty block worth `0` to fall back on. `f1[i]`, the best block ending at `i` with the skip spent, arrives two ways: I skip stop `i` itself, which extends the best no-skip block ending at `i-1` (guaranteeing a kept stop to its left) and contributes `f0[i-1]`; or I fly `i` on top of an already-used skip, `f1[i-1] + a[i]`. So

`f1[i] = max(f0[i-1], f1[i-1] + a[i])`,

and the answer is the max over `i` of `max(f0[i], f1[i])`.

The base cases carry the two corners. `f0` starts at `a[0]` — the always-legal singleton `{0}` — and `ans` starts there too, so a bare `0` never enters the maximum. `f1` starts at negative infinity, not `0`: before any stop is kept there is no skip-state, and seeding it at `0` would invent a "skipped everything, kept nothing" block worth `0`, which is illegal (a skip requires a kept stop). I use `NEG = LLONG_MIN/4` as the sentinel; it is only ever read inside a `max` or given a single `+a[i]` (`~ -2.3*10^18 + 10^9`), never accumulated, so it cannot underflow.

Trace the final loop on the worked sample `a = [3,-1,4,-1,5,-9]`, which should give `11` — fly `[3,-1,4,-1,5]` (sum `10`) and re-route around a `-1`. Start `f0=3, f1=NEG, ans=3`.

- i=1 (-1): `f1 = max(3, NEG-1) = 3`; `f0 = max(-1, 3-1) = 2`; `ans = 3`.
- i=2 (4): `f1 = max(2, 3+4) = 7`; `f0 = max(4, 2+4) = 6`; `ans = 7`.
- i=3 (-1): `f1 = max(6, 7-1) = 6`; `f0 = max(-1, 6-1) = 5`; `ans = 7`.
- i=4 (5): `f1 = max(5, 6+5) = 11`; `f0 = max(5, 5+5) = 10`; `ans = 11`.
- i=5 (-9): `f1 = max(10, 11-9) = 10`; `f0 = max(-9, 10-9) = 1`; `ans = 11`.

`11`, matching the block enumeration.

The all-negative corner is what those base cases exist for. `a = [-7]`: the loop never runs, `ans = a[0] = -7` — correct, where a `0` seed would have returned `0`. `a = [-3,-5]` (the legal optima are `{0}=-3`, `{1}=-5`, `{0,1}=-8`, and either skip `-5` or `-3`; best is `-3`): `f0=-3, f1=NEG, ans=-3`; i=1 (-5): `f1 = max(-3, NEG-5) = -3`, `f0 = max(-5, -3-5) = -5`, `ans = -3`. Correct; had `f1` started at `0`, the phantom "keep nothing" skip would have surfaced as `0` and propped up a wrong `0`.

Update order is the one transcription trap. Both new values read the *previous* pair, and crucially `f1` reads the old `f0`, so I compute `nf0, nf1` into temporaries and assign after. Written in place as `f0 = ...; f1 = max(f0, ...)`, `f1` would read the just-updated `f0` — the no-skip block ending at `i` — which models "skip stop `i` while also flying it", a contradiction that over-credits any case where the skip lands mid-block (e.g. `[5,-100,5]`, whose correct answer of `10` comes from skipping the `-100`).

Remaining corners: `n = 1` never enters the loop and returns `a[0]` (no skip is ever possible); a single `0` returns `0`, right *here* because a real stop produced it; a deep negative bridging two positive runs, e.g. `[2,-50,3,4,-1,5,-100]`, is jumped by the skip; all-positive makes re-routing pure loss, so `f1` never beats `f0` and the answer is the whole-array sum (on `2*10^5` copies of `10^9` that is `2*10^14`, confirming the 64-bit need). To separate idea-correctness from code-correctness I ran the DP against an obvious `O(n^3)` brute — every block, minus each interior element once, never the empty block — on hundreds of small random cases biased toward all-negative, zero-laden, tiny-magnitude inputs so the sign and base-case corners fire constantly. Zero mismatches. I ship the single `O(n)` scan — `f0`/`f1` in temporaries, `ans` seeded from `a[0]`, `f1` seeded at `LLONG_MIN/4`, all accumulators `long long`.
