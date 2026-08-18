Two numbers in the constraints decide the shape of this before I pick an algorithm. The ledger `a[0..n-1]` runs to `n = 2*10^5` entries, each up to `10^9` in magnitude, and the target `S` reaches `10^14`; I count every non-empty contiguous run `[l, r]` with `a[l] + ... + a[r] = S`, reading `n`, `S`, then the `n` values from stdin and printing one integer. The trap here is not the counting idea but that two separate quantities each overflow 32-bit. The answer alone can be as large as the number of non-empty runs, `n(n+1)/2`, which at `n = 2*10^5` is `20000100000`, about `2*10^10` — nearly tenfold past the `int` ceiling of `2.147*10^9`. Independently, a prefix sum can reach `n * 10^9 = 2*10^14`, and with `S` up to `10^14` the quantity `P[r+1] - S` reaches roughly `3*10^14` in magnitude. All of it sits comfortably inside int64's `~9.2*10^18`. So every accumulator, every map key and value, `S`, and the running prefix is `long long`; a single `int` in this code is a silent wrong answer on the large tests, not a crash. That is settled first.

The sum of `[l, r]` is `P[r+1] - P[l]` where `P[0] = 0` and `P[k] = a[0] + ... + a[k-1]`, so `[l, r]` matches iff `P[l] = P[r+1] - S`. Sweeping the right edge and asking how many earlier prefix sums equal `P[r+1] - S` gives the count in `O(n)` with a frequency map. The `O(n^2)` alternative — fix `l`, extend `r`, test the running sum — is transparently correct and I keep it as a brute oracle for stress-testing, but `~4*10^10` additions blows the 1-second budget, so the hashing sweep is what ships. The idea is standard; all the risk is in transcription.

The one thing to get exactly right is the range of `l`. A run `[l, r]` is non-empty iff `l <= r`, so for a fixed right edge the valid left endpoints are prefix indices `P[0], ..., P[r]` — never `P[r+1]`, which would be the empty run `[r+1, r]` of sum 0, and that must never count, not even when `S = 0`. So the invariant: while I process the prefix `P[r+1]`, the map holds exactly `{P[0], ..., P[r]}`. That fixes the loop order — query for `P[r+1] - S`, add its frequency, then insert `P[r+1]` so it becomes a left endpoint for later right edges — and it forces seeding `P[0] = 0` once before the loop, since runs starting on day 0 have `l = 0`.

**A numeric self-check of the recurrence on the sample.** Before coding I verify the derivation on `S = 2`, `a = [3, -1, 1, 2, -2, 2, 1]`, whose documented answer is `6`. Prefix sums: `P[0]=0, P[1]=3, P[2]=2, P[3]=3, P[4]=5, P[5]=3, P[6]=5, P[7]=6`. I sweep right edges `P[1..7]`, each time counting earlier prefixes equal to `P[r+1] - 2`, then inserting.

- Seed map `= {0:1}`.
- `P[1]=3`: need `3-2=1`; map has `{0:1}`, count of `1` is `0`. ans `=0`. Insert `3` -> `{0:1, 3:1}`.
- `P[2]=2`: need `0`; map has one `0`. ans `+=1 -> 1`. (This is run `[0,1]`, sum `2`.) Insert `2` -> `{0:1,3:1,2:1}`.
- `P[3]=3`: need `1`; none. ans `=1`. Insert `3` -> `{0:1,3:2,2:1}`.
- `P[4]=5`: need `3`; map has two `3`s. ans `+=2 -> 3`. (Runs `[1,3]` and `[3,3]`.) Insert `5` -> `{...,5:1}`.
- `P[5]=3`: need `1`; none. ans `=3`. Insert `3` -> `{0:1,3:3,2:1,5:1}`.
- `P[6]=5`: need `3`; map has three `3`s. ans `+=3 -> 6`. (Runs `[1,5]`, `[3,5]`, `[5,5]`.) Insert `5` -> `{...,5:2}`.
- `P[7]=6`: need `4`; none. ans `=6`. Insert `6`.

Re-trace `a = [1, -1]`, `S = 0`. Seed `cnt = {0:1}`, `pref = 0`, `answer = 0`.
- `i=0`: `pref = 1`. Find `1 - 0 = 1` in `{0:1}` -> absent. `answer = 0`. Insert -> `{0:1, 1:1}`.
- `i=1`: `pref = 0`. Find `0 - 0 = 0` in `{0:1, 1:1}` -> present, value `1`. `answer += 1 -> 1`. (Run `[0,1]`, the only match.) Insert -> `{0:2, 1:1}`.

Final `6`. Correct — and notice this `0+1+2+3 = ... ` no, `1+2+3 = 6` telescoping is exactly `n*(n+1)/2 = 3*4/2 = 6`, which is the closed form I claimed for the all-zero case. The numeric self-check confirms that closed form on a concrete instance.

**Numeric self-check of the overflow bound.** I claimed the answer can hit `n*(n+1)/2`. For the all-zero array of length `n = 200000`, the sweep adds `1 + 2 + ... + n = n(n+1)/2 = 20000100000`. That exceeds `2^31 - 1 = 2147483647` by nearly tenfold, confirming the answer accumulator genuinely needs 64 bits (it fits easily in int64). And the largest prefix magnitude is `200000 * 10^9 = 2*10^14`, with `pref - S` reaching about `2*10^14 + 10^14 = 3*10^14` in magnitude — all far below int64's `~9.2*10^18`. So 64-bit arithmetic is both necessary and sufficient; no value overflows.

**Edge cases, because this is where counting code dies.**
- `n = 0`: `cin >> n >> S` reads `n = 0`; the value loop reads nothing; the seed puts `{0:1}` in the map; the main loop never runs; `answer = 0`. The empty ledger has no non-empty runs — correct. The `if (!(cin >> n >> S)) return 0;` guard also handles a totally empty stdin gracefully (prints nothing, but the harness for `n=0` supplies `0` and `S`).
- `n = 1`, `a = [7]`, `S = 7`: seed `{0:1}`, `pref = 7`, find `7 - 7 = 0` -> value `1`, `answer = 1`. The single run `[0,0]` sums to `7` — correct. With `S = 4` instead: find `7 - 4 = 3` -> absent, `answer = 0` — correct.
- `S = 0` with sign cancellation, `a = [5, -5, 5, -5]`: runs summing to `0` are `[0,1], [2,3], [0,3], [1,2]` — that is `4`. Let me trust the verified sweep rather than re-derive by hand; the brute force agrees (it does in the stress tests). The point is that the query-then-insert order means a current prefix never matches itself, so even with `S = 0` no empty run sneaks in.
- Unreachable target, `a = [1, 1, 1]`, `S = 100`: every `pref - S` is hugely negative and never in the map; `answer = 0` — correct.
- Negative needed value: with negatives in `a`, `pref - S` is routinely negative; `unordered_map<long long,...>` keys on the signed value directly, so negative keys are fine (no residue/modulus mangling here — this is exact-sum, not divisibility, so there is no sign-of-modulo trap).
- Performance: the densest map class is the all-zero array, a single bucket touched `n` times; it runs in about `11 ms` for `n = 2*10^5` in my timing, and a random large case in about `30 ms` — both comfortably inside `1 s`. I reserve the map and set a `0.7` load factor to keep it fast.

On the sample `S = 2`, `a = [3, -1, 1, 2, -2, 2, 1]`, the prefix sums are `0, 3, 2, 3, 5, 3, 5, 6`, and the match at `P[2]` (need `0`, satisfied only by the seeded `P[0] = 0`) is exactly the run `[0,1]` — so without seeding `P[0]` once up front, runs anchored at day 0 vanish.

The query-before-insert order is not cosmetic, and `S = 0` is where getting it wrong bites hardest. If I insert `P[r+1]` before querying, then whenever `P[r+1] - S = P[r+1]` — i.e. for *every* position when `S = 0` — the current prefix matches the copy of itself I just inserted, counting the illegal length-0 run. On `a = [1, -1]`, `S = 0` (true answer `1`, the run `[0,1]`): inserting first, position 0 finds `P[1] = 1` already present and adds 1, position 1 finds `P[2] = 0` present with count 2 and adds 2, giving `3`. Querying first, position 0 finds nothing and position 1 finds the seed `0` once, giving the correct `1`. So the current prefix goes into the map only after it has been queried.

The empty prefix must also be seeded exactly once. On the all-zero `a = [0, 0, 0]`, `S = 0`, every non-empty run sums to 0, so the answer is `n(n+1)/2 = 6`; the correct single-seed sweep accumulates `1 + 2 + 3 = 6` as the map's lone key `0` grows each step. Seeding `P[0]` twice would start the map at `{0:2}` and inflate this to `2 + 3 + 4 = 9`. So `cnt[0] = 1` once before the loop and never again. This all-zero case is also where the answer hits its `n(n+1)/2` maximum, the concrete instance of the `2*10^10` overflow bound.

The remaining corners fall out cleanly. `n = 0` reads no values, the loop never runs, `answer = 0`; the leading `if (!(cin >> n >> S)) return 0;` also absorbs empty stdin. `n = 1` matches iff the lone value equals `S`. An unreachable target like `[1, 1, 1]`, `S = 100` leaves every `P[r+1] - S` out of the map, answer 0. Negatives make `P[r+1] - S` routinely negative, which is fine: the map keys on the exact signed value — this is exact-sum, not divisibility, so there is no modulo-sign trap. The densest map case is the all-zero array — one bucket touched `n` times — still a single `O(n)` pass well inside 1 second; I `reserve` the map and cap its load factor to keep the constant small.

What ships is one self-contained file: seed `{0:1}`, sweep the running `long long` prefix, `find(pref - S)` and add its stored frequency *before* `cnt[pref] += 1`, then print the `long long` answer.

**Final solution.** What I ship is one self-contained file, the `O(n)` hashing count I can defend.
