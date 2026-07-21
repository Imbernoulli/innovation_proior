The size line decides the shape before any algorithm does. `n` up to `10^7` rules out anything superlinear and forbids character-by-character output — the answer, up to ten million characters, has to be assembled once and flushed in a single write, or the I/O alone blows the 2 s limit. The ordering is the other quirk: lexicographically smallest under `H < R` makes a hit the *smaller* symbol, so "earliest hit" and "smallest string" are the same objective, and the whole game is to place hits as early as the rules allow. And any feasibility count I write will be a product like `(h+1)*(K-1)`; with `h` and `K` both near `10^7` that reaches `~10^14`, so every such count lives in 64-bit — an `int` here is a silent wrong answer on exactly the large tests and invisible on anything I can hand-check. Three integers `n K h` in, one string or `-1` out.

The one rule with teeth is "no `K` rests in a row". Restate it structurally: the `h` hits chop the timeline into `h + 1` gaps — before the first hit, `h - 1` between hits, one after the last — and the `n - h` rests fall into those gaps, each gap holding at most `K - 1` rests. So a legal loop exists iff

```
n - h <= (h + 1) * (K - 1)        (with 0 <= h <= n, K >= 1).
```

The `+1` is the whole ballgame; counting only the interior gaps undercounts capacity and rejects legal inputs. The witness that pins it down is `n=3, K=2, h=1`: true capacity `(1+1)*1 = 2` admits `RHR`, whereas the off-by-one `h*(K-1) = 1` would reject it. Two extreme regimes confirm the form — `K=1` gives capacity `0`, feasible only when `h = n` (no rest allowed at all, exactly right); and `n=7, K=3, h=1` gives capacity `2*2 = 4 < 6` rests, infeasible, since a single hit's two gaps cannot absorb six rests.

A fixed period-`K` block (`H` then `K-1` rests, repeated) trivially avoids long silence, but it emits its own hit count `~n/K` rather than the demanded `h`, and it isn't lexicographically minimal — a decoy that passes only when `h` happens to match what it produces. The real method must prefer `H` at each beat yet respect the count, which forces a *guarded* greedy.

Preferring `H` unconditionally gives `H...HR...R` — lexicographically smallest ignoring the rest rule, but it dumps all `n - h` rests into one trailing run, illegal the moment `n - h >= K`. So the greedy has to check before it grabs: a hit spent now might leave too few hits behind to break up the trailing rests. I need a suffix-feasibility test. Let `canFill(m, hh, c)` mean I can fill `m` slots with `hh` hits (and `m - hh` rests), no `K`-run, given a run of `c` rests already standing immediately before slot 0. The `hh` hits make `hh + 1` gaps; the first is glued onto the standing run so it holds only `K-1-c` more, and each of the others holds `K-1`:

```
canFill(m, hh, c)  iff  0 <= hh <= m  AND  c <= K-1  AND  (m - hh) <= (hh+1)*(K-1) - c
```

Global feasibility is `canFill(n, h, 0)`. The sweep: at a beat with `m` slots after it, `hitsLeft` hits, trailing run `c`, place `H` iff `hitsLeft >= 1 && canFill(m, hitsLeft-1, 0)` (the hit resets the run), else `R`. Because the start state is feasible and every action lands in a feasible state, the `R` branch never gets stuck mid-string.

On the sample `n=7, K=3, h=2` the guard earns its keep at beat 0: taking `H` there would require `canFill(6,1,0)`, i.e. `5 <= (1+1)*2 = 4`, false — one remaining hit can't split five rests into runs of length `<= 2` — so the greedy declines the smaller symbol and writes `R`. Beat 1 then admits `H` (`canFill(5,1,0)`: `4 <= 2*2`), and the tail is forced, yielding `RHRRHRR`. That refusal — passing over `H` although it is lexicographically smaller — is the crux of the problem.

One clause in `canFill` is not decorative: the `c <= K-1` reject. Drop it and the capacity term `(hh+1)*(K-1) - c` can stay non-negative even when the standing run `c` has already hit `K`, because a large `hh` inflates the term. Concretely `canFill(0, 2, 3)` with `K=3` computes capacity `3*2 - 3 = 3` and reads `-2 <= 3` as feasible — yet a run of `c = 3 = K` is already a violation no matter how many hits remain. The explicit `c <= K-1` is what makes the `R`-branch guard actually block `K`-runs; without it the very check meant to stop `R` from creating a long silence is toothless in the corner where the hit budget is large.

The last hazard is arithmetic, not logic. `(hh+1)*(K-1)` reaches `~3*10^14` in the worst case (`~10^7 * 10^7`). Computed in 32-bit it overflows: on the feasible `n = K = 10^7, h = 300`, capacity `(300+1)*(10^7 - 1) = 3,009,999,699` wraps past the signed-32 ceiling `2,147,483,647` to a negative, the feasibility test reads `9999700 <= (negative)` as false, and the program prints `-1` on an instance with a real answer (300 hits then rests — such a huge `K` forbids almost nothing). It matches a brute force on every small case and fails only at scale, which is precisely how the bug would ship. So all counts are `long long`; the product then evaluates in 64-bit and `~3*10^14` fits with room.

Edge cases, each run through the same guard: `n=1,K=1,h=1` → `H`; `n=1,K=1,h=0` → capacity `0 < 1`, so `-1` (a lone rest with `K=1` is illegal); `K=1` is feasible only at `h=n`; `h=0` is feasible iff `n <= K-1` (`n=2,K=3` → `RR`, `n=3,K=3` → `-1`); `h=n` is all `H`s. The tight boundary `n=8,K=3,h=2` packs every gap to exactly `K-1`, giving `RRHRRHRR`; one beyond, `n=9,K=3,h=2`, has `7 > 6` rests and prints `-1`. At `n=10^7` the answer is a single `std::string`, reserved once and written once — inside 2 s and 256 MB.

The construction runs in `O(n)` time with `O(1)` state beyond the output buffer, so a single left-to-right sweep and one buffered write clear the 2 s limit even at `n = 10^7`; the full program is the answer.
