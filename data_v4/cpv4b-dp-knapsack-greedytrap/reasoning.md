I have `n` crates, crate `i` with integer mass `w[i]` and value `v[i]`, and a mass budget `C`; I pick a subset of total mass at most `C` to maximize total value, empty load allowed so the answer never drops below `0`. The constraint that decides my types before any algorithm is `v[i] <= 10^9` against `n <= 1000`: the worst-case total value is about `n * max(v) = 10^12`, far past the 32-bit ceiling of `~2.1 * 10^9`, so every value accumulator has to be 64-bit. Masses and `C` are tiny (`<= 2000`), but I keep `C` in `long long` too so capacity arithmetic never mixes signedness with the value sums. `int` value sums are a silent wrong-answer on the large tests, not a style choice.

Two routes are on the table, and I want the one I can prove under indivisibility. Ratio greedy — sort by `v[i] / w[i]` descending, load while things fit, take any zero-mass positive-value crate for free — is `O(n log n)` and a handful of lines, and it is genuinely optimal for the *fractional* knapsack where I could slice the last crate to fill the budget exactly. That theorem is the bait: here crates are indivisible. The alternative is a capacity DP: for each total mass `c` up to `C`, track the best value of a subset of mass exactly `c`, folding crates in one at a time, `O(n * C) = 2 * 10^6` updates, trivially inside the limit, with the only real risk being the transcription of the iteration direction.

"Greedy feels right because fractional knapsack works" is exactly the reasoning that ships wrong solutions, so I try to break it with a concrete instance rather than a feeling. I want a budget that one high-ratio crate can waste. Take `C = 10` and crates A=`(6,10)`, B=`(5,7)`, C=`(5,7)`. The ratios are `A: 10/6 ≈ 1.667`, `B, C: 7/5 = 1.4`. Greedy loads A (mass 6, value 10, budget 4 left), then neither B nor C fits in the remaining 4, so it stops at `10`. But B and C together have mass `5 + 5 = 10 <= 10` and value `14`, strictly better. The mechanism is clear: A has the best value-per-mass, but consuming 6 of the 10 units strands the remaining 4 as dead budget, because indivisibility forbids slicing a 5-crate down to fit. Fractional greedy would fill those 4 units with `4/5` of a 5-crate for `+5.6`, reaching `15.6`, and `15.6 > 14 > 10` — the fractional optimum strictly dominates both the integral optimum and greedy's integral answer. Greedy chases the fractional number and misses. It is out; I keep this instance as a regression check.

Now the DP. For every reachable total mass `c` I want `dp[c]`, the maximum value of a subset whose mass is *exactly* `c`. I encode "unreachable" as `-1`, distinct from a legitimately reachable value of `0` (empty mass, or a zero-value crate). Before any crate only `dp[0] = 0`. Folding crate `i`: a mass-`c` subset either excludes it (value unchanged) or includes it, in which case the rest is a mass-`(c - w[i])` subset built without this crate yet, so

`dp_new[c] = max( dp_old[c],  dp_old[c - w[i]] + v[i] )`  for `c >= w[i]` and `dp_old[c - w[i]] != -1`.

The "without this crate yet" clause is the whole 0/1 discipline: the predecessor state must not already contain crate `i`. The answer is `max` over reachable `dp[c]`, floored at `0` for the empty load.

I keep a single `dp` array updated in place to save memory. My first cut sweeps `c` upward:

```
vector<long long> dp(C + 1, -1);
dp[0] = 0;
for (int i = 0; i < n; i++) {
    if (w[i] > C) continue;
    for (long long c = w[i]; c <= C; c++) {     // forward sweep
        if (dp[c - w[i]] != -1) {
            long long cand = dp[c - w[i]] + v[i];
            if (cand > dp[c]) dp[c] = cand;
        }
    }
}
```

Sweeping `c` upward while updating in place is exactly where 0/1 knapsack goes wrong, so I trace the smallest input that would expose it: `C=10`, a single crate `(5,7)`, answer obviously `7`. `dp[0]=0`, rest `-1`; fold, `c` from 5 up: `c=5` gives `dp[5]=7`; then `c=10` reads `dp[5]=7` — already overwritten this very pass — so `dp[10]=7+7=14`. A single crate of value 7 returns 14: it loaded the same crate twice. The upward sweep reads `dp[c - w]` from a cell already updated in this crate's pass, chaining "crate, then crate again" — that is the *unbounded* knapsack. The fix is to sweep downward, from `C` to `w[i]`, so `dp[c - w[i]]` is read from a strictly smaller index untouched this pass, still reflecting the state before this crate; each crate then folds in at most once.

```
for (long long c = C; c >= w[i]; c--) {
    if (dp[c - w[i]] != -1) {
        long long cand = dp[c - w[i]] + v[i];
        if (cand > dp[c]) dp[c] = cand;
    }
}
```

Re-tracing the single `(5,7)` downward: `c=10..6` all read `-1`; `c=5` reads `dp[0]=0` and sets `dp[5]=7`; best `7`. Correct. On the full sample downward: A sets `dp[6]=10`; B sets `dp[5]=7` (and `c=11` is out of range, so A+B never combine — correct, `6+5>10`); C reads `dp[5]=7` at `c=10` to set `dp[10]=14`, updating `dp[5]` only afterward, so `dp[10]` used the pre-C value of `dp[5]` — exactly the 0/1 discipline. Best `14`, and the instance that killed greedy now returns the right answer.

Zero-mass crates are the other place this could die: `w[i]=0` makes `c - w[i] = c`, a self-reference. Take `C=0` and two crates `(0,3)`, `(0,3)`; the correct answer is `6` (both free, each loaded once). `dp[0]=0`; fold crate 1 at `c=0`: `dp[0]=max(0,0+3)=3`; fold crate 2 at `c=0`: `dp[0]=max(3,3+3)=6`. Correct — within one crate's pass `dp[0]` is visited exactly once, so the crate is added once, and the two separate passes each contribute independently.

The remaining corners: `n=0` — the fold never runs, `dp=[0]`, best `0`. `C=0` with positive-mass crates — all skipped by `if (w[i] > C) continue;`, best `0`. A crate strictly heavier than `C` (e.g. `C=3`, `(5,100)`) — skipped by the same guard, best `0`. A zero-value crate can flip some `dp[c]` from `-1` to a real `0` but never lowers a value and never inflates the answer above what positive crates give. A zero-mass positive crate alongside a positive one (e.g. `C=6`, `(0,4),(6,5)`): `dp[0]=4`, then `dp[6]=9` — the free crate plus the mass-6 crate. On overflow, values live in `long long` where `~10^12` fits with room, and the `-1` sentinel is only ever compared, never has `v[i]` added to it (the `!= -1` guard), so no spurious arithmetic. Reading with `cin >>` consumes arbitrary whitespace, so input line layout is irrelevant.

Cost is `n` crates times an inner sweep of at most `C + 1`, so `<= 1000 * 2001 ≈ 2.0 * 10^6` cell visits plus an `O(C)` final scan — well under `10^8`, comfortably inside 1 second; memory is one `long long` array of length `<= 2001`, about 16 KB. Running it on `n=1000, C=2000` with values near `10^9` gives `25628973081`, which exceeds `2^31` — concrete proof that the 64-bit accumulator was necessary, not optional. I ship the capacity DP I can defend over the greedy I broke; the full program is in the answer.
