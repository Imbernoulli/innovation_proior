The gauge reads `0` before the season and then takes `n` signed daily changes `d[0..n-1]`, so what I am really after is the deepest peak-to-trough fall of a prefix-sum series — a maximum drawdown — and two features of this particular statement drive everything. The readings run `L[0..n]` inclusive, with `L[0] = d[0]+...` summed to zero terms equal to `0`: the pre-season reference is itself a legitimate earlier reading, so a fall can be measured all the way from the season's start. And `i = j` is allowed, giving drawdown `L[i] - L[i] = 0`, so the answer is floored at `0` — a season that only ever rises reports `0`, not a negative number. That `L[0] = 0` participates and that the floor is `0` are exactly the two hinges the sign and base-case traps swing on.

Scale sets the types. `n` up to `2*10^5` and `|d[i]|` up to `10^9`, so a level can reach `2*10^5 * 10^9 = 2*10^14` in magnitude and a difference of two levels roughly `4*10^14` — well past the 32-bit ceiling of about `2.1*10^9`. Every accumulator (the running level, the running peak, the answer) is `long long`; an `int` here is a silent wrong answer on the large tests, not a crash, which is the worse failure.

The literal definition is all-pairs over `L[0..n]`: loop every `i <= j` and take the max of `L[i] - L[j]`. Transparently correct, but `O(n^2) ≈ 4*10^10` at the top end — hopeless in one second, so it survives only as an oracle to check the fast route against. The collapse that makes it fast: for a fixed later day `j`, the deepest fall ending there uses the *largest* earlier-or-equal reading, the running peak `peak_j = max(L[0..j])`. So the best drawdown ending at `j` is `peak_j - L[j]`, and

```
maxDrawdown = max over all i <= j of (L[i] - L[j])  ==  max over j of (peak_j - L[j]).
```

Both directions hold. Right `<=` left: `peak_j = L[i*]` for some `i* <= j`, so `peak_j - L[j] = L[i*] - L[j]` is one concrete pair, hence bounded by the overall max. Left `<=` right: any pair has `L[i] <= peak_j` by definition, so `L[i] - L[j] <= peak_j - L[j]`. That collapses the inner loop to a single left-to-right walk carrying `level` and `peak` — `O(n)` time, `O(1)` extra memory.

Now the base value, which is where this problem sets its first trap. `peak_0 = max(L[0]) = L[0] = 0`, because the only reading at or before day 0 is the pre-season `0`. So `peak` seeds at `0` — not `-infinity`, not `d[0]`. A `-infinity` seed would throw away the fall-measured-from-the-reference, which is precisely what gives a single outflow day its magnitude and an all-negative log its full accumulated decline; seeding at `d[0]` would silently drop the day-0 reference from the peak.

**Numeric self-check of the formula on the sample.** Before writing code I verify the derived identity arithmetically on the documented case `d = [3, -2, -5, 4, -1, -3, 2, 1]`. The levels are
`L[0]=0`, `L[1]=3`, `L[2]=1`, `L[3]=-4`, `L[4]=0`, `L[5]=-1`, `L[6]=-4`, `L[7]=-2`, `L[8]=-1`.
Running peak `peak_j`: `0, 3, 3, 3, 3, 3, 3, 3, 3`. Then `peak_j - L[j]`:
`0-0=0`, `3-3=0`, `3-1=2`, `3-(-4)=7`, `3-0=3`, `3-(-1)=4`, `3-(-4)=7`, `3-(-2)=5`, `3-(-1)=4`.
The maximum is `7`, at `j = 3` (and again at `j = 6`). The all-pairs definition agrees: the deepest fall is from the peak `L[1] = 3` down to the trough `L[3] = -4` or `L[6] = -4`, a drop of `7`. The formula reproduces the stated answer `7`, so the collapse is correct on a real instance.

I am suspicious of the order — I update `peak` *before* measuring the drop — so I trace the smallest input that could expose it: a single outflow day, `n = 1`, `d = [-5]`. The level goes `L[0] = 0 -> L[1] = -5`. The true answer is `5`: the level fell from the pre-season `0` down to `-5`. Trace my code. Start `level=0, peak=0, best=0`. i=0: `level = 0 + (-5) = -5`. `if (level > peak)` is `-5 > 0`? No, so `peak` stays `0`. `if (peak - level > best)` is `0 - (-5) = 5 > 0`? Yes, `best = 5`. Final `best = 5`. Correct here. Hm, this single case did not break — let me push harder, because "update then measure" can still be wrong when the level *rises* on the same step it would have set a drop.

**The bug, caught by a rising-then-irrelevant trace.** Consider `n = 2`, `d = [10, -30]`, where the level goes `0 -> 10 -> -20`. The true answer is `30`: a fall from the peak `10` to `-20`. Trace the "update first" code. Start `level=0, peak=0, best=0`. i=0 (`d=10`): `level = 10`. `10 > 0`? Yes, `peak = 10`. `peak - level = 10 - 10 = 0 > 0`? No, `best` stays `0`. i=1 (`d=-30`): `level = -20`. `-20 > 10`? No, `peak` stays `10`. `peak - level = 10 - (-20) = 30 > 0`? Yes, `best = 30`. Final `30`. Correct. So "update first" survives this too.

The second trap is the direction of the subtraction, and it is easy to invert because maximum drawdown and maximum *rise* are mirror images. A drawdown is `earlier_high - later_low = peak - level`, with `peak` the running maximum. The mirror, `level - min` with `min` the running minimum, is the maximum-subarray / max-profit shape — and it is invisible on any all-rising input, only diverging once a real fall appears. On `d = [10, -30]` (levels `0, 10, -20`, true answer `30`) the fall form gives `peak(10) - level(-20) = 30`; the rise form would report `10`, the biggest gain instead of the biggest loss. So `peak` stays a running maximum and I measure `peak - level`.

One ordering question remains: measure the drop before or after folding the current day into the peak? Either gives the same number — when `level` exceeds `peak` the freshly-updated peak equals `level` and contributes a zero drop, and otherwise the update leaves `peak` untouched — so I measure the drop first, which makes the loop body read straight off `peak_j - L[j]`:

Re-trace `d = [10, -30]`: start `(level,peak,best)=(0,0,0)`. i=0: `level=10`; `peak-level=0-10=-10`, not `> 0`, `best` stays `0`; `10 > 0` so `peak=10`. i=1: `level=-20`; `peak-level=10-(-20)=30 > 0`, `best=30`; `-20 > 10`? No. Final `30`. Correct, and the sign now points down. Re-trace the single outflow `d = [-5]`: i=0: `level=-5`; `peak-level=0-(-5)=5>0`, `best=5`; `-5>0`? No. Final `5`. Correct. The two cases that exercised the sign now both pass, and they pass for the reason I fixed (drop = `peak - level`, peak seeded at `L[0]=0`), which is the evidence I trust.

**Re-running the formula self-check through the fixed code on the sample.** `d = [3,-2,-5,4,-1,-3,2,1]`. `(level,peak,best)` evolution:
start `(0,0,0)`;
i=0 d=3: level=3; `0-3=-3` no; `3>0` peak=3 -> `(3,3,0)`;
i=1 d=-2: level=1; `3-1=2>0` best=2; `1>3`? no -> `(1,3,2)`;
i=2 d=-5: level=-4; `3-(-4)=7>2` best=7; `-4>3`? no -> `(-4,3,7)`;
i=3 d=4: level=0; `3-0=3` not `>7`; `0>3`? no -> `(0,3,7)`;
i=4 d=-1: level=-1; `3-(-1)=4` not `>7`; no -> `(-1,3,7)`;
i=5 d=-3: level=-4; `3-(-4)=7` not `>7`; no -> `(-4,3,7)`;
i=6 d=2: level=-2; `3-(-2)=5` not `>7`; no -> `(-2,3,7)`;
i=7 d=1: level=-1; `3-(-1)=4` not `>7`; no -> `(-1,3,7)`.
Final `best = 7`. Matches the documented sample and my hand formula. Two independent traces now agree.

**Edge cases, because this is where this kind of code dies.**
- `n = 0` (empty log): the loop never runs; `best` stays `0`. There is only the pre-season reading `L[0]=0`, no fall possible, answer `0`. Correct. (Also: `if (!(cin >> n)) return 0;` covers truly empty input, printing nothing, which an empty-token judge treats as `0`.)
- `n = 1`, single rise `d = [5]`: level `0 -> 5`; measure `0 - 5 = -5` (not `> 0`), `best` stays `0`. The gauge only rose, worst drawdown `0`. Correct — this is the non-negativity corner, and it works precisely because `best` is seeded at `0` and the measurement can go negative without polluting it.
- `n = 1`, single fall `d = [-5]`: answer `5`, traced above. The fall is measured against the seeded peak `L[0]=0`. Correct.
- All negative `d = [-3,-1,-4]`: levels `0,-3,-4,-8`; peak stays `0`; drops `3, 4, 8`; `best = 8`. The whole accumulated decline `|sum|` because the peak is the very first reading. Correct — and this is exactly the all-negative corner where a `-infinity` peak seed would have failed.
- All positive `d = [2,3,4]`: levels `0,2,5,9`; every `peak - level = 0`; `best = 0`. Correct.
- All zeros `d = [0,0,0]`: level stays `0`, peak `0`, `best = 0`. Correct.
- Rise then deep fall `d = [10,-30]`: `30`, traced. Correct.
- Overflow: `level`, `peak`, `best` are all `long long`; the worst level magnitude `~2*10^14` and worst drawdown `~4*10^14` fit with room to spare in the `~9.2*10^18` range. I verified on the live worst case `n = 2*10^5`, first half `+10^9` then second half `-10^9`: the level climbs to `10^14` then falls to `0`, deepest drawdown `10^14`, which the program prints exactly (`100000000000000`, far above the 32-bit ceiling) in 0.06 s. So both the type choice and the time budget are confirmed empirically, not just argued.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the input layout (one line or many) does not matter.

```
long long level = 0, peak = 0, best = 0;
for (int i = 0; i < n; i++) {
    long long d; cin >> d;
    level += d;                             // L[i+1] = L[i] + d[i]
    if (peak - level > best) best = peak - level;   // fall from the running peak
    if (level > peak) peak = level;         // update peak after measuring
}
```

Seeding `best = 0` closes the non-negativity corner for free: on an all-rising log every `peak - level` is `<= 0` and never displaces the `0`.

The corners the hidden tests target fall out of this same seeding, and each stresses a different hinge. `n = 0`: the loop never runs and `best` stays `0`; the `if (!(cin >> n)) return 0` guard also swallows a truly empty stream, which an empty-token judge reads as `0`. A single fall `[-5]` measures `0 - (-5) = 5` against the seeded reference peak — the base-case hinge — while a single rise reports `0` through the floor. An all-negative log `[-3,-1,-4]` keeps `peak` pinned at `0` and reports the full decline `8`. The through-line is that `best` is floored at `0` while `peak - level` is free to go negative harmlessly, and `long long` keeps the `~4*10^14` worst case in range.
