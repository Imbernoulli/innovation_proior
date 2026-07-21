The gauge reads `0` before the season and then takes `n` signed daily changes `d[0..n-1]`, so what I am really after is the deepest peak-to-trough fall of a prefix-sum series — a maximum drawdown — and two features of this particular statement drive everything. The readings run `L[0..n]` inclusive, with `L[0] = d[0]+...` summed to zero terms equal to `0`: the pre-season reference is itself a legitimate earlier reading, so a fall can be measured all the way from the season's start. And `i = j` is allowed, giving drawdown `L[i] - L[i] = 0`, so the answer is floored at `0` — a season that only ever rises reports `0`, not a negative number. That `L[0] = 0` participates and that the floor is `0` are exactly the two hinges the sign and base-case traps swing on.

Scale sets the types. `n` up to `2*10^5` and `|d[i]|` up to `10^9`, so a level can reach `2*10^5 * 10^9 = 2*10^14` in magnitude and a difference of two levels roughly `4*10^14` — well past the 32-bit ceiling of about `2.1*10^9`. Every accumulator (the running level, the running peak, the answer) is `long long`; an `int` here is a silent wrong answer on the large tests, not a crash, which is the worse failure.

The literal definition is all-pairs over `L[0..n]`: loop every `i <= j` and take the max of `L[i] - L[j]`. Transparently correct, but `O(n^2) ≈ 4*10^10` at the top end — hopeless in one second, so it survives only as an oracle to check the fast route against. The collapse that makes it fast: for a fixed later day `j`, the deepest fall ending there uses the *largest* earlier-or-equal reading, the running peak `peak_j = max(L[0..j])`. So the best drawdown ending at `j` is `peak_j - L[j]`, and

```
maxDrawdown = max over all i <= j of (L[i] - L[j])  ==  max over j of (peak_j - L[j]).
```

Both directions hold. Right `<=` left: `peak_j = L[i*]` for some `i* <= j`, so `peak_j - L[j] = L[i*] - L[j]` is one concrete pair, hence bounded by the overall max. Left `<=` right: any pair has `L[i] <= peak_j` by definition, so `L[i] - L[j] <= peak_j - L[j]`. That collapses the inner loop to a single left-to-right walk carrying `level` and `peak` — `O(n)` time, `O(1)` extra memory.

Now the base value, which is where this problem sets its first trap. `peak_0 = max(L[0]) = L[0] = 0`, because the only reading at or before day 0 is the pre-season `0`. So `peak` seeds at `0` — not `-infinity`, not `d[0]`. A `-infinity` seed would throw away the fall-measured-from-the-reference, which is precisely what gives a single outflow day its magnitude and an all-negative log its full accumulated decline; seeding at `d[0]` would silently drop the day-0 reference from the peak.

The second trap is the direction of the subtraction, and it is easy to invert because maximum drawdown and maximum *rise* are mirror images. A drawdown is `earlier_high - later_low = peak - level`, with `peak` the running maximum. The mirror, `level - min` with `min` the running minimum, is the maximum-subarray / max-profit shape — and it is invisible on any all-rising input, only diverging once a real fall appears. On `d = [10, -30]` (levels `0, 10, -20`, true answer `30`) the fall form gives `peak(10) - level(-20) = 30`; the rise form would report `10`, the biggest gain instead of the biggest loss. So `peak` stays a running maximum and I measure `peak - level`.

One ordering question remains: measure the drop before or after folding the current day into the peak? Either gives the same number — when `level` exceeds `peak` the freshly-updated peak equals `level` and contributes a zero drop, and otherwise the update leaves `peak` untouched — so I measure the drop first, which makes the loop body read straight off `peak_j - L[j]`:

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
