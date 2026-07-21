Everything about the sizes points at a sort. `n` up to `2*10^5` rules out anything worse
than `O(n log n)`, and the shape of the cost `sum_i w[i]*C[i]` — fix an order, read off the
answer — is exactly what a single sort key would produce. So the real question is *which* key,
and a second look at the sizes settles the arithmetic before I choose one. The worst case is
every job long and heavy, `t = w = 10^4`; the `k`-th completion time is then `k*10^4`, so the
total climbs to `10^4 * 10^4 * n(n+1)/2 ≈ 10^8 * 2*10^10 = 2*10^18`. That sits under the signed
64-bit ceiling `~9.2*10^18` but a thousandfold over the 32-bit `~2.1*10^9`. Every accumulator is
`long long`; an `int` here is a silent wrong answer on the largest test, not a crash, which is the
worst kind. Settled before I touch the ordering.

Now the key. The two candidates I would reach for without thinking each ignore exactly the number
the other one looks at. Shortest-processing-time-first (sort by `t` ascending) is provably optimal
when all weights are equal — it minimizes the unweighted sum of completion times — but it never sees
weight, so a long yet very heavy job that ought to run early stays late. Heaviest-first (sort by `w`
descending) is the mirror: optimal when all lengths are equal, blind to the fact that a heavy but
long job delays everything behind it. The cost multiplies the two numbers together, so a key that
reads only one of them cannot be right in general. I won't commit to either until one breaks on a
concrete instance.

Shortest-job-first is easy to break. Three jobs `A=(1,1), B=(3,5), C=(2,1)`. Sorting by `t` gives
`A,C,B`, completion times `1,3,6`, cost `1*1 + 1*3 + 5*6 = 34`. Pull the heavy `B` to the front
instead — `B,A,C`, completion times `3,4,6`, cost `5*3 + 1*4 + 1*6 = 25`. Strictly better, and the
reason is visible: `B` carries weight 5, so a unit of delay on `B` costs five times a unit of delay
on the weight-1 jobs; buying `B` an earlier finish by delaying two light jobs is a bargain.
Symmetrically, heaviest-first must fail on a heavy-but-long job parked ahead of many short ones.
Both single keys are out; I need one that weighs time against weight together.

The exchange argument is the tool. Take any schedule, look at two adjacent jobs, and ask whether
swapping them lowers the cost — everything outside the pair is untouched, since the jobs after them
shift by the same `t[i]+t[j]` either way. Let `P` be the time accumulated before the pair. Order
`i` then `j` contributes `w[i]*(P+t[i]) + w[j]*(P+t[i]+t[j])`; order `j` then `i` contributes
`w[j]*(P+t[j]) + w[i]*(P+t[j]+t[i])`. Subtract them: the `(w[i]+w[j])*P` terms cancel, the
`w[i]*t[i]` and `w[j]*t[j]` self-terms cancel, and what remains is

```
(i before j) - (j before i) = w[j]*t[i] - w[i]*t[j].
```

So `i` belongs no later than `j` exactly when `t[i]*w[j] <= t[j]*w[i]` — dividing through by the
positive `w[i]*w[j]`, ascending order of the ratio `t/w`, Smith's rule. The preference is total and
transitive (it is the order induced by the real number `t/w`), so a schedule sorted by it admits no
improving adjacent swap, and since adjacent swaps connect every permutation to every other, it is
globally optimal. This is the coupled comparator the cost was asking for, neither `t` alone nor `w`
alone, collapsing to each only in the degenerate cases that made those single keys look plausible.
On the three jobs the ratios are `B=0.6, A=1, C=2`, order `B,A,C`, cost 25 — the optimum I found by
hand, now with a reason under it.

Two things about the comparator have to be right in code, and both are specific to how `std::sort`
and these value ranges behave. First, equal ratios. Jobs like `(2,1), (4,2), (6,3)` all have ratio
2, and there the cross-product difference `w[j]*t[i] - w[i]*t[j]` is exactly 0 — either order costs
the same, the cost is tie-invariant. But `std::sort` demands a strict weak ordering: a comparator
that returns true both ways on such a pair is undefined behavior. The naive "ascending, so use `<=`"
is precisely the trap — `t[i]*w[j] <= t[j]*w[i]` makes both `cmp(i,j)` and `cmp(j,i)` true on a tie,
which can crash or scramble the output on inputs built from many equal ratios. Keep the comparator
strict (`<`) and break ties by index, so the order is deterministic instead of
implementation-defined; the answer is unchanged because ties never move the cost. Second, no floats:
compare the cross-multiplied integers `t[i]*w[j]` against `t[j]*w[i]`, never the `double` ratio
`t/w`, because two distinct ratios that lie very close can round to the same double and silently
violate that same ordering contract. The products are at most `10^4 * 10^4 = 10^8`, nowhere near the
64-bit limit, so cross-multiplication is exact and safe.

That fixes the comparator and the one-pass accumulation behind it:

```cpp
sort(idx.begin(), idx.end(), [&](int i, int j) {
    long long lhs = t[i]*w[j], rhs = t[j]*w[i];
    if (lhs != rhs) return lhs < rhs;   // ascending t/w, exact
    return i < j;                        // deterministic tie-break
});
long long cost = 0, clock = 0;
for (int k = 0; k < n; k++) { int i = idx[k]; clock += t[i]; cost += w[i]*clock; }
```

The two degenerate cases pin down the comparator's *direction*, the easiest thing to get backwards.
With all weights equal it reduces to `t[i] < t[j]` — shortest-job-first, the known unweighted
optimum. With all times equal it reduces to `w[j] < w[i]`, i.e. `i` first when `w[i] > w[j]` —
heaviest-first, right when lengths are equal. Both single-key optima fall out of the coupled key in
the correct direction, which is what tells me the direction is right and not flipped. And `n = 0`
reads cleanly, runs no loop, prints 0.

To be sure the ordering is optimal and not merely plausible, I check it against a brute-force oracle
that tries all `n!` permutations and takes the minimum — correct by construction, usable up to
`n ≈ 8`. Across random small instances spanning equal-ratio ties, extreme time-versus-weight skew,
and plain random values, plus the empty/single/equal-weight/equal-time corners, the sort matches the
brute minimum every time, including the `34`-vs-`25` instance that motivated the whole design.
`O(n log n)` on `n = 2*10^5` is comfortably under the one-second limit. The full self-contained
program is in the answer.
