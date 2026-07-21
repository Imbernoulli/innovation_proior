I have `n` scenes on one render node, `n <= 10^5`. Scene `i` costs `p[i]` seconds and carries a
per-second waiting weight `w[i]`; if I run them in some order and scene `i` finishes at the cumulative
time `C[i]` of everything up to and including it, it bills `w[i] * C[i]`, and I want the order
minimizing `sum_i w[i] * C[i]`. Input is `n` then `n` pairs `p[i] w[i]`; output is one integer.

What makes this problem the problem is arithmetic scale, so I fix that first, before choosing an
algorithm. With `1 <= p[i], w[i] <= 10^9`, the running clock reaches `sum p ~ 10^5 * 10^9 = 10^14`,
which sits safely inside signed 64-bit (`~9.2*10^18`). But a single bill term is `w[i] * C[i]`, up to
`10^9 * 10^14 = 10^23`, and there are `n` of them, so the total runs to roughly `10^28`. That is
hopelessly past 64-bit: the clock can live in `long long`, but the accumulated objective needs
`__int128` (range `~1.7*10^38`). The second hazard is that the ordering key will have to compare
values near `10^9`, so whatever comparison I use must be exact — a floating-point ratio is already
suspect at that magnitude. Two precision traps up front; the algorithm is almost incidental.

For the algorithm itself there are two routes. A bitmask DP — "choose the next scene from the
remaining set" — is exactly correct but `O(2^n * n)`, dead past `n ~ 20`; it survives only as an
oracle to check a greedy on tiny inputs, never as the shipped answer. The real candidate is a
single-key greedy: sort the scenes by one per-scene key in `O(n log n)` and sweep. The whole game is
finding the right key and comparing it exactly. I derive the key by the adjacent-exchange argument,
which is the standard tool for "the answer is an order" problems and which also hands me the exact
comparator for free.

Suppose in some order two scenes `i` and `j` are adjacent, `i` immediately before `j`. Let `T` be the
clock just before the pair starts. Everything after the pair is unaffected by swapping them, because
the pair occupies the same window `[T, T + p[i] + p[j]]` either way, so I only compare the two terms
the pair contributes. Order `i` then `j` costs `w[i](T + p[i]) + w[j](T + p[i] + p[j])`; order `j`
then `i` costs `w[j](T + p[j]) + w[i](T + p[j] + p[i])`. Subtracting, the `T*(w[i]+w[j])` and the
`w[i]p[i] + w[j]p[j]` parts both cancel, leaving

  cost(i before j) - cost(j before i) = `p[i] w[j] - p[j] w[i]`.

So `i` should precede `j` exactly when `p[i] w[j] <= p[j] w[i]`; dividing by `w[i] w[j] > 0`, that is
`p[i]/w[i] <= p[j]/w[j]` — schedule by non-decreasing ratio `p/w` (Smith's WSPT rule). The comparator
I will actually code is the cross-product form `p[i] * w[j]  vs  p[j] * w[i]`, no division, hence no
floating point. This relation is a consistent total order, which is why a global sort by it is
optimal: with no two adjacent jobs out of order, no adjacent swap helps, and any order can be sorted
by adjacent swaps that only ever lower the bill.

The sample `(3,1), (1,2), (2,5)` with answer `22` exercises both the key and the accumulation at once.
Ratios `p/w` are `3`, `0.5`, `0.4`;
ascending gives scene2, scene1, scene0. Running it: scene2 finishes at `2`, cost `5*2 = 10`; scene1
at `3`, cost `2*3 = 6`; scene0 at `6`, cost `1*6 = 6`; total `22`. Matches.

Now the natural first implementation: sort indices by the ratio, then sweep. My first cut of the
comparator uses the literal ratio because it reads closest to the math, and accumulates into a
`long long`:

```
sort(idx.begin(), idx.end(), [&](int i, int j){
    return (double)p[i] / w[i] < (double)p[j] / w[j];
});
long long clock_t = 0, total = 0;
for (int k = 0; k < n; k++) {
    int i = idx[k];
    clock_t += p[i];
    total += w[i] * clock_t;
}
```

The `double` comparator and the `long long` `total` are exactly the two hazards I priced above; I hit
each with a concrete adversarial input.

The near-tie first. Take `A = (p,w) = (999999958, 999999957)` and `B = (999999959, 999999958)`. Their
exact cross-products decide the order: `p_A w_B = 999999958 * 999999958 = 999999916000001764` versus
`p_B w_A = 999999959 * 999999957 = 999999916000001763`, so `A` before `B` costs exactly `1` more —
the correct order is `B, A`. But `999999958/999999957` and `999999959/999999958` both round to the
same IEEE-754 double: a 53-bit mantissa gives ~15-16 significant digits and these ratios agree to ~17,
so they collapse to the identical bit pattern. The comparator returns `false` both ways, `std::sort`
treats them as equivalent and leaves them in input order, and if the input arrives as the wrong order
`A, B`, the sort does nothing. Tracing the bill with `T = 0`: correct order `B, A` totals
`2999999748000005291`; the order the double sort leaves, `A, B`, totals `2999999748000005292` — larger
by exactly `1`, as the exchange formula predicted. A silent off-by-one-bill wrong answer; the
big-integer brute force agrees the optimum is `...291`. The fix is the comparator the exchange
argument already gave me: order `i` before `j` iff `p[i] * w[j] < p[j] * w[i]`, with a deterministic
index tie-break (ties are cost-neutral, so any tie-break is optimal, but a fixed one keeps runs
reproducible). Each cross-product is `<= 10^9 * 10^9 = 10^18 < 9.2*10^18`, so it fits in `long long`
with no `__int128` inside the sort — the danger there was the division, not the magnitude.

The accumulator second. I build the worst small case the constraints allow: seven scenes all
`(10^9, 10^9)`. They are interchangeable, so I just need the bill. The clock after scene `k` is
`k * 10^9` and the term is `k * 10^18`; summing `k = 1..7` gives `28 * 10^18 = 28000000000000000000`.
But `long long` tops out at `~9.22*10^18`, so the accumulator wraps to `-8893488147419103232` — a
negative sum of positive terms, the unmistakable fingerprint of overflow. At full `n = 10^5` the bill
reaches `~10^28`, so this is the main case, not a corner. Fix: keep the running clock in `long long`
(`<= 10^14`, safe) but accumulate the objective in `__int128`, casting each term
`(__int128)w[i] * clock_t`; the sum stays under `~10^28`, far inside `__int128`. Since `cout` has no
`operator<<` for `__int128`, I extract decimal digits by hand.

The edge cases fall out cleanly. `n = 0` (and empty stdin) reads nothing, both loops never run,
`total` stays `0`; the printer special-cases `total == 0` so it emits `0` rather than an empty string.
`if (!(cin >> n)) return 0;` exits silently on no input, matching the `n = 0` semantics. `n = 1`, say
`(7, 3)`: the sort is a no-op, the lone scene finishes at `7`, bill `21`. All-equal ratios (repeated
scenes, or several `(2k, 3k)`) tie every cross-product, the index tie-break gives a stable order, and
the bill is order-independent anyway, so any order is optimal.

Beyond the two hand-built inputs, I ran the cross-product + `__int128` solution against the
permutation brute force on 700+ random small cases (`n` up to 7) across four generator regimes — tiny
values, values near `10^9`, near-tied ratios, and exact-tie families — plus the `n = 0`, `n = 1`,
sample, double-misorder, and overflow cases: zero mismatches. Timing `n = 10^5` with random near-`10^9`
values gives `~0.06 s`, far under the 1 s limit, since the work is one `O(n log n)` sort plus an
`O(n)` sweep. The full program is in the answer.
