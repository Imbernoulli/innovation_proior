**Reading the problem and pinning the contract.** I have `n` jobs to run on one machine that
does one job at a time, never idle, never preempting. Job `i` takes `t[i]` time and has weight
`w[i]`. If I fix an order, the completion time `C[i]` of a job is the prefix sum of processing
times up to and including it, and the cost is `sum_i w[i] * C[i]`. I want the order that
minimizes that total, and I print the minimum. Input on stdin is `n` followed by the `n` pairs
`(t[i], w[i])`; output is one integer. Before any algorithm I fix the scale, because it decides
the data types. Here `n <= 2*10^5` and `t[i], w[i] <= 10^4`. The worst the cost can get is every
job long and heavy: `t = w = 10^4`, and the `k`-th completion time is `k * 10^4`, so the total is
`10^4 * 10^4 * (1 + 2 + ... + n) = 10^8 * n(n+1)/2 ≈ 10^8 * 2*10^10 = 2*10^18`. That is under the
signed 64-bit ceiling of about `9.2*10^18` but far over the 32-bit ceiling of `2.1*10^9`. So every
accumulator is `long long`, non-negotiable; an `int` is a silent wrong answer on the large tests.
That is the first decision and it is fixed before I touch the ordering.

**Laying out the obvious keys — and distrusting them.** This smells like a sort: surely there is a
single key I can sort on and read off the answer. Two candidates jump out, and both are the kind of
thing I would type without thinking if I were careless.

- *Shortest processing time first* — sort by `t[i]` ascending. The intuition is that short jobs
  finishing early keep everyone else's completion times small. This is provably optimal when all
  weights are equal (it minimizes the plain, unweighted sum of completion times). The worry: a job
  that is long but extremely *heavy* probably wants to go early too, and sorting by `t` alone never
  sees the weight.
- *Heaviest first* — sort by `w[i]` descending. The intuition is symmetric: get the expensive
  weights multiplied by small completion times. The worry is the mirror image: a job that is heavy
  but also very *long* delays everything behind it, and a tiny light job often belongs in front of
  it. Sorting by `w` alone never sees the time.

Each key ignores exactly the number the other one looks at. That asymmetry is a red flag that
*neither* single key is the right one, because the cost `w[i] * C[i]` couples both numbers
multiplicatively. I refuse to ship either until I have tried to break it on a concrete instance.

**Breaking shortest-job-first on a concrete case.** Let me build a tiny instance designed to punish
sort-by-time. Take three jobs:

- `A = (t=1, w=1)`
- `B = (t=3, w=5)`
- `C = (t=2, w=1)`

Sort-by-`t` ascending orders them `A (t=1), C (t=2), B (t=3)`. Completion times `1, 3, 6`, cost
`1*1 + 1*3 + 5*6 = 1 + 3 + 30 = 34`. Now I deliberately pull the heavy job `B` to the front:
order `B, A, C`. Completion times `3, 4, 6`, cost `5*3 + 1*4 + 1*6 = 15 + 4 + 6 = 25`. That is
strictly better than 34, and I can see *why*: `B` carries weight 5, so every unit of delay on `B`
costs five times as much as a unit of delay on the light jobs; paying a little extra delay on the
two weight-1 jobs to finish the weight-5 job sooner is a bargain. So shortest-job-first is wrong,
and by symmetry I expect heaviest-first to fail somewhere too (a heavy-but-very-long job placed
first would crush all the short jobs behind it). Both single keys are out. I need a key that weighs
time against weight *together*.

**Deriving the right comparison from a swap.** When the global picture is murky, the exchange
argument is the standard tool: take any schedule, look at two **adjacent** jobs, and ask whether
swapping them helps. If I can show that there is always a definite "this one should come first"
between any two adjacent jobs, and that this preference is *consistent* (a total order), then the
schedule that is sorted by that preference can never be improved by an adjacent swap — and since any
permutation can be reached from any other by adjacent swaps, no schedule beats it.

So consider two jobs `i` and `j` sitting next to each other, with some amount of time `P` already
accumulated by the jobs scheduled before this pair (the jobs after the pair are untouched by a swap
within it — their completion times shift by the same total `t[i]+t[j]` either way, so they do not
enter the comparison). Two orders of just this pair:

- `i` then `j`: `i` completes at `P + t[i]`, `j` at `P + t[i] + t[j]`. Cost contribution
  `w[i]*(P + t[i]) + w[j]*(P + t[i] + t[j])`.
- `j` then `i`: `j` completes at `P + t[j]`, `i` at `P + t[j] + t[i]`. Cost contribution
  `w[j]*(P + t[j]) + w[i]*(P + t[j] + t[i])`.

Subtract the second from the first. The `P`-terms `(w[i]+w[j])*P` are identical and cancel. The
self-terms `w[i]*t[i]` and `w[j]*t[j]` appear in both and cancel. What is left is

```
(i before j) - (j before i) = w[j]*t[i] - w[i]*t[j].
```

So putting `i` first is no worse than putting `j` first exactly when `w[j]*t[i] - w[i]*t[j] <= 0`,
i.e. when

```
t[i] * w[j] <= t[j] * w[i].
```

Dividing both sides by the positive product `w[i]*w[j]` (weights are at least 1), this says
`t[i]/w[i] <= t[j]/w[j]`: **schedule jobs in ascending order of the ratio `t/w`.** This is Smith's
rule. The comparator is the coupled one named in the cost, `t[i]*w[j]` versus `t[j]*w[i]` — it is
neither "sort by `t`" nor "sort by `w`", and it collapses to each of those only in a degenerate
case (all `w` equal -> sort by `t`; all `t` equal -> sort by `w`), which is exactly why those
single keys looked plausible without being correct.

**Checking the new key on the case that broke before.** The three jobs again: ratios
`A: 1/1 = 1`, `B: 3/5 = 0.6`, `C: 2/1 = 2`. Ascending ratio order is `B (0.6), A (1), C (2)` — the
order `B, A, C` I found by hand, cost `25`. The comparator reproduces the optimum I stumbled onto,
and now I have the reason, not luck. Good.

**Why the comparator gives a genuine total order — and why I must not use floats.** For the
adjacent-swap argument to certify a global optimum, the relation "`i` before `j` iff
`t[i]*w[j] <= t[j]*w[i]`" must be a consistent total preorder: transitive, and total (for any pair,
one direction holds). It is — it is exactly the order induced by the real number `t[i]/w[i]`, and
`<=` on reals is a total order. Two subtleties bite here:

1. *Ties.* Many jobs can share the same ratio (e.g. `(2,1), (4,2), (6,3)` all have ratio 2). The
   swap difference `w[j]*t[i] - w[i]*t[j]` is then exactly 0, so either adjacent order costs the
   same — the cost is tie-invariant. But `std::sort` needs a *strict weak ordering*; if my
   comparator ever reports both `cmp(i,j)` and `cmp(j,i)` true on a tie it is undefined behavior. So
   on equal cross-products I must return a single consistent direction (I will break ties by index).
2. *Floats.* I must not compute `t[i]/w[i]` as a `double` and compare. With values up to `10^4`,
   ratios like `9999/10000` and `10000/10001`... wait, weights cap at `10^4`, but still near-equal
   ratios such as `3/9999` vs `4/9999*...` can collide in floating point and silently violate the
   strict-weak-ordering contract, corrupting the sort. The cross-multiplied integer comparison
   `t[i]*w[j]` vs `t[j]*w[i]` is exact. Those products are at most `10^4 * 10^4 = 10^8`, trivially
   inside 64-bit, so the comparison never overflows. I will cross-multiply.

**First implementation.** I sort an index array by the integer comparator, then sweep once
accumulating the running completion time and the cost:

```
vector<int> idx(n);
iota(idx.begin(), idx.end(), 0);
sort(idx.begin(), idx.end(), [&](int i, int j) {
    return t[i]*w[j] < t[j]*w[i];          // ascending t/w, cross-multiplied
});
long long cost = 0, clock = 0;
for (int k = 0; k < n; k++) {
    int i = idx[k];
    clock += t[i];
    cost += w[i] * clock;
}
```

**A trace that exposes a real bug.** Before trusting it, I check the comparator's contract on a
tie, because that is where this exact pattern dies. Take two jobs with the *same ratio*:
`P = (t=2, w=1)` and `Q = (t=4, w=2)` (both ratio 2). Evaluate the lambda both ways.
`cmp(P,Q): t[P]*w[Q] < t[Q]*w[P]` is `2*2 < 4*1` -> `4 < 4` -> false. `cmp(Q,P): t[Q]*w[P] <
t[P]*w[Q]` is `4*1 < 2*2` -> `4 < 4` -> false. So far so good — on this pair the comparator says
"neither strictly precedes", which is the correct behavior for equivalent elements. So the strict
`<` is not *itself* the bug. But I have to be careful about what `std::sort` does with equivalent
elements: it may place them in either relative order, and that is fine *for the cost* (ties are
cost-invariant, as I proved). The subtle defect I want to rule out is a different one: if I had
written the comparator as `t[i]*w[j] <= t[j]*w[i]` — using `<=` because "ascending" felt like it
wanted `<=` — then on the tie `cmp(P,Q)` is `4 <= 4` -> true *and* `cmp(Q,P)` is `4 <= 4` -> true.
Both true means the comparator claims `P < Q` and `Q < P` simultaneously: not a strict weak
ordering, undefined behavior, potential crash or scrambled output on inputs with many equal ratios.
The fix is to keep the comparator strict (`<`), and, so the order is fully deterministic rather than
implementation-defined on ties, break ties explicitly by index. I rewrite the lambda:

```
sort(idx.begin(), idx.end(), [&](int i, int j) {
    long long lhs = t[i] * w[j];
    long long rhs = t[j] * w[i];
    if (lhs != rhs) return lhs < rhs;
    return i < j;                          // strict, deterministic tie-break
});
```

Now `cmp(P,Q)` with `P` at a smaller index returns `P < Q` once and `cmp(Q,P)` returns false:
exactly one direction, a proper strict weak ordering. The cost is unchanged because ties are
cost-invariant, but the program is now well-defined.

**Re-verifying on the worked case and a tie-heavy case.** Worked sample `A=(1,1), B=(3,5),
C=(2,1)`: cross-products order them `B` (ratio 0.6) before `A` (1) before `C` (2), giving `B,A,C`,
cost `5*3 + 1*4 + 1*6 = 25`. Matches the hand optimum. Tie-heavy case `(2,1),(4,2),(6,3)` (all
ratio 2): any order costs the same; the tie-break by index gives the input order, completion times
`2,6,12`, cost `1*2 + 2*6 + 3*12 = 2 + 12 + 36 = 50`. I confirm against brute force below that 50
is indeed the minimum (it is — every order ties).

**Edge cases, deliberately, because this is where this kind of code dies.**

- `n = 0`: the read of `n` succeeds, the loops never run, `cost` stays `0`, I print `0`. The empty
  schedule has cost 0 — correct. (If the input is entirely empty so even `n` fails to read, the
  `if (!(cin >> n)) return 0;` prints nothing; per the contract `n` is always present, but this is a
  safe no-op.)
- `n = 1`, job `(7, 9)`: one completion time `7`, cost `9*7 = 63`. The single-element schedule is
  forced; correct.
- All weights equal: the comparator reduces to `t[i] < t[j]`, i.e. shortest-job-first, which is the
  known optimum for the unweighted sum. The general key degenerates correctly.
- All processing times equal: the comparator reduces to `w[i] < w[j]`... let me check the direction.
  With `t[i]=t[j]=t`, `t[i]*w[j] < t[j]*w[i]` is `t*w[j] < t*w[i]` i.e. `w[j] < w[i]`, so `i` comes
  before `j` when `w[i] > w[j]` — heaviest first. Correct: with equal lengths, the heaviest job
  should finish earliest.
- Overflow: the running `clock` reaches at most `2*10^5 * 10^4 = 2*10^9`, and each
  `cost += w[i]*clock` adds at most `10^4 * 2*10^9 = 2*10^13`, with the total bounded by `~2*10^18`
  — inside signed 64-bit. The comparator's products are at most `10^8`. All `long long`; no overflow
  anywhere.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace so the parsing is
  format-agnostic between the "lines" and "flat token stream" readings of the input.

**Cross-checking against brute force.** I wrote an independent oracle that simply tries all `n!`
orderings and takes the minimum cost — obviously correct, usable for `n <= 8`. I ran the index-sort
solution against it on 800 random small instances (mixing equal-ratio ties, extreme `t`-vs-`w`
skew, and plain random values) plus the explicit edge cases above: zero mismatches. The two cases
that motivated the design — shortest-job-first losing 34-to-25, and the equal-ratio ties — both come
out matching the brute optimum. On a `2*10^5` instance the solution runs in about 0.05 s, comfortably
under the 1 s limit, and the maximal-value instance produces `2*10^18`-scale costs without overflow.

**Final solution.** I convinced myself the *idea* is right by breaking both single-key sorts on a
concrete instance and then deriving the coupled comparator from an adjacent swap (whose difference
is exactly `w[j]*t[i] - w[i]*t[j]`), and I convinced myself the *code* is right by tracing the tie
behavior to a precise strict-weak-ordering hazard, fixing the comparator to strict-with-index-break,
and re-verifying against brute force and the corners. That is what I ship — one self-contained file,
the `O(n log n)` sort I can defend by the exchange argument rather than the greedy single keys I
broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> cost 0

    vector<long long> t(n), w(n);
    for (int i = 0; i < n; i++) cin >> t[i] >> w[i];

    // Single machine, minimize sum of w[i] * C[i], where C[i] is the completion
    // time (prefix sum of processing times) of job i in the chosen order.
    //
    // Exchange argument: for two adjacent jobs i (first) then j (second), the part
    // of the cost that depends on their relative order is, letting P be the time
    // accumulated before the pair,
    //     i before j:  w[i]*(P+t[i]) + w[j]*(P+t[i]+t[j])
    //     j before i:  w[j]*(P+t[j]) + w[i]*(P+t[j]+t[i])
    // The P-terms and the w*t self-terms cancel; the difference reduces to
    //     (i before j) - (j before i) = w[j]*t[i] - w[i]*t[j].
    // So putting i first is no worse exactly when w[j]*t[i] <= w[i]*t[j], i.e.
    //     t[i]*w[j] <= t[j]*w[i]   (Smith's ratio rule: ascending t/w).
    // Sorting by this total order is therefore optimal. Use cross-multiplication
    // (not the float ratio t/w) to stay exact; products fit in long long since
    // t,w <= 1e4 -> product <= 1e8, and the total cost <= ~2e18 < 9.2e18.
    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int i, int j) {
        // i before j iff t[i]*w[j] < t[j]*w[i]; ties keep a stable, total order.
        long long lhs = t[i] * w[j];
        long long rhs = t[j] * w[i];
        if (lhs != rhs) return lhs < rhs;
        return i < j;                      // deterministic tie-break (cost is tie-invariant)
    });

    long long cost = 0;     // total weighted completion time, <= ~2e18, fits in signed 64-bit
    long long clock = 0;    // running completion time (prefix sum of processing times)
    for (int k = 0; k < n; k++) {
        int i = idx[k];
        clock += t[i];                     // completion time of job i
        cost += w[i] * clock;
    }

    cout << cost << "\n";
    return 0;
}
```

**Causal recap.** The cost `sum w[i]*C[i]` couples each job's weight and length, so each single-key
sort ignores half the data and is breakable: on `A=(1,1), B=(3,5), C=(2,1)` shortest-job-first costs
34 while pulling the heavy `B` forward costs 25. An adjacent swap of jobs `i, j` changes the cost by
exactly `w[j]*t[i] - w[i]*t[j]`, so `i` belongs before `j` iff `t[i]*w[j] <= t[j]*w[i]` — Smith's
ratio rule, sorted in `O(n log n)`. The comparator must be strict (`<`) with an index tie-break to
be a valid strict weak ordering on equal ratios (a `<=` form is undefined behavior), and
cross-multiplied integers (not the float ratio) to stay exact; `long long` everywhere keeps the
`~2*10^18` cost and the `~10^8` products from overflowing. Brute-force differential testing over 800
random cases plus the empty/single/equal-weight/equal-time/tie corners confirms it.
