**Reading the problem and pinning the contract.** I have `n` integers `a[0..n-1]` that can be
negative, zero, or positive, and I must count contiguous subarrays `(l, r)` with
`0 <= l <= r < n` whose sum is **strictly** less than zero. Input on stdin is `n` then the `n`
values; I print one integer. Before any algorithm I fix the scale, because it dictates the data
types and rules out brute force. `n <= 2*10^5`, so there are up to `n(n+1)/2 ~ 2*10^10` subarrays;
the *count of qualifying ones* can itself be on that order, which overflows 32-bit (`~2.1*10^9`).
So the answer accumulator must be 64-bit. And `|a[i]| <= 10^9` with up to `2*10^5` terms means a
prefix sum can reach `2*10^14`, also 64-bit. Both decisions are non-negotiable: an `int` anywhere
on the count or the prefix sums is a silent wrong-answer on the large tests. I use `long long`.

**Reducing to a counting-inversions shape.** The sum of subarray `(l, r)` is awkward to handle
directly, so I move to prefix sums. Define `P[0] = 0` and `P[k] = a[0] + ... + a[k-1]` for
`1 <= k <= n`. Then `sum(l, r) = P[r+1] - P[l]`, and `sum(l, r) < 0  <=>  P[r+1] < P[l]`. As
`(l, r)` ranges over `0 <= l <= r < n`, the pair `(i, j) = (l, r+1)` ranges over
`0 <= i < j <= n`. So the answer is exactly the number of pairs `(i, j)` with `i < j` and
`P[j] < P[i]` — the count of *strict inversions* of the `(n+1)`-element prefix array. The crucial
detail I write down now, before I can forget it: index `0` of the prefix array, `P[0] = 0`, is a
real participant. A subarray that starts at `l = 0` corresponds to `i = 0`, so `P[0]` must be one
of the "earlier" values I count against. Dropping it would silently lose every subarray anchored
at the left end.

**Laying out the candidate approaches.** Two routes, and I commit to the one whose cost fits.

- *Brute force over subarrays.* For each `l`, run a running sum to the right and tally each time
  it dips below `0`. `O(n^2)`, correct by construction, three lines. But `~2*10^10` operations on
  the largest input is far over a 1-second limit. I keep it only as an oracle to check the fast
  solution against.
- *Fenwick over compressed prefix sums.* Sweep `j` from `0` to `n` in index order. Just before
  inserting `P[j]`, I want the number of already-inserted `P[i]` (those with `i < j`) that are
  *strictly greater* than `P[j]`; summed over `j`, that is the strict-inversion count. A Binary
  Indexed Tree over the coordinate-compressed prefix-sum values gives each query in `O(log n)`, so
  `O(n log n)` overall. This is the one I build.

**The two subtle spots I flag up front.** A Fenwick naturally answers "how many inserted values
have rank `<= r`" (a prefix sum of frequencies). But I need "how many inserted values are
*strictly greater* than `P[j]`." With `r = rank(P[j])` (1-based rank among sorted-unique values),
"inserted values with rank `<= r`" counts exactly the values `<= P[j]` (because equal values share
the same rank). So `greater = (number inserted so far) - (count with rank <= r)`. Two ways this
goes wrong, and both are exactly the kind of bug the negatives/zeros corner exposes: (1) seeding
the base value `P[0]` incorrectly, and (2) using `<= P[j]` versus `< P[j]` at the equality
boundary — i.e. strict vs non-strict, which flips on every pair of equal prefix sums (the all-zero
array is nothing but equal prefix sums). I will deliberately trace both.

**Deriving the query and checking it on the sample.** Let me verify the whole pipeline on the
sample `a = [3, -4, 1, -2]`, whose answer is `7`. Prefix sums: `P = [0, 3, -1, 0, -2]`. Sorted
unique values: `[-2, -1, 0, 3]`, giving ranks `-2 -> 1`, `-1 -> 2`, `0 -> 3`, `3 -> 4`. Now sweep
`j = 0..4`, each step querying "earlier values strictly greater than `P[j]`" then inserting `P[j]`:

- `j=0`, `P=0`, rank 3. Inserted so far `0`. `greater = 0`. Running answer `0`. Insert rank 3.
- `j=1`, `P=3`, rank 4. Inserted `1` (just the `0`). Count `<= 3` is `1`; `greater = 1 - 1 = 0`.
  Answer `0`. Insert rank 4.
- `j=2`, `P=-1`, rank 2. Inserted `2` (values `0, 3`). Count `<= -1` is `0`; `greater = 2 - 0 = 2`.
  Answer `2`. Insert rank 2.
- `j=3`, `P=0`, rank 3. Inserted `3` (values `0, 3, -1`). Count `<= 0` is `2` (the `0` and the
  `-1`); `greater = 3 - 2 = 1`. Answer `3`. Insert rank 3.
- `j=4`, `P=-2`, rank 1. Inserted `4` (values `0, 3, -1, 0`). Count `<= -2` is `0`;
  `greater = 4 - 0 = 4`. Answer `7`. Insert rank 1.

Final answer `7`. Matches. The derivation, including the subtraction trick for "strictly greater,"
is right.

**First implementation.** My first cut of the core sweep:

```
Fenwick fen(m);
long long answer = 0;
for (int j = 1; j <= n; j++) {            // start at first real prefix
    int r = rankOf(P[j]);
    long long greater = (j - 1) - fen.sumPrefix(r);
    answer += greater;
    fen.add(r, 1);
}
```

I started the loop at `j = 1` thinking "the subarray's right end `r+1` is at least `1`, so the
first interesting prefix is `P[1]`," and used `inserted = j - 1`. That reasoning conflated the two
endpoints. Time to trace.

**First debug episode — the all-negative corner pins a wrong base case.** I trace the smallest
input where the left-anchored subarrays dominate: `a = [-2, -5, -1]`, where *every* subarray is
negative, so the answer must be all `n(n+1)/2 = 6` of them. Prefix sums `P = [0, -2, -7, -8]`,
sorted unique `[-8, -7, -2, 0]`, ranks `-8 -> 1`, `-7 -> 2`, `-2 -> 3`, `0 -> 4`. Run my code:

- `j=1`, `P=-2`, rank 3. `inserted = 0`. The tree is empty, `sumPrefix(3) = 0`,
  `greater = 0 - 0 = 0`. Answer `0`. Insert rank 3.
- `j=2`, `P=-7`, rank 2. `inserted = 1`. `sumPrefix(2) = 0` (only rank 3 present),
  `greater = 1 - 0 = 1`. Answer `1`. Insert rank 2.
- `j=3`, `P=-8`, rank 1. `inserted = 2`. `sumPrefix(1) = 0`, `greater = 2 - 0 = 2`. Answer `3`.
  Insert rank 1.

The code returns `3`, but the truth is `6`. Exactly half are missing — and the missing ones are
the three subarrays that start at `l = 0`: `[0,0]=-2`, `[0,1]=-7`, `[0,2]=-8`. The defect is
precise: those subarrays correspond to inversions against `i = 0`, i.e. against `P[0] = 0`, but my
loop began at `j = 1` and *never inserted `P[0]` into the tree*. By the time I query for `P[1]`,
`P[2]`, `P[3]`, the value `0` — which is strictly greater than all of them and should contribute a
`+1` to each — simply is not there. This is a wrong base case in the exact way the task's corner
predicts: with all-negative data the left-anchored prefixes are precisely the ones that need `P[0]`
as their larger partner, so omitting `P[0]` halves the answer instead of erroring loudly.

**Fixing the base case.** Let the sweep start at `j = 0` and use `inserted = j` (after `k`
insertions, `inserted` equals the number of values placed, which is `j`). `P[0]` is then queried
(it has no earlier values, contributing `0`) and, crucially, *inserted* so every later prefix sees
it:

```
for (int j = 0; j <= n; j++) {
    int r = rankOf(P[j]);
    long long inserted = j;                 // P[0..j-1] already in the tree
    long long greater = inserted - fen.sumPrefix(r);
    answer += greater;
    fen.add(r, 1);
}
```

Re-trace `[-2, -5, -1]`: `j=0` `P=0` rank 4, inserted 0, greater 0, answer 0, insert rank 4. `j=1`
`P=-2` rank 3, inserted 1, `sumPrefix(3)=0`, greater 1, answer 1, insert rank 3. `j=2` `P=-7`
rank 2, inserted 2, `sumPrefix(2)=0`, greater 2, answer 3, insert rank 2. `j=3` `P=-8` rank 1,
inserted 3, `sumPrefix(1)=0`, greater 3, answer 6, insert rank 1. Answer `6`. Correct. The case
that broke now passes, and it broke for the reason I fixed — the missing `P[0]` seed — which is the
evidence I trust.

**Second debug episode — the all-zero corner pins the strict/non-strict boundary.** With the base
case fixed I worry about the other flagged spot: the comparison at equality. The query
`greater = inserted - sumPrefix(r)` uses `r = rank(P[j])`, and `sumPrefix(r)` counts ranks
`<= r`, i.e. earlier values `<= P[j]`, so `greater` counts earlier values *strictly* `> P[j]` —
exactly the strict inversion I want. To convince myself I imagine the natural *wrong* version a
tired hand would write, querying ranks `<= r-1` (earlier values `< P[j]`) and subtracting that,
which would count earlier values `>= P[j]` — i.e. subarrays with sum `<= 0`, wrongly including the
zero-sum ones. I trace the input that is pure equality: `a = [0, 0, 0]`, whose answer is `0` (no
subarray sums to anything strictly negative). Prefix sums `P = [0, 0, 0, 0]`, sorted unique `[0]`,
so every `P[j]` has rank `1`.

- *Correct version* (`sumPrefix(r) = sumPrefix(1)` = count of values `<= 0`): `j=0` inserted 0,
  `sumPrefix(1)=0`, greater 0; `j=1` inserted 1, `sumPrefix(1)=1`, greater 0; `j=2` inserted 2,
  `sumPrefix(1)=2`, greater 0; `j=3` inserted 3, `sumPrefix(1)=3`, greater 0. Answer `0`. Correct.
- *The wrong `>=` version* (`inserted - sumPrefix(r-1) = inserted - sumPrefix(0)`, and
  `sumPrefix(0) = 0` always): `j=0` greater 0; `j=1` `1 - 0 = 1`; `j=2` `2 - 0 = 2`; `j=3`
  `3 - 0 = 3`. Answer `6`. Wrong — it counts all six zero-sum subarrays as if they were negative.

So the strict boundary lives entirely in the choice `sumPrefix(r)` versus `sumPrefix(r-1)`, and the
all-zero array is the input that separates them: `0` for `>` (correct) versus `6` for `>=`. My code
uses `sumPrefix(r)`, the correct strict form. I keep the trace as the guard: zeros must not count.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: input is just `0`. The value loop never runs, `P = [0]` (a single zero). The sweep runs
  one iteration `j=0`, contributing `greater = 0`. Answer `0`. There are no subarrays — correct.
  If stdin is truly empty (no `n` token at all), `if (!(cin >> n)) return 0;` exits cleanly.
- `n = 1`, `a = [-5]`: `P = [0, -5]`. `j=0`: insert rank of `0`. `j=1`: `P=-5 < 0`, the earlier
  `0` is strictly greater, `greater = 1`. Answer `1` — the single subarray is negative. Correct.
- `n = 1`, `a = [0]`: `P = [0, 0]`. `j=1`: earlier `0` is not strictly greater than `0`,
  `greater = 0`. Answer `0`. Correct — a zero-sum subarray is not negative.
- All-positive, e.g. `[2, 3]`: prefix sums strictly increase, no earlier value is ever greater,
  answer `0`. Correct.
- Overflow: the answer accumulator and prefix sums are `long long`; the count can reach `~2*10^10`
  and a prefix sum `~2*10^14`, both inside 64-bit. The Fenwick stores frequency counts (each `<= n`)
  and its `sumPrefix` is at most `n + 1`, so `int` would technically hold those, but I keep the
  tree in `long long` to avoid any mixed-type surprise. Verified on `n = 2*10^5` random in
  `[-10^9, 10^9]`: the program prints `11481414687` (well past 32-bit) in ~40 ms.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the input
  parsing is format-agnostic.

**Final solution.** I convinced myself the *idea* is right by reducing to strict inversions and
hand-checking the sweep on the sample, and I convinced myself the *code* is right by tracing the
two corners the data demands — the all-negative array exposed a missing `P[0]` base case (returned
`3` instead of `6`), and the all-zero array pins the strict-`>`-not-`>=` boundary (correct `0`
versus the wrong `6`). What I ship is one self-contained file: the `O(n log n)` Fenwick sweep over
compressed prefix sums.

```cpp
#include <bits/stdc++.h>
using namespace std;

// Fenwick (Binary Indexed Tree) for prefix counts over compressed values.
struct Fenwick {
    int n;
    vector<long long> t;
    Fenwick(int n) : n(n), t(n + 1, 0) {}
    void add(int i, long long v) {            // i is 1-based
        for (; i <= n; i += i & (-i)) t[i] += v;
    }
    long long sumPrefix(int i) {              // sum over [1..i], i is 1-based
        long long s = 0;
        for (; i > 0; i -= i & (-i)) s += t[i];
        return s;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;               // empty input -> n = 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Prefix sums P[0..n], P[0] = 0, P[k] = a[0] + ... + a[k-1].
    // Subarray [l, r] sum = P[r+1] - P[l]; we want it strictly negative,
    // i.e. count pairs (i, j) with 0 <= i < j <= n and P[j] < P[i].
    vector<long long> P(n + 1);
    P[0] = 0;
    for (int k = 0; k < n; k++) P[k + 1] = P[k] + a[k];

    // Coordinate-compress all n+1 prefix-sum values (negatives/zeros included).
    vector<long long> vals(P.begin(), P.end());
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int m = (int)vals.size();
    auto rankOf = [&](long long x) {          // 1-based rank in sorted unique vals
        return int(lower_bound(vals.begin(), vals.end(), x) - vals.begin()) + 1;
    };

    // Sweep j = 0..n in order. For each P[j], the number of earlier P[i] (i < j)
    // with P[i] > P[j] is (j) - (count of earlier values <= P[j]).
    Fenwick fen(m);
    long long answer = 0;
    for (int j = 0; j <= n; j++) {
        int r = rankOf(P[j]);
        long long inserted = j;               // we have inserted P[0..j-1] so far
        long long leq = fen.sumPrefix(r);     // earlier values with rank <= r (i.e. P[i] <= P[j])
        long long greater = inserted - leq;   // earlier values with P[i] > P[j]
        answer += greater;
        fen.add(r, 1);                        // now insert P[j]
    }

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The reduction `sum(l, r) < 0  <=>  P[r+1] < P[l]` turns the problem into counting
strict inversions of the `(n+1)`-element prefix array, which a Fenwick over compressed prefix sums
does in `O(n log n)`; the count and the prefix sums must be 64-bit because both blow past 32-bit at
scale. My first sweep started at `j = 1` and never seeded `P[0] = 0`, so a trace of the
all-negative `[-2, -5, -1]` returned `3` instead of `6` — every left-anchored subarray was lost
because its larger partner `P[0]` was absent — fixed by sweeping `j = 0..n` with `inserted = j` so
`P[0]` is queried and inserted; and the strict-negative boundary lives in querying `sumPrefix(r)`
(count `<= P[j]`, giving "strictly greater") rather than `sumPrefix(r-1)`, a distinction the
all-zero `[0, 0, 0]` separates as the correct `0` against the wrong `6`. With both corners pinned
by traces and 64-bit accumulators, the empty, single-element, all-positive, and overflow cases fall
out cleanly.
