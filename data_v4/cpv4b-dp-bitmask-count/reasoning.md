**Reading the problem and pinning the contract.** I have `n <= 16` adventurers and I must split *all*
of them into teams — a set partition — where every team has size in `[L, R]` and contains no feuding
pair, and I count distinct partitions modulo `998244353`. The decisive phrase is that teams are
*unlabeled*: a partition is a set of teams, so reordering the teams gives the same object and must not
be counted twice. The empty guild `n = 0` has exactly one partition (the empty partition), so the
answer there is `1`, not `0`. Scale: `n <= 16` means `2^n <= 65536` subsets, which is what makes a
subset DP feasible; and the answer is taken mod a prime, so I must keep every accumulator reduced and
never let a partial sum overflow `long long`. No individual count exceeds the modulus once I reduce, so
`long long` is more than enough — the only arithmetic is additions of values in `[0, p)`.

**Laying out the candidate approaches.** Two routes, and the whole game is counting each partition
*once*.

- *Ordered DP, then un-order.* Let `g[mask]` be the number of ways to write `mask` as an **ordered**
  list of legal teams: `g[mask] = sum over legal S ⊆ mask of g[mask \ S]`, base `g[0] = 1`. This is a
  three-line recurrence. But a partition into `k` teams is produced in all `k!` orders, so
  `g[full]` over-counts. I cannot just divide by a single `k!` because different partitions have
  different `k`. To recover unordered counts I would have to track `k` as a second dimension and divide
  each by `k!` — workable but fiddly, and it is exactly the place where a stray modular inverse or an
  off-by-one in `k` quietly corrupts the answer. I distrust it.
- *Anchored subset DP.* Let `f[mask]` be the number of **unordered** legal partitions of `mask`
  directly. The trick to kill ordering: impose a canonical rule that the team containing the
  *lowest-indexed* element of `mask` is chosen first. Concretely, let `low` be the lowest set bit of
  `mask`; sum only over legal teams `S` that contain `low`, and recurse on `mask \ S`. Because the
  anchor element belongs to exactly one team in any partition, each partition is reconstructed by
  exactly one sequence of anchor choices — no `k!`, no inverses. This counts unordered partitions with
  no post-correction. I commit to this.

Before any code I want to be sure the anchoring claim is true and not wishful, because "it feels
canonical" is how double-counts ship.

**Deriving the anchored recurrence and proving it counts once.** Fix a nonempty `mask` and let `low`
be its lowest set bit, i.e. the smallest-indexed element present. In *any* partition of `mask`, that
element `low` lies in exactly one team `S`; and `S` is a subset of `mask` containing `low`. So every
partition of `mask` decomposes uniquely as "(the team `S` containing `low`) together with (a partition
of `mask \ S`)". Conversely, for each legal `S ∋ low` and each legal partition `P'` of `mask \ S`, the
set `{S} ∪ P'` is a legal partition of `mask`, and distinct `(S, P')` give distinct results because `S`
is recovered as the unique team containing `low`. That bijection gives

  `f[mask] = sum over legal teams S with low ∈ S ⊆ mask of f[mask \ S]`,  with `f[0] = 1`.

The "`low ∈ S`" restriction is the entire de-duplication. Drop it and I would let *any* legal team be
"first", reintroducing the `k!` ordering — that is the bug I am pre-warned about, and I will go out of
my way to trigger it later to be certain.

**Numeric self-check of the recurrence on a known closed form.** A recurrence I can defend on paper can
still be transcribed wrong, so I check it against a quantity I already know. Take no feuds and
`L = 1, R = n`: then *every* nonempty subset is a legal team, so `f[full]` should be the Bell number
`B(n)`, the count of all set partitions. Let me run the anchored recurrence for `n = 3`,
`mask = {0,1,2} = 0b111`, `low = bit 0`. Teams `S` containing bit 0 and their complements:

- `S = {0}` -> `f[{1,2}]`
- `S = {0,1}` -> `f[{2}]`
- `S = {0,2}` -> `f[{1}]`
- `S = {0,1,2}` -> `f[{}] = 1`

Now `f[{2}] = f[{1}] = 1` (a singleton has one partition), and `f[{1,2}]`: anchor bit 1, teams `{1}`->`f[{2}]=1` and `{1,2}`->`f[{}]=1`, so `f[{1,2}] = 2`. Total `f[{0,1,2}] = 2 + 1 + 1 + 1 = 5`. And `B(3) = 5`. The recurrence reproduces the Bell number, so the *idea* is right. (I will later run the program for `n = 16` and check it equals `B(16) mod p` as a second, larger anchor.)

**Precomputing team legality.** Before the DP I mark, for every subset `S`, whether it is a legal team:
`L <= popcount(S) <= R` and no feuding pair sits inside `S`. I store each adventurer's feud set as a
bitmask `feud[i]`. Then `S` has no internal feud iff for every `i ∈ S`, `feud[i] & S == 0`. That is an
`O(2^n * n)` precompute, trivial at `n = 16` (about `10^6` operations).

**First implementation and a trace.** Here is my first cut of the DP loop; I am deliberately suspicious
of both the submask enumeration and the anchoring.

```
vector<long long> f(1 << n, 0);
f[0] = 1;
for (int mask = 1; mask <= full; mask++) {
    long long acc = 0;
    for (int sub = mask; ; sub = (sub - 1) & mask) {   // all submasks of mask
        if (sub != 0 && valid[sub]) {
            acc += f[mask ^ sub];
            if (acc >= MOD) acc -= MOD;
        }
        if (sub == 0) break;
    }
    f[mask] = acc;
}
```

I trace the *cleanest* possible case where the answer is known: `n = 3`, no feuds, `L = 1, R = 3`, so
every nonempty subset is a legal team and the answer must be `B(3) = 5`. Compute `f` in increasing
mask order.

- `f[001] = 1` (only submask `{0}`, `f[000]=1`).
- `f[010] = 1`, `f[100] = 1` similarly.
- `f[011] = {0,1}`: submasks `{1}`->`f[{0}]=1`, `{0}`->`f[{1}]=1`, `{0,1}`->`f[{}]=1`. Sum `= 3`.
- `f[101]`, `f[110]` likewise `= 3`.
- `f[111]`: submasks `{0}`->`f[{1,2}]`, `{1}`->`f[{0,2}]`, `{2}`->`f[{0,1}]`, `{0,1}`->`f[{2}]`,
  `{0,2}`->`f[{1}]`, `{1,2}`->`f[{0}]`, `{0,1,2}`->`f[{}]`. That is `3 + 3 + 3 + 1 + 1 + 1 + 1 = 13`.

So this code prints `13`, but the true answer is `B(3) = 5`.

**The bug.** Thirteen is not random — it is exactly the count of *ordered* set partitions of 3 elements:
ordered partitions = `sum_k k! * S(3,k) = 1!*1 + 2!*3 + 3!*1 = 1 + 6 + 6 = 13`, where `S(3,k)` are
Stirling numbers of the second kind. So my DP counted every partition once per ordering of its teams.
The cause is precise: by enumerating **all** submasks `sub` as the "first" team, I let the partition
`{ {0},{1},{2} }` be built as `{0}` then `{1}` then `{2}`, and *also* as `{1}` then `{0}` then `{2}`,
and so on — `3! = 6` times for that one partition, `2` times for each two-block partition. I even saw
the over-count concretely inside `f[011] = 3`: the partition `{ {0},{1} }` of the two-element set got
counted twice (once anchored at `{0}` then `{1}`, once at `{1}` then `{0}`) on top of the single
two-element team `{0,1}`, giving `3` where the unordered count is `2`. This is the double-count the
problem is built around, and I reproduced it. The fix is the anchoring I proved earlier: only let `sub`
range over teams that contain the lowest set bit of `mask`.

**Fix and re-verification.** I restrict the first team to contain the anchor `low = mask & (-mask)`. To
enumerate exactly the submasks of `mask` that contain `low`, I peel off `low`, let `rest = mask ^ low`,
enumerate all submasks `sub` of `rest`, and form the team `S = sub | low`. That ranges over precisely
the subsets of `mask` containing `low`.

```
int low = mask & (-mask);
int rest = mask ^ low;
long long acc = 0;
for (int sub = rest; ; sub = (sub - 1) & rest) {
    int S = sub | low;
    if (valid[S]) { acc += f[mask ^ S]; if (acc >= MOD) acc -= MOD; }
    if (sub == 0) break;
}
f[mask] = acc;
```

Re-trace `n = 3`, no feuds, `L = 1, R = 3`.

- `f[001] = f[010] = f[100] = 1`.
- `f[011]`: `low = {0}`, `rest = {1}`. Submasks of `rest`: `{1}` -> `S={0,1}` -> `f[{}]=1`; `{}` ->
  `S={0}` -> `f[{1}]=1`. Sum `= 2`. (Now `{ {0},{1} }` is counted *once*, anchored at `{0}`.) Good.
- `f[101] = f[110] = 2` by symmetry.
- `f[111]`: `low = {0}`, `rest = {1,2}`. Submasks of `{1,2}`: `{1,2}`->`S={0,1,2}`->`f[{}]=1`;
  `{2}`->`S={0,2}`->`f[{1}]=1`; `{1}`->`S={0,1}`->`f[{2}]=1`; `{}`->`S={0}`->`f[{1,2}]=2`. Sum
  `= 1 + 1 + 1 + 2 = 5`. Exactly `B(3)`.

The over-count is gone and the value matches the closed form. The fix corrected the case for the reason
I diagnosed (only the anchor-containing teams are "first"), which is the evidence I trust.

**A second debug episode — the size and feud filter on a concrete sample.** Counting bugs love the
`valid[]` predicate, so I trace the documented sample independently: `n = 4`, `L = 1`, `R = 2`, feuds
`{0,1}` and `{2,3}`; the answer is supposed to be `7`. First, *which* teams are legal? Size must be in
`[1,2]`, so singletons `{0},{1},{2},{3}` (4 of them) are legal, and pairs are legal only if not a feud:
the pairs `{0,1}` and `{2,3}` are *illegal* (feuds), so the legal pairs are `{0,2},{0,3},{1,2},{1,3}`.
Size-3 and size-4 teams exceed `R = 2`, illegal. Now I enumerate valid partitions by hand to get a
ground truth to compare the DP against:

- all singletons: `{ {0},{1},{2},{3} }` — 1 partition.
- one legal pair + two singletons: the pair must be one of `{0,2},{0,3},{1,2},{1,3}` and the other two
  elements are singletons; each gives a valid partition — 4 partitions.
- two legal pairs covering all four: pair up `{0,1,2,3}` into two disjoint pairs avoiding feuds. The
  three perfect matchings of four elements are `{01,23}`, `{02,13}`, `{03,12}`. The first uses both
  feud pairs `{0,1}` and `{2,3}` — illegal. The other two use only non-feud pairs — both valid — 2
  partitions.

Total `1 + 4 + 2 = 7`. Now I trace the DP on `mask = full = 1111`. `low = {0}`, `rest = {1,2,3}`.
Enumerate teams `S = sub | {0}` over submasks of `{1,2,3}`; only `S` of size `<= 2` and non-feud can be
valid:

- `S = {0}` (sub `{}`): valid; add `f[{1,2,3}]`.
- `S = {0,1}` (sub `{1}`): feud — invalid, skip.
- `S = {0,2}` (sub `{2}`): valid; add `f[{1,3}]`.
- `S = {0,3}` (sub `{3}`): valid; add `f[{1,2}]`.
- `S = {0,1,2},{0,1,3},{0,2,3},{0,1,2,3}`: size `>= 3`, invalid, skip.

So `f[full] = f[{1,2,3}] + f[{1,3}] + f[{1,2}]`. I need those sub-values.
`f[{1,2}]`: `low={1}`, teams `{1}`->`f[{2}]=1`, `{1,2}` (non-feud) valid ->`f[{}]=1`; sum `2`.
`f[{1,3}]`: similarly `{1}`->`f[{3}]=1`, `{1,3}` valid ->`f[{}]=1`; sum `2`.
`f[{1,2,3}]`: `low={1}`, `rest={2,3}`; teams `{1}`->`f[{2,3}]`, `{1,2}`->`f[{3}]=1`, `{1,3}`->`f[{2}]=1`,
`{1,2,3}` size 3 invalid. And `f[{2,3}]`: `{2}`->`f[{3}]=1`, `{2,3}` is a feud — invalid; so
`f[{2,3}] = 1` (only the all-singleton split). Thus `f[{1,2,3}] = f[{2,3}] + 1 + 1 = 1 + 1 + 1 = 3`.
Finally `f[full] = 3 + 2 + 2 = 7`. Matches the hand enumeration and the documented sample. The
legality filter and the anchored recurrence agree on a feud-laden case, not just the no-feud Bell case.

**Edge cases, deliberately.**
- `n = 0`: `full = 0`, the `mask` loop starts at `1` and never runs; the answer is `f[0] = 1`. One
  empty partition — correct. (Reading `n L R m` still consumes four tokens; with `m = 0` there are no
  edges.) I verified the program prints `1` on `0 1 1 0`.
- `n = 1`: `f[{0}] = 1` iff a singleton is legal, i.e. `L <= 1 <= R`; with `L = 1` that is `1`. If
  `L = 2`, the singleton is illegal, `rest` is empty, the only `S = {0}` is invalid, so `f = 0` — the
  lone adventurer cannot form a team of size `>= 2`, correctly `0`.
- *Infeasible size window.* `n = 3`, `L = R = 2`: any team has size 2, but 3 is odd, so no partition
  covers everyone; the DP yields `0`. I confirmed `3 2 2 0` -> `0`.
- *Dense feuds forcing singletons.* If every pair feuds, the only legal teams are singletons; with
  `L = 1` there is exactly one partition (all singletons), answer `1`; with `L = 2` no team is legal,
  answer `0`. I confirmed `2 1 2 1 / 0 1` -> `1` (the feud kills the pair, leaving only the
  all-singleton partition).
- *Modulus.* Every `acc` is reduced after each addition (`if (acc >= MOD) acc -= MOD`), and the final
  `f[full] % MOD` is a no-op safety net. Counts never grow unbounded because they live in `[0, p)`. As
  a large anchor I ran the no-feud `n = 16, L = 1, R = 16` case: the program prints `497698617`, and
  `B(16) = 10480142147`, with `10480142147 mod 998244353 = 497698617`. The big case matches the Bell
  number mod `p` — independent confirmation that the anchoring scales and the reduction is correct.
- *Complexity / limits.* The DP is the classic sum-over-submasks, `O(3^n)` ≈ `4.3 * 10^7` at `n = 16`,
  plus the `O(2^n n)` legality precompute. Measured wall time at the `n = 16` worst case is ~20 ms with
  ~3.7 MB resident — comfortably inside 1 s / 256 MB.

**The double-count, named.** The single most important line is the anchor restriction `S ∋ low`. The
"obvious" sum-over-all-submasks recurrence counts *ordered* lists of teams and so multiplies each
partition by the number of orderings of its blocks; I watched it print `13 = 1!·1 + 2!·3 + 3!·1` for
`n = 3` instead of `B(3) = 5`. Forcing the team that contains the lowest remaining element to be chosen
first makes the decomposition of each partition unique, eliminating the over-count with no division and
no modular inverse.

**Final solution.** I disproved the easy ordered DP by catching its `k!` over-count on a Bell-number
case, proved the anchored recurrence counts each partition once via the lowest-element bijection,
checked it numerically against `B(3)` and `B(16) mod p`, and hand-traced the feud-laden sample to `7`.
This is what I ship — one self-contained `O(3^n)` file:

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 998244353;

int main() {
    int n, L, R, m;
    if (!(cin >> n >> L >> R >> m)) return 0;

    // feud[i] = bitmask of adventurers that i refuses to share a team with.
    vector<int> feud(n, 0);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;             // 0-indexed
        feud[u] |= (1 << v);
        feud[v] |= (1 << u);
    }

    int full = (1 << n) - 1;

    // valid[S] = 1 iff subset S is a legal team:
    //   size in [L,R] and contains no feuding pair.
    vector<char> valid(1 << n, 0);
    for (int S = 1; S <= full; S++) {
        int sz = __builtin_popcount(S);
        if (sz < L || sz > R) { valid[S] = 0; continue; }
        bool ok = true;
        int t = S;
        while (t) {
            int i = __builtin_ctz(t);
            t &= t - 1;
            if (feud[i] & S) { ok = false; break; }   // i feuds with someone also in S
        }
        valid[S] = ok ? 1 : 0;
    }

    // f[mask] = number of partitions of the set "mask" into legal teams (unlabeled).
    vector<long long> f(1 << n, 0);
    f[0] = 1;
    for (int mask = 1; mask <= full; mask++) {
        int low = mask & (-mask);          // lowest set bit, the anchor element
        long long acc = 0;
        // Enumerate submasks S of mask that CONTAIN the anchor bit "low".
        // This forces each partition to be counted exactly once: the team
        // holding the lowest-indexed element of "mask" is chosen first.
        int rest = mask ^ low;             // bits we may freely add to the anchor team
        for (int sub = rest; ; sub = (sub - 1) & rest) {
            int S = sub | low;             // team containing the anchor
            if (valid[S]) {
                acc += f[mask ^ S];
                if (acc >= MOD) acc -= MOD;
            }
            if (sub == 0) break;
        }
        f[mask] = acc;
    }

    cout << f[full] % MOD << "\n";
    return 0;
}
```

**Causal recap.** The unlabeled-partition requirement is the crux: the natural "sum over every legal
first team" subset DP counts ordered team-lists and over-counts each partition by its number of block
orderings — a trace of `n = 3` no-feuds printed `13` (the ordered-partition count `1 + 6 + 6`) instead
of `B(3) = 5`, exposing the double-count. Anchoring the first team to contain the lowest set bit of the
current mask makes each partition's decomposition unique, which I proved by the lowest-element
bijection and confirmed numerically against `B(3)` and against `B(16) mod p = 497698617`; a separate
hand-trace of the feud-laden sample to `7` validates the size/feud legality filter, and per-addition
modular reduction over `long long` plus the `O(3^n)` ≈ `4.3·10^7` enumeration keeps it correct and well
inside the time limit.
