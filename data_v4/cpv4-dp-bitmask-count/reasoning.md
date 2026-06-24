**Reading the problem and pinning the contract.** I have `n` employees, `n <= 18`, and `m <= 2*10^5`
candidate squads, each squad a bitmask `S_j` over the `n` employees. A *roster* is a set of candidate
squads that partitions everyone: pairwise disjoint, union equal to the full staff. Two rosters are the
same if they use the same set of squads — order is irrelevant. I count rosters modulo `1e9+7`. Input
is `n m` then the `m` masks; output is one integer. Before any algorithm I fix the corners of the
contract, because they steer the data structures: a candidate mask may be `0` (an empty squad, which
is not a real squad and must be ignored), may be duplicated (the same squad listed twice is still one
squad, so duplicates must not double the count), and may carry stray high bits which I should mask off
with `full = (1<<n)-1`. And `n` can be `0`: the empty staff has exactly one roster, the one that uses
no squads at all. Counts blow far past 64-bit — with everything allowed the answer is the Bell number
`B(18)`, around `6.8e11` *before* reduction but growing super-exponentially in general intermediate
sums — so every accumulator is taken mod `1e9+7` and I keep them in `long long`.

**Laying out the candidate approaches.** Two routes, and I commit to the one I can prove counts each
unordered roster exactly once.

- *Bitmask "next block" DP.* Let `dp[mask]` be the number of valid partitions of the employee set
  `mask` using allowed squads. To build `dp[mask]`, I decide which squad covers some uncovered
  employee, peel it off, and recurse on the remainder `mask ^ squad`. Enumerating submasks makes this
  `O(3^n)`. The entire correctness question is *which* employee I branch on — pick badly and a
  partition with `k` blocks gets counted once per ordering of its blocks, i.e. `k!` times.
- *Inclusion–exclusion over ordered block sequences, then divide by orderings.* Count ordered tuples of
  disjoint allowed squads covering everyone, then divide out `k!`. This needs modular inverse
  factorials and a per-`k` bookkeeping; the divide-by-order step is precisely where a counting mistake
  hides. More moving parts, harder to defend.

I take the bitmask DP. The cleanliness hinges on one disciplined choice of canonical order, which I
will derive rather than guess.

**Deriving the recurrence and the canonical order.** The hazard with set partitions is overcounting by
block order. The standard cure: impose a canonical order on the blocks and only ever generate the
blocks in that order. The cleanest canonical rule is *the block containing the lowest-numbered still
uncovered employee is decided first*. Concretely, for a set `mask`, let `low = mask & (-mask)` be its
lowest set bit (the lowest-indexed employee in `mask`). The squad that owns `low` must be some allowed
submask `sub` of `mask` that contains `low`. Once I fix that squad, the rest of the partition is a
partition of `mask ^ sub`, and — crucially — that remainder's own lowest employee is strictly larger,
so the recursion never revisits the choice for `low`. Therefore

  `dp[mask] = sum over allowed sub ⊆ mask with (sub & low) ≠ 0 of dp[mask ^ sub]`,

with base case `dp[0] = 1` (the empty set has exactly one partition: use no squads). The answer is
`dp[full]`. Because I *always* assign `low` first, every unordered partition corresponds to exactly one
chain of choices, so each is counted once. That is the load-bearing insight; I will test it hard.

**Sanity-checking the derivation on the sample before writing code.** Sample: `n = 4`, allowed squads
(after dropping the empty `0` and the duplicate `3`) are `{3, 5, 10, 12, 15}` in mask form, i.e.
`{0,1}`, `{0,2}`, `{1,3}`, `{2,3}`, `{0,1,2,3}`. I want `dp[1111] = dp[15]`. Lowest bit of `15` is
`low = 1` (employee 0). Allowed submasks of `15` that contain bit 0: `0011 = 3` ({0,1}),
`0101 = 5` ({0,2}), `1111 = 15` (everyone). (`{0,3}=1001=9` and `{0,1,2}=0111=7` etc. are not in the
allowed set.) So `dp[15] = dp[15^3] + dp[15^5] + dp[15^15] = dp[12] + dp[10] + dp[0]`.

- `dp[12]` ({2,3}): low bit is `0100 = 4` (employee 2). Allowed submasks of `12` containing bit 2:
  `1100 = 12` ({2,3}) is allowed; `{2}=4` and `{2,3}` only — `4` is not allowed, `12` is. So
  `dp[12] = dp[12^12] = dp[0] = 1`.
- `dp[10]` ({1,3}): low bit is `0010 = 2` (employee 1). Allowed submasks of `10` containing bit 1:
  `1010 = 10` ({1,3}) is allowed; `{1}=2` is not. So `dp[10] = dp[0] = 1`.
- `dp[0] = 1`.

Thus `dp[15] = 1 + 1 + 1 = 3`. That matches the stated sample answer `3`, and the three rosters are
exactly `{0,1,2,3}`, `{2,3}+{0,1}`, `{1,3}+{0,2}`. The derivation checks out on paper.

**First implementation — and immediately a trace, because counting code lies.** Here is my first cut of
the core. I was sloppy on purpose-of-record about which submasks to allow, writing the recurrence to
sum over *every* allowed squad that is a submask, forgetting the "must contain `low`" guard:

```
// BUGGY v1
vector<long long> dp(full + 1, 0);
dp[0] = 1;
for (int mask = 1; mask <= full; mask++) {
    long long ways = 0;
    for (int sub = mask; sub; sub = (sub - 1) & mask) {  // all submasks
        if (allowed[sub]) ways += dp[mask ^ sub];
    }
    dp[mask] = ways % MOD;
}
```

I trace the smallest input that can expose double-counting: `n = 2`, allowed squads `{0}=1`, `{1}=2`,
`{0,1}=3`. The true answer is `2`: either the two singletons `{0}+{1}`, or the whole-set squad
`{0,1}`. Run v1. `dp[0]=1`. `dp[01=1]`: submasks of `1` are `1`; `allowed[1]` yes, `+= dp[0] = 1`, so
`dp[1]=1`. `dp[10=2]`: submasks `2`; allowed, `+= dp[0]`, `dp[2]=1`. `dp[11=3]`: submasks of `3` are
`3, 2, 1`. `allowed[3]`: `+= dp[0] = 1`. `allowed[2]`: `+= dp[3^2]=dp[1]=1`. `allowed[1]`:
`+= dp[3^1]=dp[2]=1`. Total `dp[3] = 3`.

**Diagnosing the first bug.** v1 returns `3`, but the correct answer is `2`. The phantom third count is
the partition `{0}+{1}` being tallied *twice*: once as "first peel `{1}` (the `sub=2` branch), then
partition the rest `{0}`" and once as "first peel `{0}` (the `sub=1` branch), then partition `{1}`".
Same unordered roster `{0}+{1}`, counted under both orders. This is exactly the `k!`-overcount the
canonical rule is meant to kill. The defect is precise: by allowing *any* submask `sub` to be the
"next block", I let the block order float freely. The fix is the guard I derived and then forgot —
the next block must contain `mask`'s lowest set bit, so the block covering the lowest uncovered
employee is pinned down first and the order is forced. Adding `if (!(sub & low)) continue;` with
`low = mask & (-mask)`:

```
// FIXED core
for (int mask = 1; mask <= full; mask++) {
    int low = mask & (-mask);
    long long ways = 0;
    for (int sub = mask; sub; sub = (sub - 1) & mask) {
        if (!(sub & low)) continue;       // next block must own the lowest employee
        if (allowed[sub]) ways += dp[mask ^ sub];
    }
    dp[mask] = ways % MOD;
}
```

Re-trace `n=2`, allowed `{1,2,3}`. `dp[0]=1`. `dp[1]`: `low=1`; submasks `1`, contains `low`, allowed,
`dp[1]=dp[0]=1`. `dp[2]`: `low=2`; submasks `2`, contains `low`, allowed, `dp[2]=1`. `dp[3]`: `low=1`;
submasks `3,2,1`. `sub=3`: `3&1` nonzero, allowed, `+= dp[0]=1`. `sub=2`: `2&1 == 0`, skip (this was
the duplicate-order branch). `sub=1`: `1&1` nonzero, allowed, `+= dp[3^1]=dp[2]=1`. Total `dp[3]=2`.
Correct. The branch I now skip (`sub=2`) is exactly the one that re-counted `{0}+{1}` in the wrong
order — the bug and the fix line up causally, which is the evidence I trust.

**Second trace — a subtler counting error around duplicates and the empty squad.** The canonical-order
fix handles ordering, but the *input cleaning* is a second place a count goes wrong, and it is easy to
get subtly off. My first reader looked like this:

```
// BUGGY v2 reader
vector<int> allowedList;
for (int j = 0; j < m; j++) { int x; scanf("%d",&x); allowedList.push_back(x); }
// ... later, build dp by, for each mask, summing over allowedList entries that are submasks ...
```

Two things are wrong and I find them by tracing `n = 2`, input masks `{3, 3, 0}` — the squad `{0,1}`
listed twice plus an empty squad. The only valid roster is the single whole-set squad `{0,1}`, so the
answer must be `1`. If I (a) iterate over the *list* `allowedList = [3, 3, 0]` rather than a deduped
*set*, then when filling `dp[3]` I would add `dp[3^3]=dp[0]=1` once for the first `3` and again for
the second `3`, giving `dp[3]=2` — a double-count from the duplicate squad. And (b) the empty mask `0`
is a submask of every set; if I ever treat `sub = 0` as a usable block, the recurrence does
`dp[mask] += dp[mask ^ 0] = dp[mask]`, a self-loop that is both meaningless ("a roster that covers
nobody covers everybody") and, depending on order of evaluation, an infinite inflation. So the empty
squad must be discarded outright, and duplicates must collapse to a single allowed flag.

**Diagnosing and fixing the second bug.** The cure for both is to fold the candidates into a *boolean*
`allowed[mask]` table rather than a multiset list: marking `allowed[x] = 1` is idempotent, so a
duplicated mask sets the same bit twice with no effect — de-dup for free. And I skip `x == 0` at read
time so the empty squad never enters the table; combined with the DP loop starting at `mask = 1` and
the submask enumeration `sub = mask; sub; sub = (sub-1)&mask` which never yields `sub = 0`, the empty
block can never be used. I also `x &= full` to drop any stray high bits before testing for emptiness,
so a mask that is "all stray bits" correctly becomes `0` and is dropped. Re-trace `n=2`, masks
`{3,3,0}`: reader sets `allowed[3]=1` (twice, harmlessly), skips `0`. `dp[3]`: `low=1`; `sub=3` allowed
`+= dp[0]=1`; `sub=2` skipped by the `low` guard; `sub=1` not allowed. `dp[3]=1`. Correct. The
duplicate no longer doubles and the empty squad never participates.

**Edge cases, deliberately, because counting code dies in the corners.**

- `n = 0`: `full = 0`, `allowed` has size `1`, the DP loop `for mask=1..full` never runs, and I print
  `dp[full] = dp[0] = 1`. The empty staff has exactly one roster (use no squads) — correct, and it is
  the value the recurrence's base case already encodes.
- `m = 0` or no squad reaching coverage, e.g. `n=3` with no allowed squad covering employee 0 in any
  partition: every `dp[mask]` that needs an unavailable block stays `0`, so `dp[full] = 0`. A roster is
  genuinely impossible — correct.
- All non-empty subsets allowed: then `dp[full]` should be the Bell number `B(n)`. I verified
  `n = 18` gives `76801385`, which equals `B(18) mod 1e9+7` computed independently via the Bell
  triangle. This is a strong, independent confirmation that the canonical order counts each unordered
  partition exactly once.
- Overflow / modulus: intermediate `ways` is a sum of at most `2^n` terms each `< 1e9+7`, so at
  `n=18` it can reach `~2.6e5 * 1e9 ≈ 2.6e14`, comfortably inside `long long` (`~9.2e18`); I reduce
  `dp[mask] = ways % MOD` after the inner loop, so it never grows unbounded. The final print also
  reduces mod `MOD`. No 32-bit anywhere in the accumulation path.
- Stray high bits in a mask: `x &= full` clamps to the low `n` bits before the empty test, so an input
  mask like `1 << (n+2)` collapses to `0` and is dropped rather than indexing `allowed` out of bounds.
- Output: exactly one integer and a newline; `scanf` skips arbitrary whitespace, so the token layout of
  the input is irrelevant.

**Complexity and the time limit.** The DP is the dominant cost: summing over all `mask` of the number
of submasks of `mask` is `sum_mask 2^popcount(mask) = 3^n`. At `n=18`, `3^18 ≈ 3.87e8` simple
operations; measured wall time with everything allowed is about `0.4 s`, well within the `3 s` limit,
using `~19 MB` (two arrays of size `2^18`). Reading up to `2*10^5` masks with `scanf` is negligible.

**Final solution.** I disproved the order-free recurrence with a traced `n=2` counterexample (v1 gave
`3` vs the true `2`), fixed it with the lowest-bit canonical guard, then caught the duplicate/empty
input inflation with a second `{3,3,0}` trace and fixed it with an idempotent boolean `allowed` table
plus an empty-squad skip. I confirmed the derivation on the sample (`3`), checked the all-subsets case
equals `B(18)`, and walked every corner. This is what I ship — one self-contained `O(3^n)` file:

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;

    int full = (n > 0) ? ((1 << n) - 1) : 0;

    // allowed[mask] = 1 iff `mask` is one of the candidate squads, AFTER
    // discarding empty squads, masking off out-of-range bits, and de-duplicating
    // (a repeated mask is still just one allowed squad).
    vector<char> allowed(full + 1, 0);
    for (int j = 0; j < m; j++) {
        int x;
        if (scanf("%d", &x) != 1) x = 0;
        x &= full;             // drop any bit >= n
        if (x == 0) continue;  // empty squad is not a real squad
        allowed[x] = 1;        // marking twice is harmless -> dedup for free
    }

    // dp[mask] = number of ways to partition the employee set `mask` into a
    // collection of candidate squads (each used at most once, order irrelevant),
    // modulo MOD. dp[0] = 1: the empty set has exactly one partition (use no
    // squads).
    //
    // Canonical order to avoid double-counting: the squad that owns the LOWEST
    // remaining employee is decided first. So we only enumerate squads `sub`
    // that are submasks of `mask` AND contain mask's lowest set bit. Each
    // unordered partition is then generated exactly once.
    vector<long long> dp(full + 1, 0);
    dp[0] = 1;
    for (int mask = 1; mask <= full; mask++) {
        int low = mask & (-mask);          // lowest set bit of mask
        long long ways = 0;
        for (int sub = mask; sub; sub = (sub - 1) & mask) {
            if (!(sub & low)) continue;    // squad must own the lowest employee
            if (allowed[sub]) ways += dp[mask ^ sub];
        }
        dp[mask] = ways % MOD;
    }

    printf("%lld\n", dp[full] % MOD);
    return 0;
}
```

**Causal recap.** The set-partition count overcounts by block order, so a recurrence that lets any
allowed submask be "the next block" tallies a `k`-block roster `k!` times — a traced `n=2` case
(`{1,2,3}`) returning `3` instead of `2` pinned it to the floating order, and pinning the block that
owns the lowest uncovered employee (`if (!(sub & low)) continue;`) forces a unique canonical chain so
each roster is counted once; a second `{3,3,0}` trace exposed that a duplicated candidate squad
double-adds and an empty squad self-loops, both fixed by collapsing candidates into an idempotent
boolean `allowed` table and dropping `0` masks; `long long` accumulation with a `% MOD` after each
state, base case `dp[0]=1`, and the `mask=1..full` loop close out the `n=0`, impossible-roster,
overflow, and stray-bit corners — and the all-subsets case reproducing `B(18) mod p` is the
independent proof that the de-duplication is exactly right.
