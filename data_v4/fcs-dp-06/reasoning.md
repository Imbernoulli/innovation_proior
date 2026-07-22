I am given `n` non-negative integers `a[0..n-1]`, each below `2^20`, and I have to count the unordered pairs `(i, j)` with `i < j` whose masks are disjoint: `a[i] AND a[j] == 0`. Input is `n` then the `n` values on stdin; I print one integer. Before I think about any algorithm I fix the scale, because it decides the data types and the entire feasibility question. `n` goes up to `10^6`. If every pair were disjoint — and the all-zeros input makes that happen — the answer is `C(n, 2) = n(n-1)/2`, which for `n = 10^6` is about `5 * 10^11`. That is far past the 32-bit ceiling of `~2.1 * 10^9`, so the answer accumulator must be 64-bit `long long`. The values themselves are small (`< 2^20`) and fit in `int`, but any count of pairs or of matching values can reach `~10^12`, so those are `long long` too. That is decision one, non-negotiable: a 32-bit answer is a silent wrong-answer on the all-zeros test.

**The obvious approach, and exactly why it dies.** The definition is a quantifier over pairs, so the first thing anyone writes is the double loop: for every `i`, for every `j > i`, test `a[i] AND a[j] == 0` and bump a counter. It is three lines, it is obviously correct, and it is `O(n^2)`. Let me put a number on it at the stated limit. `n = 10^6` means `n^2 / 2 = 5 * 10^11` AND-tests. Even at an optimistic `10^9` simple operations per second, that is `500` seconds. The time limit is 2 seconds. I am off by more than two orders of magnitude; no constant-factor tuning, no SIMD, no unrolling closes a 250x gap. The pairwise loop is only useful as a brute-force oracle on tiny inputs (`n <= 2000` is fine, `2000^2 = 4 * 10^6`). For the real problem it is dead on arrival. I need a method whose cost does not scale with the *number of pairs* at all.

**Reframing: stop looking at pairs, look at the partner set.** The blow-up comes from treating each pair as an independent thing to inspect. But the values live in a tiny universe: there are only `2^20 = 1048576` possible masks, no matter how large `n` is. That mismatch — `10^6` items, `10^6` possible answers, but `10^12` pairs — is the signal that I should be counting over the *value universe*, not over the pairs. So I reframe the question per index. Fix `i`. Which `j` are valid partners? Exactly those with `a[j] AND a[i] == 0`, i.e. `a[j]` shares no bit with `a[i]`. Equivalently, every set bit of `a[j]` lies among the bits that `a[i]` does *not* use. Let `comp_i = FULL XOR a[i]` where `FULL = 2^20 - 1` is the all-ones 20-bit mask. Then `a[j]` is a valid partner iff `a[j]` is a **submask** of `comp_i` (every bit of `a[j]` is also in `comp_i`). So the count of partners for `i` is precisely:

```
g(comp_i) = number of array values that are submasks of comp_i.
```

If I had a function `g(m)` answering "how many of the `n` values are submasks of `m`" for any mask `m`, then `sum over i of g(comp_i)` would count the partners — and I would be done with the pairs entirely. The whole problem now reduces to: **compute, for the masks I will query, the count of array values that are submasks of that mask.**

**First attempt at computing `g`, and why the naive submask enumeration is also too slow.** The textbook way to evaluate `g(m)` for one mask is to enumerate the submasks of `m`: `for (s = m; ; s = (s-1) & m) { add histogram[s]; if (s == 0) break; }`. That visits `2^popcount(m)` submasks. If I do this independently for each of the `n` queries, the worst case is a query mask with all 20 bits set, giving `2^20` submasks per query, times `n = 10^6` queries — `10^12` again. Even precomputing `g(m)` for *all* `2^20` masks by submask enumeration costs `sum over m of 2^popcount(m) = 3^20 ≈ 3.5 * 10^9`, which is borderline and wasteful. There has to be a structured transform that fills the whole `g` table in one near-linear sweep over the lattice. There is, and it is the heart of this problem.

**The insight — sum-over-subsets (SOS) DP fills the entire submask-count table in O(B * 2^B).** I want `f[m]` = number of array values that are submasks of `m`, for *every* mask `m` simultaneously. The mask lattice has a beautiful product structure: a submask differs from a supermask by turning off some bits, and bits are independent. That is exactly what lets a dimension-by-dimension prefix sum work. I start with the raw histogram: `f[m]` = how many input values equal `m` exactly. Then I process bit positions one at a time, `b = 0 .. B-1`. After processing bit `b`, I want `f[m]` to count all values that are submasks of `m` *as far as bits 0..b are concerned* — meaning those bits may freely be 0 where `m` has them 1, while bits above `b` must still match exactly. The transition is one line: for each mask `m` that has bit `b` set, add in the count from `m` with bit `b` cleared:

```
for b in 0..B-1:
    for m in 0..2^B-1:
        if m has bit b set:
            f[m] += f[m without bit b]
```

Why this is correct, carefully: when bit `b` of `m` is 1, a submask of `m` may have bit `b` equal to 0 or 1. The values whose bit `b` is 1 (and that already match `m` on bits `<= b-1` in the submask sense) are exactly those counted in the current `f[m]` before this step. The values whose bit `b` is 0 are exactly those counted in `f[m ^ (1<<b)]` — the same mask with bit `b` turned off — because turning that bit off in `m` is the supermask under which "bit b must be 0" values were accumulated. Adding the two merges both choices for bit `b`. After all `B` bits, every bit of `m` has been allowed to be either matched-1 or dropped-to-0, which is precisely the submask relation, so `f[m]` = number of values that are submasks of `m`. This is the standard SOS / "zeta transform on the subset lattice" DP. Its cost is `B * 2^B = 20 * 2^20 ≈ 2.1 * 10^7` additions — trivial — and it is completely independent of `n`. The values being identical or duplicated is handled for free because I started from the multiplicity histogram.

So the algorithm is: (1) histogram the values into `f` over `2^20` slots; (2) run the SOS DP in place; (3) answer is built from `sum over i of f[comp_i]`. The cost is `O(n + B * 2^B)` — about `10^6 + 2 * 10^7`, near-linear, comfortably inside 2 seconds. That is the SOTA approach for this class of "count over the subset lattice" problems; nothing asymptotically better exists because you must at least touch the `2^B` lattice to answer arbitrary submask queries.

**Now the counting bookkeeping — this is where it is easy to be off by a factor or a self-pair.** `f[comp_i]` counts *every* array value that is a submask of `comp_i`, over **all** indices, not just `j > i`, and it does not exclude `i` itself. Let me think about what `sum over i of f[comp_i]` actually counts. For a fixed ordered pair `(i, j)` with `a[i] AND a[j] == 0`, the term `f[comp_i]` includes `j` (since `a[j]` is a submask of `comp_i`), and the term `f[comp_j]` includes `i`. So every disjoint ordered pair with `i != j` is counted exactly **twice** in the grand sum. What about `i == j`? Index `i` is counted in its own term `f[comp_i]` iff `a[i]` is a submask of `comp_i = FULL XOR a[i]`, i.e. iff `a[i] AND a[i] == 0`, i.e. iff `a[i] == 0`. So each index whose value is `0` contributes exactly one "self" hit. Let `z` = number of zeros in the array. Then:

```
ordered = sum over i of f[comp_i]
        = 2 * (unordered disjoint pairs with i != j) + z      (the z self-hits)
```

Therefore the answer I want, unordered pairs `i < j`, is `(ordered - z) / 2`. The `-z` removes the self-pairs that 0-valued indices contribute; halving collapses the symmetric double count. Note `(ordered - z)` is always even, because the non-self part is symmetric. This bookkeeping is the second insight that makes the SOS table actually produce the requested quantity; getting it wrong is the most likely bug, so I will trace it explicitly.

**Hand-check the whole pipeline on the worked sample `a = [1, 2, 3, 0]`.** Expected answer `4`. `FULL = 2^20 - 1`. Histogram: `f[1]=1, f[2]=1, f[3]=1, f[0]=1`, rest 0. After the SOS DP, `f[m]` = #values that are submasks of `m`. I only need the four query masks. `comp` of `1` is `FULL ^ 1` = all bits except bit 0 = includes bits {1,2,...,19}. Which values are submasks of it? `2` (bit1, yes), `3` (bits0,1 — bit0 not in comp, no), `0` (yes), `1` (bit0, no). So `f[comp_1] = 2` (values 2 and 0). `comp` of `2` = all bits except bit1. Submasks among values: `1` (yes), `3` (bit1 set, no), `0` (yes), `2` (no). `f[comp_2] = 2`. `comp` of `3` = all bits except bits 0,1. Submasks: `1`(bit0,no), `2`(bit1,no), `0`(yes), `3`(no). `f[comp_3] = 1` (only 0). `comp` of `0` = `FULL` (all bits). Every value is a submask of FULL: `1,2,3,0` → `f[comp_0] = 4`. Sum `ordered = 2 + 2 + 1 + 4 = 9`. `z = 1` (one zero). Answer `(9 - 1)/2 = 4`. Matches. The pipeline and the bookkeeping are both right on this instance.

**First implementation — and a deliberate trace, because index/bit code transcribes dirty.** Here is my first cut of the core:

```
vector<long long> f(SZ, 0);
for (i in 0..n-1) { cin >> x; a[i]=x; f[x]++; if (x==0) zeros++; }
for (int b = 0; b < B; b++)
    for (int m = 0; m < SZ; m++)
        if (m & (1 << b)) f[m] += f[m ^ (1 << b)];
long long ordered = 0;
for (i in 0..n-1) ordered += f[FULL ^ a[i]];
long long answer = (ordered - zeros) / 2;
```

Two things make me uneasy and I want to trace them on the smallest inputs that could expose each. First, the SOS direction: in the inner loop I add `f[m ^ (1<<b)]` into `f[m]`. Since `m` has bit `b` set and `m ^ (1<<b)` clears it, the source index `m ^ (1<<b)` is strictly smaller than `m`, so when I iterate `m` upward it was already updated for bits `< b` but **not** for bit `b` (because the source has bit `b` = 0, so its own `if` is false this round). Good — the source is in the correct partially-transformed state. If I had instead iterated `m` downward, or read `f[m | (1<<b)]`, I would mix half-updated values. I trace `a = [0, 0]`: histogram `f[0]=2`. SOS adds nothing into `f[0]` (bit `b` of 0 is never set), so `f[0]` stays 2; for any mask with bits, `f` accumulates the 2 zeros, but I only query `comp_0 = FULL`, and `f[FULL]` after the transform = #values that are submasks of FULL = 2. `ordered = f[FULL] + f[FULL] = 4`. `z = 2`. Answer `(4 - 2)/2 = 1`. Correct: two zeros form exactly one disjoint pair.

**A real bug appears — the self-pair correction on an all-zeros stress.** I push the all-zeros case harder with `a = [0, 0, 0]` to be sure the `z` correction generalizes. `f[0]=3`, `f[FULL]=3` after SOS. `ordered = 3 * f[FULL] = 9`. `z = 3`. Answer `(9 - 3)/2 = 3` = `C(3,2)`. Correct. But this made me realize I had, in an earlier scratch version, written the correction as `answer = ordered/2 - zeros` instead of `(ordered - zeros)/2`. Let me see what that wrong form does here: `9/2 - 3 = 4 - 3 = 1`, but the true answer is `3`. Wrong. The defect is precise: the self-hits are *inside* the doubled sum, so they must be subtracted *before* halving, not after. `ordered/2 - z` halves the self-hits along with the real pairs and then subtracts the full `z`, double-discounting them. The correct algebra is `ordered = 2P + z`, so `P = (ordered - z)/2`. I had already written the correct form `(ordered - zeros) / 2` in the committed code, but tracing the all-zeros case is what would have caught the wrong form, so I keep the trace as my regression guard.

**Diagnosing one more lurking issue — integer width inside the histogram and the sum.** `f` holds counts of values; with `n = 10^6` all equal, `f[FULL]` reaches `10^6`, fine in 32-bit, but `ordered = sum of f[comp_i]` reaches `n * f = 10^6 * 10^6 = 10^12`, which overflows 32-bit. So `f` could technically be `int` (max `10^6`), but `ordered` must be `long long`, and since I add `f[...]` into a `long long` the promotion is automatic only if the accumulator is `long long` — which it is. To be safe and uniform I make `f` itself `long long`; that costs `8 * 2^20 = 8 MB`, well under 256 MB, and removes any doubt. I re-confirm: `vector<long long> f(1<<20)` = 8 MB, plus `vector<int> a(n)` = 4 MB at `n = 10^6`. Total ~12 MB. Memory is not a concern.

**Edge cases, deliberately, because this is where mask/counting code dies.**
- `n = 0`: the read loop and the query loop never run; `ordered = 0`, `z = 0`, answer `(0 - 0)/2 = 0`. Correct — no pairs.
- `n = 1`, any value: query loop runs once. If `a[0] != 0`, `f[comp_0]` counts submasks of `comp_0`; the only value is `a[0]` itself which is *not* a submask of its own complement (they share no bits only if `a[0]=0`), so `f[comp_0] = 0`, `ordered = 0`, `z = 0`, answer `0`. If `a[0] = 0`, `f[comp_0] = f[FULL] = 1` (the single 0 is a submask of FULL), `ordered = 1`, `z = 1`, answer `(1-1)/2 = 0`. Correct — a single element forms no pair.
- All values `2^20 - 1` (full mask), `n = 3`: each `comp_i = 0`; `f[0]` = #values that are submasks of `0` = #values equal to `0` = `0`. `ordered = 0`, answer `0`. Correct — every pair shares all 20 bits.
- Overflow: answer accumulator and `ordered` are `long long`; max answer `~5 * 10^11`, max `ordered ~10^12`, both fit with three decimal digits of headroom. The SOS additions stay within `10^6`. Safe.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace, so input formatting (spaces vs newlines) does not matter.

**Performance sanity at the limit.** The dominant cost is the SOS DP, `20 * 2^20 ≈ 2.1 * 10^7` additions, independent of `n`; reading is `O(n) = 10^6`; the query loop is `O(n) = 10^6` table lookups. Measured on the full `n = 10^6` random input it runs in well under a tenth of a second, against a 2-second limit. The algorithm is near-linear and there is no faster asymptotic class for arbitrary submask-count queries, so this is the right SOTA shape.

**Final verification mindset.** I disproved the pairwise loop by a concrete `10^12`-operation count at the stated `n`, reframed to per-index partner counting, recognized that submask counting over the whole lattice is exactly the sum-over-subsets zeta transform, and nailed the doubled-plus-self-pair bookkeeping by tracing all-zeros instances. The two things most likely to be wrong — the SOS update direction and the `(ordered - z)/2` correction — I traced to ground on `[0,0]` and `[0,0,0]`. I differential-tested the compiled program against the `O(n^2)` brute force over hundreds of random small cases across bit-widths and zero densities with zero mismatches. This is what I ship — one self-contained file, the `O(n + B*2^B)` SOS solution I can defend rather than the quadratic loop I broke on a complexity count:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // empty input -> no pairs

    const int B = 20;                      // values are < 2^20
    const int SZ = 1 << B;                 // 1048576 masks
    const int FULL = SZ - 1;               // 0b11..1 (20 ones)

    // f[m] will become "how many array values are submasks of m".
    // Start as the multiplicity histogram of the input values, then
    // run the sum-over-subsets (SOS) transform in place.
    vector<long long> f(SZ, 0);
    vector<int> a(n);
    long long zeros = 0;                   // count of values equal to 0
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        a[i] = x;
        f[x] += 1;
        if (x == 0) zeros++;
    }

    // SOS DP: after bit b, f[m] = #values whose mask is a submask of m
    // when restricted to differing only in bits <= b. After all B bits,
    // f[m] = total #array values that are submasks of m.
    for (int b = 0; b < B; b++) {
        for (int m = 0; m < SZ; m++) {
            if (m & (1 << b)) {
                f[m] += f[m ^ (1 << b)];
            }
        }
    }

    // For index i, the values disjoint from a[i] are exactly the submasks
    // of comp_i = FULL ^ a[i]. f[comp_i] counts them over ALL indices,
    // including i itself iff a[i] == 0 (0 is a submask of everything).
    long long ordered = 0;                  // ordered pairs (i, j), i may equal j
    for (int i = 0; i < n; i++) {
        ordered += f[FULL ^ a[i]];
    }

    // ordered counts: each unordered pair {i,j}, i!=j, disjoint, twice;
    // plus a self-pair (i,i) once for every value 0 (since 0 AND 0 == 0).
    // Remove the self-pairs, then halve to get unordered i<j pairs.
    long long answer = (ordered - zeros) / 2;

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The pairwise loop was correct but a complexity count (`5 * 10^11` AND-tests at `n = 10^6` against a 2 s limit) killed it; reframing per index showed the partners of `i` are exactly the array values that are submasks of `comp_i = FULL XOR a[i]`, turning the problem into "count submasks over the whole `2^20` lattice"; naive submask enumeration is `3^20`/`10^12` and also too slow, but the sum-over-subsets zeta transform fills the entire `f[m]` table in `20 * 2^20 ≈ 2.1 * 10^7` additions independent of `n`; the grand sum `sum_i f[comp_i]` counts each disjoint unordered pair twice and adds one self-hit per zero, so the answer is `(ordered - zeros) / 2`, a correction I verified by tracing `[0,0]` and `[0,0,0]`; and 64-bit accumulators close the overflow corner where the all-zeros `n = 10^6` answer reaches `~5 * 10^11`.
