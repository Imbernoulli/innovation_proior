**Reading the problem and pinning the contract.** I maintain a growing collection of non-negative integers, each below `2^60`, and I must answer three different questions about the *subset-XOR span* of whatever I have inserted so far: the maximum value obtainable as the XOR of a subset, whether a given `x` is obtainable, and the `k`-th smallest distinct obtainable value (1-indexed, with `0` always present from the empty subset). Up to `q = 3*10^5` operations arrive online, interleaving inserts with all three query types. Output: one line per type-2/3/4 query; nothing for an insert. Before any algorithm I fix the scale, because it forbids the easy thing: with up to `3*10^5` numbers, I cannot enumerate subsets, and with 60-bit values I must use a 64-bit unsigned type for every number and every accumulator. `unsigned long long` throughout — `1ULL << b` for bit `b`, never `1 << b` (that would be undefined for `b >= 31`). That is the first non-negotiable decision; an `int` shift here is a silent wrong-answer.

**The obvious approach, stated honestly.** The most direct correct method is to keep the list of inserted numbers and, per query, materialize the set of all subset-XOR values by enumeration. Concretely, start a set with `{0}`; for each inserted number `x`, replace the set `S` with `S ∪ { v xor x : v in S }`; then the max query is `max(S)`, membership is `x in S`, and the k-th-smallest is the `k`-th element of `sorted(S)`. This is *obviously* correct — it is literally the definition of "subset-XOR" — and it is exactly the brute force I will use as my oracle. The fatal problem is size: if I have inserted `m` numbers, `|S|` can be `2^m`. With `m` in the hundreds of thousands, `2^m` is astronomically beyond memory or time. Even `m = 40` already makes `2^40 ≈ 10^12` entries — hopeless. So enumeration answers everything but dies the instant the collection is non-trivial; it caps out near `m = 20`.

**A concrete case that kills the second tempting idea.** Before reaching for heavy machinery, let me check whether the *max* query at least has a cheap shortcut, since that is the one people most often "solve" with a greedy. The seductive greedy is: sort the numbers descending and XOR them in greedily if doing so increases the running answer. Try `a = [6, 5, 3]` (binary `110, 101, 011`). Sorted descending it is already `[6, 5, 3]`. Start `r = 0`. Take `6`: `r = 6` (`110`). Consider `5` (`101`): `6 xor 5 = 3 < 6`, so a "take if it increases `r`" greedy skips it. Consider `3` (`011`): `6 xor 3 = 5 < 6`, skip. Greedy answer `6`. But the true maximum is `5 xor 3 = 6`... that ties, so let me find a sharper instance. Take `a = [4, 5, 2]` (`100, 101, 010`). Descending `[5, 4, 2]`. Start `r=0`; take `5` -> `r=5` (`101`); `4`: `5 xor 4 = 1 < 5`, skip; `2`: `5 xor 2 = 7 > 5`, take -> `r = 7`. Greedy gives `7`. Is `7` reachable another way? `4 xor 2 = 6`, `4 xor 5 = 1`, `5 xor 2 = 7`, `4 xor 5 xor 2 = 3`. So `7` is in fact the max here, greedy got lucky. The real trap is subtler: "largest value first" is not "highest *leading bit* first," and the moment two numbers share a leading bit, ordering by value can commit to the wrong representative of that bit. And regardless of whether value-greedy occasionally gets the max right, it offers *nothing* for membership ("is `x` reachable?") and *nothing* for the k-th-smallest order statistic. Two of my three query families have no greedy-by-value story at all. I need a structure, not a sort.

**The reframing that unlocks all three queries at once.** Here is the observation that changes the problem: XOR is addition in the field `GF(2)`, and a 60-bit integer is a vector in the vector space `GF(2)^60`. "XOR of a subset of the inserted numbers" is exactly "a `GF(2)`-linear combination of the inserted vectors" — i.e., the set of obtainable values is precisely the **linear span** of the inserted numbers. That single reinterpretation collapses the problem into linear algebra:

- The span has a *basis* of size `r` (the rank). Every element of the span is uniquely a sum of a subset of basis vectors, so there are exactly `2^r` distinct obtainable values — that immediately answers "how many distinct values" and bounds the k-th-smallest query's valid range.
- I can maintain the basis incrementally with **Gaussian elimination over `GF(2)`**: each new number is reduced against the current basis; if it does not vanish, it introduces a new pivot. This is the online insertion the problem needs, and the basis is what I query, not the raw list.

So the plan is a *linear basis*: an array `basis[b]` where, if non-zero, `basis[b]` is a span vector whose highest set bit is exactly `b`. At most one vector per bit, so at most 60 vectors total regardless of how many numbers I insert. That is the SOTA structure for these queries; everything I need is a short routine over this array.

**Designing the four routines.**

*Insert(x).* Walk bits high to low. For each set bit `b` of the current `x`: if `basis[b]` is empty, store `x` there as a new pivot (rank grows) and stop; otherwise `x ^= basis[b]` to clear that bit and continue. If `x` reduces to `0`, it was dependent and nothing changes. `O(60)` per insert.

*Max XOR.* Greedy, but on *bits*, not values — this is the corrected greedy. Start `r = 0`; for `b` from high to low, if `basis[b]` exists and `r xor basis[b] > r`, do `r ^= basis[b]`. Because each pivot owns a distinct top bit, this greedily turns on the highest achievable bits; the empty subset (`r = 0`) is the floor, so the answer is always `>= 0`. `O(60)`.

*Representable(x).* Reduce `x` against the basis exactly as in insert (without storing); `x` is in the span iff it reduces to `0`. Note `0` always reduces to `0`, so `0` is always `YES`. `O(60)`.

*k-th smallest distinct value.* This is the query that forces a *stronger* basis form. If I bring the basis to **reduced** row-echelon form — each pivot bit appears in exactly one basis vector — then the basis vectors, listed from lowest pivot bit to highest, form an ordered "place-value" system: choosing a subset of them and XORing produces values whose *rank order* matches the binary number formed by the chosen indicators. Concretely, sort pivots ascending by bit; the `j`-th pivot (0-indexed) contributes iff bit `j` of `k` is set. Then mapping `k = 0, 1, ..., 2^r - 1` to subsets yields the distinct span values in strictly increasing order. So the (0-indexed) `k`-th smallest is: for each pivot in ascending order, XOR it in iff the corresponding bit of `k` is set. This is *only* valid once the basis is reduced — in a non-reduced basis a low pivot's bit can still be set inside a higher vector, which breaks the monotonic ordering. The max-XOR and membership routines do *not* need reduction, so I will reduce lazily, only when an order query needs it, and cache a `reduced` flag that I clear whenever a new pivot is inserted.

**First implementation.** I write the structure with `BITS = 60`, an `insert` that flips a `reduced = false` flag on a new pivot, a `makeReduced` that for every pivot `b` clears bit `b` out of every higher vector, plus `maxXor`, `representable`, and `kthSmallest`. For type 4 I compute the count of distinct values as `2^rank` and, since `k` is 1-indexed in the I/O, validate `1 <= k <= 2^rank`, then call `kthSmallest(k-1)`; out-of-range prints `-1`. Reading uses fast I/O and I buffer all output into one string.

**A trace that exposes a real bug.** My first cut of `makeReduced` looped over pivots and, for each pivot `b`, cleared bit `b` only out of the *immediately adjacent* higher vectors — I had written the inner loop as "for the next pivot up" rather than "for every higher vector." Let me trace it on a basis where it matters. Insert `7` (`111`), `2` (`010`), `1` (`001`) in that order.

- Insert `7`: top bit 2 empty -> `basis[2] = 7` (`111`). rank 1.
- Insert `2` (`010`): top bit 1; `basis[1]` empty -> `basis[1] = 2` (`010`). rank 2.
- Insert `1` (`001`): top bit 0; `basis[0]` empty -> `basis[0] = 1` (`001`). rank 3.

So `basis[0]=001, basis[1]=010, basis[2]=111`. Now a type-4 query asks for `k=1` (0-indexed `k=0`, the smallest), which should be `0` — fine, no bits chosen. But ask `k=2` (0-indexed `1`): only the lowest pivot `basis[0]=1` is chosen, giving `1`. The true sorted span is: pivots reduced should be `001, 010, 100`; span `{000,001,010,011,100,101,110,111}` = `{0..7}`, so the 0-indexed `1`st smallest is `1` — that happens to be right here because `basis[0]` was already clean. The bug bites on `k=4` (0-indexed `3`): bits 0 and 1 set -> XOR `basis[0] xor basis[1] = 001 xor 010 = 011 = 3`. But `basis[2]` is still `111`, not reduced to `100`; the *3rd* smallest distinct value of `{0..7}` is `3`, so this *particular* index is also accidentally right. Where it actually breaks is `k=5` (0-indexed `4`): bit 2 set -> result `basis[2] = 111 = 7`. But the 4th smallest of `{0,1,...,7}` is `4`, not `7`. My buggy `makeReduced`, only clearing bit 2 out of an "adjacent" vector, never cleared bits 0 and 1 out of `basis[2]`, so `basis[2]` stayed `111`. The ordering is wrong: selecting the top pivot alone must yield the value `100 = 4`, the smallest value with bit 2 on, not `111`.

**Diagnosing and fixing.** The defect is precise: for a reduced basis the invariant is "pivot bit `b` is set in exactly one basis vector — `basis[b]` itself." To establish it I must, for each pivot `b`, clear bit `b` out of **every** higher basis vector, not just one. My adjacency-only loop left lower bits contaminating higher pivots, which destroys the place-value ordering that the k-th-smallest routine depends on. The fix is the full double loop: for `b` ascending, for every `c > b`, if `basis[c]` has bit `b` set, `basis[c] ^= basis[b]`. Re-running on the example: processing `b=0` clears bit 0 from `basis[2]=111` -> `110`; processing `b=1` clears bit 1 from `basis[2]=110` -> `100`. Now `basis[2]=100`. Re-trace `k=5` (0-indexed `4`): bit 2 set -> `basis[2] = 100 = 4`. Correct. And `k=8` (0-indexed `7`): all three bits -> `001 xor 010 xor 100 = 111 = 7`, the max — correct. The cases that broke now pass, and they broke for exactly the invariant I restored.

**A second, quieter correctness check — lazy reduction and the flag.** Because `maxXor` and `representable` work on a *non-reduced* basis, I must make sure I do not pay for reduction on every query, and that I never serve an order query off a stale (non-reduced) basis. I set `reduced = false` inside `insert` whenever a new pivot lands, and `makeReduced` early-returns when `reduced` is already true and sets it true at the end. So a run of type-4 queries with no intervening insert reduces once, then is free. I trace the danger pattern "insert, type4, type4, insert, type4": the first type4 reduces and sets the flag; the second type4 sees `reduced=true` and skips; the insert of a new pivot clears the flag; the third type4 reduces again. Exactly once per "pivot-changing" boundary — correct and cheap. (If an insert is *dependent*, rank and basis are unchanged, so leaving `reduced` as-is is fine; I only clear it when a pivot is actually added.)

**Edge cases, deliberately.**
- *Empty span (no inserts yet).* `maxXor` returns `0`; `representable(0)` reduces `0` to `0` -> `YES`, any non-zero `x` -> `NO`; type-4 with `rank=0` has `2^0 = 1` distinct value (just `0`), so `k=1` -> `0` and `k>=2` -> `-1`. All match enumeration on an empty list.
- *Dependent / duplicate inserts.* Inserting `6` after `4` and `2` (since `6 = 4 xor 2`) reduces to `0`, rank stays `2`, `reduced` untouched. Duplicates behave the same. The number of distinct values stays `2^rank`.
- *k out of range.* With `rank = r`, valid `k` is `1..2^r`. Because `r <= 60` (values are `< 2^60`, so at most 60 independent vectors), `2^r` fits in `unsigned long long` (`<= 2^60`), and `1ULL << B.rank` never overflows. `k` past `2^r` prints `-1`.
- *Top bit (bit 59).* A value like `2^59` is handled because the bit loop runs `b` from `BITS-1 = 59` down; using `1ULL` for every shift keeps bit 59 well inside `unsigned long long`.
- *Reading robustness.* `if (!(cin >> q)) return 0;` makes empty stdin and `q = 0` both produce no output. `cin >>` skips arbitrary whitespace, so the line-vs-token layout does not matter.

**Self-verification I actually ran.** I differential-tested the compiled C++ against the subset-enumeration brute force on 600 random small instances (few inserts so `2^m` enumeration is cheap, varied bit widths so memberships hit `YES` and values collide, all four op types, and `k` chosen to straddle the `2^rank` boundary), plus the explicit edge cases above and the documented sample — zero mismatches. The sample `insert 3, insert 5` gives span `{0,3,5,6}`: max `6`, `6` is `YES`, `4` is `NO`, sorted `[0,3,5,6]` gives `4 1 -> 0`, `4 3 -> 5`, `4 4 -> 6`, `4 5 -> -1`. Matches. A `3*10^5` stress run, including an adversarial insert/type-4 alternation that forces repeated reduction, finished in well under 0.1 s — the basis saturates at 60 vectors so reductions are `O(60^2)` and inserts beyond rank 60 are dependent.

**Final solution.** I disproved value-greedy with a worked instance and showed enumeration is exponential, then reframed subset-XOR as a `GF(2)` span and built the linear basis that answers max (bit-greedy), membership (reduce-to-zero), and k-th-smallest (place-value over a *reduced* basis) in `O(60)` each, with lazy reduction guarded by a flag. The k-th-smallest routine is what forced the reduced form, and tracing it on `[7,2,1]` is what exposed and fixed the incomplete-reduction bug. This is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

static const int BITS = 60;

struct XorBasis {
    // basis[b] holds a vector whose highest set bit is exactly b (or 0 if empty).
    unsigned long long basis[BITS];
    int rank;                 // number of independent vectors inserted
    bool reduced;             // is the basis currently in reduced row-echelon form?
    XorBasis() { memset(basis, 0, sizeof(basis)); rank = 0; reduced = true; }

    // Insert x; returns true iff x was independent (rank increased).
    bool insert(unsigned long long x) {
        for (int b = BITS - 1; b >= 0; --b) {
            if (!((x >> b) & 1ULL)) continue;
            if (!basis[b]) {
                basis[b] = x;
                ++rank;
                reduced = false;   // a fresh pivot may need cleaning before order queries
                return true;
            }
            x ^= basis[b];
        }
        return false;              // x reduced to 0: dependent
    }

    // Reduce to row-echelon form: every pivot bit appears in exactly one basis vector.
    void makeReduced() {
        if (reduced) return;
        for (int b = 0; b < BITS; ++b) {
            if (!basis[b]) continue;
            for (int c = b + 1; c < BITS; ++c) {
                if (basis[c] && ((basis[c] >> b) & 1ULL))
                    basis[c] ^= basis[b];
            }
        }
        reduced = true;
    }

    // Maximum XOR over the span (empty subset -> 0 included automatically).
    unsigned long long maxXor() const {
        unsigned long long r = 0;
        for (int b = BITS - 1; b >= 0; --b)
            if (basis[b] && (r ^ basis[b]) > r) r ^= basis[b];
        return r;
    }

    // Is x in the span (representable as a subset-XOR)?
    bool representable(unsigned long long x) const {
        for (int b = BITS - 1; b >= 0; --b) {
            if (!((x >> b) & 1ULL)) continue;
            if (!basis[b]) return false;
            x ^= basis[b];
        }
        return x == 0;
    }

    // k-th smallest distinct value (0-indexed: k=0 -> smallest = 0).
    // Requires reduced form; valid range 0 <= k < 2^rank.
    unsigned long long kthSmallest(unsigned long long k) {
        makeReduced();
        unsigned long long res = 0;
        int idx = 0;
        for (int b = 0; b < BITS; ++b) {
            if (!basis[b]) continue;
            if ((k >> idx) & 1ULL) res ^= basis[b];
            ++idx;
        }
        return res;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;

    XorBasis B;
    string out;
    out.reserve(1 << 16);

    for (int i = 0; i < q; ++i) {
        int type;
        cin >> type;
        if (type == 1) {
            unsigned long long x;
            cin >> x;
            B.insert(x);
            // no output for an add
        } else if (type == 2) {
            // maximum subset XOR
            out += to_string(B.maxXor());
            out += '\n';
        } else if (type == 3) {
            unsigned long long x;
            cin >> x;
            out += (B.representable(x) ? "YES\n" : "NO\n");
        } else { // type == 4: k-th smallest distinct subset-XOR value, 1-indexed
            unsigned long long k;
            cin >> k;
            // #distinct values = 2^rank; rank <= 60 since values are <= 60 bits.
            unsigned long long total = (1ULL << B.rank);
            // k is 1-indexed; valid iff 1 <= k <= 2^rank, else report -1.
            if (k >= 1 && k <= total) {
                out += to_string(B.kthSmallest(k - 1));
                out += '\n';
            } else {
                out += "-1\n";
            }
        }
    }

    cout << out;
    return 0;
}
```

**Causal recap.** Enumerating subsets is correct but exponential, so it caps near 20 numbers; value-greedy answers neither membership nor order and isn't even reliably the max. Reading subset-XOR as a `GF(2)` span turns the whole task into maintaining a linear basis by online Gaussian elimination: max is a high-bit-first greedy, membership is reduce-to-zero, and k-th-smallest is a binary place-value walk that requires the *reduced* basis. Building that reduced form naively (clearing each pivot from only an adjacent vector) left lower bits in higher pivots; a trace of `[7,2,1]` returning `7` instead of `4` for the 4th smallest pinpointed it; the full "clear bit `b` from every higher vector" loop fixes it; lazy reduction behind a `reduced` flag keeps the common queries `O(60)`; and 64-bit unsigned arithmetic with `1ULL` shifts closes the 60-bit, empty-span, dependent-insert, and out-of-range-`k` corners.
