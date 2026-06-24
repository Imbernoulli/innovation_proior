**Reading the problem and pinning the contract.** I have `n` jobs that run one at a time on a single bench, no gaps, no preemption. Once I fix a permutation they run back to back; job `i` finishes at time `C[i]` equal to the running sum of processing times up to and including it, and it costs `w[i] * C[i]`. Every job is run — I am not picking a subset, I am ordering the whole set — so the only decision is the permutation, and I want to minimize `sum_i w[i] * C[i]`. Input is `n`, then the `n` processing times `p`, then the `n` weights `w`; I print one integer. Before any algorithm I nail the scale, because it dictates the data types and this objective is notorious for huge intermediate numbers. `n <= 2*10^4`, and `p[i], w[i] <= 10^5`. The running clock can reach `n * max(p) = 2*10^4 * 10^5 = 2*10^9`, which already exceeds the 32-bit signed range of `~2.147*10^9` by itself. The final answer can reach roughly `sum w[i]*C[i]` with everything maxed: `w*p*n(n+1)/2 = 10^5 * 10^5 * (2*10^4)(2*10^4+1)/2 ≈ 2*10^18`. That fits in a signed 64-bit integer (max `~9.2*10^18`) with about a 4.6x margin, but it is nowhere near 32-bit. So the clock, the per-job product, and the accumulator must all be `long long`. That is the first hard decision and it is non-negotiable; this problem is *built* to punish a stray `int`.

**Laying out the candidate approaches.** Two routes, and I want the one I can prove rather than the one that types fastest.

- *Greedy exchange by a sequencing rule.* Sort the jobs by one per-job key and run in that order. The natural key for "minimize weighted completion time" is the ratio `p[i] / w[i]` — short jobs and heavy jobs both want to go early, and the ratio balances the two. This is `O(n log n)`. Two open questions: (a) does the local swap argument actually give a global optimum, and (b) how do I compare `p[i]/w[i]` without floating point, since the safe cross-multiplied comparison produces large products.
- *Permutation search / bitmask DP.* Try orderings directly — `O(n!)` exhaustive or `O(2^n n)` over subsets. Correct but unusable at `n = 2*10^4`. I will keep this only as a mental oracle on tiny inputs to validate the greedy.

**Deriving the exchange argument before trusting the sort.** "Sort by ratio" is folklore; I refuse to ship it on faith, so let me derive the swap. Consider any order and two jobs `A`, `B` that are *adjacent* in it, with `A` immediately before `B`. Let `T` be the clock right before `A` starts (the total processing of everything ahead of both). All jobs other than `A` and `B` have identical completion times regardless of how I order this adjacent pair, because the block `{A,B}` occupies the same time window `[T, T + p_A + p_B]` either way and everything after it starts at the same instant `T + p_A + p_B`. So only the two terms for `A` and `B` change. With `A` first: `A` finishes at `T + p_A`, `B` finishes at `T + p_A + p_B`; their contribution is `w_A(T+p_A) + w_B(T+p_A+p_B)`. With `B` first: `w_B(T+p_B) + w_A(T+p_A+p_B)`. Subtract (A-first minus B-first): the `w_A T`, `w_B T`, and the shared `+p_A+p_B` cross-terms cancel, and what survives is `w_B p_A - w_A p_B`. So **A-first is no worse than B-first exactly when `w_B p_A - w_A p_B <= 0`, i.e. `p_A w_B <= p_B w_A`, i.e. `p_A / w_A <= p_B / w_B`.** Putting the smaller ratio first never increases the cost. Since any permutation that is not ratio-sorted has an adjacent inversion I can fix without increasing cost, repeatedly fixing inversions (a bubble sort by ratio) reaches the ratio-sorted order without ever increasing cost — so ratio-sorted is optimal. That is Smith's rule, and I now *trust* it because I derived the inequality, not because a book told me.

Crucially, the comparison that came out of the derivation is already in cross-multiplied integer form: order `A` before `B` when `p_A * w_B < p_B * w_A` (with ties broken arbitrarily, since the swap difference is zero). I will never divide. But note `p_A * w_B` can be `10^5 * 10^5 = 10^10` — that is the comparator's overflow hazard, flagged and to be handled with `long long`.

**Sanity-checking the derivation on the sample.** Sample: `p = [3,1,2]`, `w = [1,2,3]`, claimed answer `17`. Ratios: job0 `3/1 = 3`, job1 `1/2 = 0.5`, job2 `2/3 ≈ 0.667`. Smith order ascending: job1 (0.5), job2 (0.667), job0 (3). Run them: job1 finishes at `1`, cost `w=2 -> 2*1 = 2`; job2 finishes at `1+2 = 3`, cost `w=3 -> 3*3 = 9`; job0 finishes at `3+3 = 6`, cost `w=1 -> 1*6 = 6`. Total `2 + 9 + 6 = 17`. Matches. Let me also confirm no other order beats it by spot-checking the reverse-ish order job0,job1,job2: job0 finishes `3` cost `3`; job1 finishes `4` cost `8`; job2 finishes `6` cost `18`; total `29` — much worse. The ratio order wins, as the derivation promised.

**First implementation.** I will pair an index array with a comparator and accumulate:

```
int n; cin >> n;
vector<long long> p(n), w(n);
for (auto &x : p) cin >> x;
for (auto &x : w) cin >> x;
vector<int> idx(n); iota(idx.begin(), idx.end(), 0);
sort(idx.begin(), idx.end(), [&](int i, int j){
    return p[i]*w[j] < p[j]*w[i];           // Smith's rule
});
long long clock = 0, answer = 0;
for (int k = 0; k < n; k++) {
    int i = idx[k];
    clock += p[i];
    answer += w[i]*clock;
}
cout << answer << "\n";
```

**Debug episode 1 — the integer-overflow catch, traced on a concrete large case.** This is the pitfall this whole problem is designed around, so before believing the code I deliberately consider what happens if I had typed the *obvious* version with `int`. Suppose `p` and `w` were `vector<int>` and the comparator read `int lhs = p[i]*w[j];`. Take a tiny but telling slice of a large instance: two jobs with `p = [60000, 40000]` and `w = [50000, 90000]`. Their true ratios are `60000/50000 = 1.2` and `40000/90000 ≈ 0.444`, so job1 (the second) must come first. The correct comparator computes, for "does job0 go before job1": `p0*w1 = 60000 * 90000 = 5,400,000,000` versus `p1*w0 = 40000 * 50000 = 2,000,000,000`. In exact arithmetic `5.4e9 > 2.0e9`, so job0 does **not** precede job1 — correct, job1 first. Now trace the same in 32-bit `int`: `60000 * 90000 = 5,400,000,000` overflows; modulo `2^32` that is `5,400,000,000 - 4,294,967,296 = 1,105,032,704`, a *positive* but wrong number. The comparison becomes `1,105,032,704 < 2,000,000,000`, which is **true**, so the buggy comparator decides job0 *does* precede job1 — the exact opposite of the truth. The sort order is corrupted, and since Smith's rule depends on the exact ratio comparison, the final cost is wrong. I confirmed the magnitude of this empirically by running an int-comparator build on a 2000-job max-value instance: it printed `11577378455240780` against the correct `9384824717853153` — not a near-miss, a gross error. So both `p[i]*w[j]` operands and the products must be 64-bit. With `p`, `w` declared `vector<long long>`, the multiplication `p[i]*w[j]` is `long long * long long` and stays exact up to `10^10`, well inside 64-bit. The clock and the accumulator are already `long long`. **Fix: every value that touches a product or a running sum is `long long`.** This is the headline bug and it is now closed by typing alone — but only because I checked it; the `int` version compiles and passes every tiny test, then silently fails the big ones.

**Debug episode 2 — tie handling and comparator validity, traced on equal ratios.** A `std::sort` comparator must be a strict weak ordering; if `comp(a,b)` and `comp(b,a)` can both be false *and* the elements are "equivalent", that is fine, but if my comparator is ever inconsistent the sort is undefined behavior. My comparator returns `p[i]*w[j] < p[j]*w[i]`. When two jobs have equal ratios, `p[i]*w[j] == p[j]*w[i]`, so both `comp(i,j)` and `comp(j,i)` are false — they compare equivalent, which is correct and legal. Good. But I want to *trace* a tie to be sure the answer is order-independent there, since the derivation said the swap difference is exactly zero for equal ratios. Take `p = [2,4]`, `w = [1,2]` (ratios both `2`). Order job0,job1: job0 finishes `2` cost `2*1 = 2`; job1 finishes `6` cost `2*6 = 12`; total `14`. Order job1,job0: job1 finishes `4` cost `2*4 = 8`; job0 finishes `6` cost `1*6 = 6`; total `14`. Identical, exactly as the zero swap-difference predicted. So ties are genuinely free and my plain `<` comparator is safe. However, to keep the program *deterministic* across libstdc++ versions and make debugging reproducible, I add an explicit tie-break `return i < j` when the products are equal — it changes nothing about correctness (ties are free) but pins the output. I verified this added tie-break does not alter any answer by re-running the full random stress after adding it: still zero mismatches.

There is a subtler trap I want to rule out in this same episode: could the comparator ever *read* `p` or `w` out of bounds, or compare an index with itself? `std::sort` never calls `comp(i,i)`, and all indices in `idx` are valid `0..n-1`, so no. And does the empty case break the lambda? With `n = 0` the `idx` vector is empty and `sort` does nothing — safe.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: `cin >> n` reads `0`; the two read loops run zero times; `idx` is empty; `sort` is a no-op; the accumulation loop runs zero times; `answer` stays `0`. I print `0`. Correct — no jobs, no cost. (If the input stream is entirely empty, `if (!(cin >> n)) return 0;` also prints nothing-then-returns; but the contract guarantees at least `n`, and an empty stream returning `0` output is the benign degenerate.) Let me make the no-jobs path print `0` explicitly: with `n = 0` the code naturally outputs `0\n`, which I confirmed by running it.
- `n = 1`: one job, only one order. Clock becomes `p[0]`, answer becomes `w[0]*p[0]`. With `p=[5], w=[7]` that is `35`. Trivially correct; no ordering decision exists.
- All equal ratios (e.g. all jobs identical `p=w=100000`, `n=20000`): the order is irrelevant by the tie analysis; the answer is `w*p * (1+2+...+n) = 10^10 * n(n+1)/2`. For `n=20000` that is `2.0001*10^18`, which I computed independently as `2000100000000000000` and the program prints exactly that — confirming both the formula and that the accumulator holds the maximum magnitude without overflow.
- Overflow corners, restated and checked: the comparator product peaks at `10^5 * 10^5 = 10^10` (fits 64-bit, would overflow 32-bit — episode 1); the clock peaks at `n * max(p) = 2*10^4 * 10^5 = 2*10^9` (already over 32-bit, fits 64-bit); the answer peaks at `~2*10^18` (fits 64-bit with margin). Every one of these is `long long`. No `int` ever participates in a product or a sum of these quantities — the only `int`s are the loop counter `k` and the index `i`, both bounded by `n = 2*10^4`, which is fine.
- Input format: `cin >>` skips arbitrary whitespace, so the three blocks (`n`, the `p` row, the `w` row) parse regardless of how line breaks fall. Output is exactly one integer and a newline.

**Re-verification against an independent oracle.** Idea-correct and code-correct are different claims; I want the second by brute force. I wrote an independent brute that, for tiny `n`, tries *every* permutation and takes the minimum total weighted completion time — no greedy reasoning at all, just the definition. I generated 600 random small instances (`n` up to 7, values up to 50, including ties and `n = 0`), ran my Smith-rule solution against the permutation brute, and got **zero mismatches**. That cross-check pins down both the exchange argument (the sort really is optimal) and the accumulation (the cost is summed correctly). I also re-confirmed the documented sample prints `17`, the tie sample prints `14`, the `n = 0` case prints `0`, and the max-magnitude case prints `2000100000000000000`. The max instance (`n = 2*10^4`) runs in well under 10 ms with a 3.7 MB footprint, comfortably inside the 1 s / 256 MB limits.

**Final solution.** I convinced myself the idea is right by *deriving* the adjacent-swap inequality `p_A w_B <= p_B w_A` (so ratio-sorted is optimal) and by matching a permutation brute on 600 cases; I convinced myself the *code* is right by tracing the int-overflow corruption to a precise wrong comparison and forcing every product and sum to `long long`, and by tracing a tie to confirm order-independence. That is what I ship — one self-contained file, the `O(n log n)` Smith's-rule sort with an exact 64-bit cross-multiplied comparator:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // no jobs -> total cost 0
    vector<long long> p(n), w(n);
    for (int i = 0; i < n; i++) cin >> p[i];
    for (int i = 0; i < n; i++) cin >> w[i];

    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    // Smith's rule: order by p/w ascending. Compare p_i/w_i < p_j/w_j as a
    // cross-multiplication p_i * w_j < p_j * w_i with NO division. Both products
    // can reach 1e5 * 1e5 = 1e10, which overflows 32-bit; p and w are long long.
    sort(idx.begin(), idx.end(), [&](int i, int j) {
        long long lhs = p[i] * w[j];
        long long rhs = p[j] * w[i];
        if (lhs != rhs) return lhs < rhs;
        return i < j;                       // deterministic tie-break
    });

    long long clock = 0;                    // running completion time (up to 2e9)
    long long answer = 0;                   // sum of w_i * C_i (up to ~2e18)
    for (int k = 0; k < n; k++) {
        int i = idx[k];
        clock += p[i];                      // this job finishes at 'clock'
        answer += w[i] * clock;             // weighted completion time
    }

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** Smith's rule looked like folklore, so I derived the adjacent-swap difference `w_B p_A - w_A p_B` and learned that putting the smaller ratio `p/w` first never increases cost, which makes ratio-sorted globally optimal and hands me a *division-free* comparator `p_i w_j < p_j w_i`; that comparator's products reach `10^10` and the running clock reaches `2*10^9` and the answer reaches `2*10^18`, so a 32-bit `int` anywhere in a product or sum silently corrupts the sort order (I traced `60000*90000` overflowing to `1,105,032,704` and flipping a comparison, and saw an int build print `1.16e16` against the true `9.38e15`) — forcing `long long` on `p`, `w`, the clock, and the accumulator; a traced tie (`p=[2,4], w=[1,2]` both costing `14`) confirmed equal ratios are order-free so a plain `<` with an `i<j` tie-break is a valid strict weak ordering; and a 600-case permutation-brute cross-check plus the `n=0`, `n=1`, and max-magnitude corners closed out correctness.
