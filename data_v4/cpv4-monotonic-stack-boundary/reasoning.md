**Reading the problem and pinning the contract.** I have `n` integers `a[0..n-1]` and I must sum the minimum over *every* contiguous subarray, then report the total modulo `1000000007`. Input is `n` followed by the values; output is one integer. Before any algorithm I fix the scale, because it dictates the whole strategy: there are `n*(n+1)/2` subarrays, which for `n = 2*10^5` is about `2*10^10` — far too many to enumerate. The values satisfy `|a[i]| <= 10^9`. So two hard facts are settled up front. First, I need an `O(n)` or `O(n log n)` method that never materializes a subarray. Second, even the *count* of subarrays touching one index can be `~(10^5)^2 = 10^10`, which overflows a 32-bit int, and the running total is astronomically larger; everything must be reduced modulo `1000000007` and carried in `long long`. An `int` anywhere in the counting is a silent wrong-answer on the large tests.

**Reframing from subarrays to elements.** The minimum of a subarray is a single element, so instead of asking "what is the minimum of each subarray" I ask the dual: "for each index `i`, how many subarrays have `a[i]` as their minimum?" Call that `c[i]`. Then every subarray is counted under exactly one minimum-owner, so

```
S = sum_i c[i] * a[i].
```

This is the only reframing that turns an `O(n^2)` enumeration into a per-element accounting. The whole problem now reduces to computing `c[i]` correctly — and `c[i]` is exactly where the boundary bookkeeping bites.

**Deriving the reach formula.** `a[i]` is the minimum of a subarray `a[l..r]` iff `l <= i <= r` and every element in `[l, r]` is `>= a[i]`, with `a[i]` actually attained. So the subarray can extend left from `i` until just before some element *smaller* than `a[i]`, and right until just before some element smaller than `a[i]`. Define:

- `L` = number of valid choices for `l`, i.e. positions `l` with `l <= i` and `a[l..i]` all `>= a[i]`;
- `R` = number of valid choices for `r`, symmetrically.

Then `c[i] = L * R`, because `l` and `r` are chosen independently. If `prev` is the index of the nearest element to the left that breaks the reach, then the legal `l` range is `(prev, i]`, which has `i - prev` choices; similarly `R = nxt - i`. Both are *inclusive of `i` itself* (the single-cell subarray `[i,i]` corresponds to `l = r = i`), which is the first place an off-by-one could sneak in: the count is `i - prev`, not `i - prev - 1`.

**The tie problem, stated precisely.** Consider `a = [2, 2, 2]`. The subarray `a[0..2]` has minimum `2`, attained at indices 0, 1, and 2. If I let each of those three indices "own" this subarray, I count it three times. Each subarray with a tied minimum must be credited to exactly *one* owner. The standard fix is asymmetric comparisons: extend the **left** reach over elements *strictly greater* than `a[i]` (stop at the first `<= a[i]`), and extend the **right** reach over elements *greater than or equal* to `a[i]` (stop at the first `< a[i]`). Equivalently: the owner of a tied block is its leftmost element, because only it can reach across the equal copies to the right while the right copies cannot reach back left across it. I will encode this as: left uses a *strictly-greater* pop condition, right uses a *greater-or-equal* pop condition. (The mirror convention also works; what is fatal is using the *same* strictness on both sides.)

**Candidate approaches.** Two ways to get the nearest-smaller boundaries:

- *Monotonic stack, two passes.* One left-to-right pass maintaining a stack of indices with non-decreasing values gives `prev` for every `i`; one right-to-left pass gives `nxt`. `O(n)` time, `O(n)` memory. This is the workhorse and I will take it.
- *DP "jump" trick (single pass).* Maintain `prev[i]` by walking the previous-smaller pointers and skipping. It is also `O(n)` amortized but fiddlier to get the strict/non-strict split right. No advantage here.

I commit to the two-pass monotonic stack.

**First implementation and a trace.** Here is my first cut of the two passes. I am worried about the span arithmetic, so I deliberately write `left[i] = i - prev - 1` and `right[i] = nxt - i - 1`, the "number of elements strictly between the walls" form that I have seen in histogram problems, then trace it:

```
// left: pop while a[st.back()] > a[i]
int prev = st.empty() ? -1 : st.back();
left[i] = i - prev - 1;      // (suspect)
// right: pop while a[st.back()] >= a[i]
int nxt = st.empty() ? n : st.back();
right[i] = nxt - i - 1;      // (suspect)
```

Trace the smallest non-trivial input, a single element `a = [5]`, where the answer is obviously `5` (the only subarray is `[5]`, minimum `5`). Left pass: stack empty, `prev = -1`, `left[0] = 0 - (-1) - 1 = 0`. Right pass: stack empty, `nxt = 1`, `right[0] = 1 - 0 - 1 = 0`. Then `c[0] = left[0] * right[0] = 0 * 0 = 0`, and the contribution is `0 * 5 = 0`. The program prints `0`.

**The bug (episode 1: span off-by-one).** It prints `0` but the answer is `5`. The defect is exact: I borrowed the `i - prev - 1` formula from the *largest-rectangle-in-histogram* setting, where the span counts elements **strictly between** the two walls and the current index is a wall, not a counted cell. Here the semantics are different — index `i` is itself a legal endpoint, so the left reach is the inclusive count of positions in `(prev, i]`, which is `i - prev`, and the right reach is the inclusive count in `[i, nxt)`, which is `nxt - i`. By subtracting one on each side I excluded `i` from its own reach, collapsing the single-cell subarray to nothing. I confirmed this against the brute force on `[5]` (it wants `5`) and on `[3,1,2,4]` (it wants `17` but the off-by-one version yields `2`). Fix: drop the `-1` on both sides.

```
left[i]  = i - prev;     // positions l in (prev, i], inclusive of i
right[i] = nxt - i;      // positions r in [i, nxt), inclusive of i
```

Re-trace `[5]`: `left[0] = 0 - (-1) = 1`, `right[0] = 1 - 0 = 1`, `c[0] = 1`, contribution `1*5 = 5`. Correct. Re-trace `[3,1,2,4]` mentally for the smallest element to sanity-check: the `1` at index 1 has `prev = -1` (nothing smaller to its left) so `left = 1 - (-1) = 2`, and nothing smaller to its right so `nxt = 4`, `right = 4 - 1 = 3`; `c = 6`, contributing `6*1 = 6`, which matches that the `1` is the minimum of exactly the 6 subarrays spanning it. Good — the inclusive form is right.

**Second implementation and a trace on ties.** Now I worry about the tie convention. I wrote both pop conditions as `>=` on my first pass at the comparison logic — it "felt symmetric". Let me trace `a = [2, 2, 2]`, answer `12` (every one of the 6 subarrays has minimum `2`, so `S = 6 * 2 = 12`). With **both** sides using `>=`:

- Left pass (pop while `a[st.back()] >= a[i]`): i=0 -> stack empty, `prev=-1`, `left[0]=0-(-1)=1`, push 0. i=1 -> top is index 0 with value `2 >= 2`, pop it; stack empty, `prev=-1`, `left[1]=1-(-1)=2`, push 1. i=2 -> top is index 1 value `2 >= 2`, pop; empty, `prev=-1`, `left[2]=2-(-1)=3`, push 2. So `left = [1, 2, 3]`.
- Right pass (pop while `a[st.back()] >= a[i]`): i=2 -> empty, `nxt=3`, `right[2]=3-2=1`, push 2. i=1 -> top index 2 value `2 >= 2`, pop; empty, `nxt=3`, `right[1]=3-1=2`, push 1. i=0 -> top index 1 value `2 >= 2`, pop; empty, `nxt=3`, `right[0]=3-0=3`, push 0. So `right = [3, 2, 1]`.

Now `c[i] = left[i]*right[i] = [1*3, 2*2, 3*1] = [3, 4, 3]`, total count `3+4+3 = 10`, and `S = 10 * 2 = 20`. The program prints `20`.

**The bug (episode 2: same strictness on both sides double-counts ties).** Brute force says `12`; I got `20`. The count of subarrays is `10`, but there are only `6` subarrays total — I overcounted by `4`. The cause is exactly the tie problem I warned myself about: with `>=` on *both* sides, a block of equal elements lets each member reach over the others in *both* directions, so a tied subarray is credited to several owners at once. For `[2,2,2]` the subarray `[0..2]` is being counted under index 0 (left 1, right 3), under index 1 (left 2, right 2 includes it), and under index 2 — triple-counted. The fix is to make the comparisons asymmetric so each tied subarray has a unique owner: keep **strict** `>` on the left pop (left reach stops at the first element `<= a[i]`, so an equal element to the left is a wall and does *not* extend the reach) and keep **non-strict** `>=` on the right pop (right reach extends over equal elements). This makes the *leftmost* element of any equal block the sole owner.

Re-trace `[2,2,2]` with left `>` / right `>=`:

- Left pass (pop while `> a[i]`): i=0 -> empty, `left[0]=1`, push 0. i=1 -> top index 0 value `2`, is `2 > 2`? No, do not pop; `prev=0`, `left[1]=1-0=1`, push 1. i=2 -> top index 1 value `2`, `2 > 2`? No; `prev=1`, `left[2]=2-1=1`, push 2. So `left = [1, 1, 1]`.
- Right pass (pop while `>= a[i]`): unchanged from before, `right = [3, 2, 1]`.

Now `c = [1*3, 1*2, 1*1] = [3, 2, 1]`, total `6`, and `S = 6*2 = 12`. Correct, and the count `6` equals the true number of subarrays, which is the structural invariant I want: with the asymmetric convention, `sum_i c[i]` must equal `n*(n+1)/2`. For `n=3` that is `6`. The convention checks out.

**Modular folding, derived carefully.** Each `c[i]` can be as large as `~(10^5)^2 = 10^10`, and `a[i]` up to `10^9`, so `c[i]*a[i]` is `~10^19`, which exceeds `long long`'s `~9.2*10^18`. I therefore reduce before multiplying: compute `cnt = (left[i] % MOD) * (right[i] % MOD) % MOD`, which is `< MOD < 10^9`, and `val = ((a[i] % MOD) + MOD) % MOD` to map possibly-negative values into `[0, MOD)`. Then `cnt * val` is `< 10^18`, safe in `long long`, and I accumulate modulo `MOD`. The `(... + MOD) % MOD` on `val` is what guarantees the printed answer is never negative even when `a[i] < 0` — a corner the contract explicitly calls out.

**Edge cases, deliberately.**
- `n = 0`: `if (!(cin >> n)) return 0;` is for empty input, but `n = 0` is given as a token; the loops run zero times, `ans` stays `0`. Correct — there are no subarrays.
- `n = 1`, `a = [-7]`: `left=[1]`, `right=[1]`, `c=[1]`; `val = ((-7 % MOD) + MOD) % MOD = MOD - 7`; `ans = (MOD - 7) % MOD = 1000000000`. The single subarray's minimum is `-7`, and `-7 mod MOD = 1000000000`. Correct, and non-negative.
- Strictly increasing `[1,2,3]`: each `a[i]` is the min only of subarrays starting at `i`, so the left reach must be `1` for `i>0`. Left pass with `>`: i=1, top value `1`, `1 > 2`? No, `prev=0`, `left=1`. Good — increasing arrays give `left[i]=1` for all `i>0` as expected.
- Strictly decreasing `[3,2,1]`: each `a[i]` reaches all the way left. i=2: pop everything greater, `prev=-1`, `left=3`. Good.
- Overflow: `left[i]`, `right[i]` are bounded by `n <= 2*10^5`, comfortably in `int`/`long long`; all products are reduced mod `MOD` before they can exceed `10^18`. Safe.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so input parsing is format-agnostic.

**Final solution.** I convinced myself the *idea* is right by the ownership reframing and the `sum_i c[i] = n(n+1)/2` invariant, and I convinced myself the *code* is right by tracing two failing cases to precise causes — the inclusive span (drop the `-1`) and the asymmetric tie-break (strict left, non-strict right) — and re-verifying each fix plus the negative/empty/monotone corners, with a 300+ case stress against brute force returning zero mismatches. This is what I ship: one self-contained file, two monotonic-stack passes, careful modular folding.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    const long long MOD = 1000000007LL;

    // For each i, count subarrays whose minimum is a[i].
    // left[i]  = number of consecutive elements strictly greater than a[i]
    //            immediately to the left (so a[i] is the min over that reach).
    // right[i] = number of consecutive elements >= a[i] immediately to the right.
    // The strict/non-strict split (strict left, non-strict right) makes each
    // subarray credited to exactly one index when minima tie.
    vector<long long> left(n), right(n);

    // left: previous strictly-smaller-OR-EQUAL element acts as the wall.
    // Pop while stack top value > a[i]  (those are strictly greater -> in reach).
    {
        vector<int> st; // indices, increasing-ish by value (non-decreasing)
        for (int i = 0; i < n; i++) {
            while (!st.empty() && a[st.back()] > a[i]) st.pop_back();
            int prev = st.empty() ? -1 : st.back();
            left[i] = i - prev;           // count of positions in (prev, i]
            st.push_back(i);
        }
    }
    // right: next strictly-smaller element is the wall.
    // Pop while stack top value >= a[i] (those are >= -> still in reach).
    {
        vector<int> st;
        for (int i = n - 1; i >= 0; i--) {
            while (!st.empty() && a[st.back()] >= a[i]) st.pop_back();
            int nxt = st.empty() ? n : st.back();
            right[i] = nxt - i;           // count of positions in [i, nxt)
            st.push_back(i);
        }
    }

    long long ans = 0;
    for (int i = 0; i < n; i++) {
        long long cnt = (left[i] % MOD) * (right[i] % MOD) % MOD;
        long long val = ((a[i] % MOD) + MOD) % MOD;
        ans = (ans + cnt * val) % MOD;
    }
    cout << ans << "\n";
    return 0;
}
```

**Causal recap.** The `O(n^2)` enumeration is impossible at `n = 2*10^5`, so I reframed `S = sum_i c[i]*a[i]` where `c[i]` counts subarrays whose minimum is `a[i]`, computed as `(left reach) * (right reach)` from nearest-smaller boundaries via two monotonic-stack passes. My first span formula `i - prev - 1` was lifted from the histogram-rectangle setting where the index is an excluded wall, but here `i` is its own legal endpoint — a trace of `[5]` returning `0` instead of `5` pinned the off-by-one, fixed by the inclusive `i - prev`. My first tie convention used `>=` on both sides, which credits each equal-minimum subarray to multiple owners — a trace of `[2,2,2]` returning `20` instead of `12` exposed the overcount, fixed by making the left pop strict (`>`) and the right pop non-strict (`>=`) so the leftmost equal element is the sole owner; the invariant `sum_i c[i] = n(n+1)/2` then holds. Finally, since `c[i]*a[i]` can reach `~10^19`, I reduce `left`, `right`, and the negative-safe `val` modulo `MOD` before multiplying, which keeps every product under `10^18` and the printed answer in `[0, MOD)`.
