**Reading the problem and pinning the contract.** I have an array `a[0..n-1]` of positive integers. Every subarray `a[l..r]` has a minimum, and I assign that subarray to a single *owner*: the index of the minimum, with ties broken to the **leftmost** position achieving the minimum. Let `c[i]` be the number of subarrays owned by `i`. I must output `S = (sum_i i*c[i]) mod (10^9+7)`, with `i` 0-based. Before any algorithm I fix the scale, because it dictates types and where the modulus has to live: `n <= 2*10^5`, so the total number of subarrays is `n(n+1)/2 ~ 2*10^10` — already past 32-bit. A single `c[i]` can be as large as `n(n+1)/2` too (one index can own almost everything). And `i*c[i]` with `i` up to `2*10^5` and `c[i]` up to `~2*10^10` is `~4*10^15`, which still fits in `long long` (`~9.2*10^18`) but absolutely not in `int`. So: read into 64-bit, form counts in 64-bit, and reduce modulo `10^9+7` when accumulating `S`. The modulus is on the *final weighted sum*, not on the raw counts I use to build geometry. That distinction is going to matter.

**A global invariant I can lean on.** Every subarray has exactly one owner, so `sum_i c[i] = n(n+1)/2` no matter what the array looks like. This is a free correctness oracle: after I compute all `c[i]`, their sum must equal `n(n+1)/2`. I will use this both on paper and as a mental assertion. If my `<`/`<=` convention double-counts, the sum overshoots; if it drops subarrays, the sum undershoots. That makes the classic monotonic-stack pitfall *visible* rather than silent.

**Candidate approaches.** Two routes.

- *Brute force.* For each left endpoint `l`, sweep `r` rightward, maintain the running minimum and the index of its leftmost occurrence, and increment that owner each step. `O(n^2)`. Obviously correct — it literally implements the definition, including the leftmost tie-break (only move the owner when I see a *strictly* smaller value). At `n = 2*10^5` this is `~2*10^10` steps, far too slow, but it is my reference oracle.
- *Monotonic stack.* For a fixed `i`, the subarrays for which `a[i]` is the chosen (leftmost) minimum form a rectangle: `l` ranges over some `(L, i]` and `r` over some `[i, R)`, so `c[i] = (i - L)*(R - i)`. The barriers `L` and `R` come from nearest-smaller-element scans. `O(n)`. The whole risk is the exact `<` vs `<=` asymmetry that encodes "leftmost wins ties."

I commit to the stack for speed but I am going to *derive* the asymmetry rather than guess it, then check it against brute on equal-heavy inputs.

**Deriving the barriers — and being very careful about ties.** Fix `i`. For `a[i]` to be the owner of `[l, r]` (with `l <= i <= r`) under "leftmost minimum wins," I need three conditions:

1. No element strictly less than `a[i]` anywhere in `[l, r]`. Otherwise that element is a smaller value and `a[i]` is not even a minimum.
2. No element *equal to* `a[i]` strictly to the **left** of `i` inside `[l, r]`. An equal element at some `j < i` is also a minimum and is more-left, so the leftmost-minimum rule hands ownership to `j`, not `i`.
3. Elements equal to `a[i]` to the **right** of `i` are fine — `i` is still the leftmost occurrence of the minimum, so `i` keeps ownership.

Translate to barriers. Going left from `i`, I must stop as soon as I would include an element `<= a[i]` (condition 1 forbids `<`, condition 2 forbids `=`). So the left barrier `L` is the nearest index `j < i` with `a[j] <= a[i]`, and `l` may be anything in `(L, i]`, giving `i - L` choices. Going right from `i`, I must stop only at an element `< a[i]` (condition 1), while equal elements are allowed (condition 3). So the right barrier `R` is the nearest index `j > i` with `a[j] < a[i]`, and `r` ranges over `[i, R)`, giving `R - i` choices. If no such barrier exists, `L = -1` or `R = n`.

So the convention is **asymmetric**: `<=` on the left, `<` on the right. That is exactly the convention that breaks the symmetry of equal elements so each shared subarray is counted by precisely one (the leftmost) owner. `c[i] = (i - L)*(R - i)`.

**Sanity-checking the derivation on the sample.** Take `a = [2, 1, 2, 1, 3]`, `n = 5`. Let me compute `L` (nearest `<=` on the left) and `R` (nearest `<` on the right) per index.

- i=0 (a=2): left none -> L=-1. Right: nearest `<2` is index 1 (a=1) -> R=1. c = (0-(-1))*(1-0) = 1*1 = 1.
- i=1 (a=1): left nearest `<=1`: index 0 has 2 (no), nothing -> L=-1. Right nearest `<1`: none (no value below 1) -> R=5. c = (1-(-1))*(5-1) = 2*4 = 8.
- i=2 (a=2): left nearest `<=2`: index 1 (a=1<=2) -> L=1. Right nearest `<2`: index 3 (a=1) -> R=3. c = (2-1)*(3-2) = 1*1 = 1.
- i=3 (a=1): left nearest `<=1`: index 1 (a=1<=1, yes!) -> L=1. Right nearest `<1`: none -> R=5. c = (3-1)*(5-3) = 2*2 = 4.
- i=4 (a=3): left nearest `<=3`: index 3 (a=1) -> L=3. Right none -> R=5. c = (4-3)*(5-4) = 1*1 = 1.

So `c = [1, 8, 1, 4, 1]`. Sum = 15 = 5*6/2. The invariant holds — that is strong evidence the `<=`/`<` split is right. And note the two `1`s at indices 1 and 3: index 1 (leftmost) takes 8, index 3 takes 4; the `<=`-on-left at i=3 correctly *stops* at the equal element at index 1 so the subarrays spanning both ones are credited to index 1, not double-counted. Finally `S = 0*1 + 1*8 + 2*1 + 3*4 + 4*1 = 0 + 8 + 2 + 12 + 4 = 26`. That is the documented answer.

**First implementation and a trace.** I write the two stack passes. My first attempt — and I want to be honest about how I first typed it, because the tie-break is exactly the thing I am prone to flip:

```
// left[i]: previous index with a[j] < a[i]   (FIRST TRY: strict)
for (int i = 0; i < n; i++) {
    while (!st.empty() && a[st.back()] >= a[i]) st.pop_back();
    left_b[i] = st.empty() ? -1 : st.back();
    st.push_back(i);
}
st.clear();
// right[i]: next index with a[j] < a[i]
for (int i = n - 1; i >= 0; i--) {
    while (!st.empty() && a[st.back()] >= a[i]) st.pop_back();
    right_b[i] = st.empty() ? n : st.back();
    st.push_back(i);
}
```

Here I made both passes pop on `>=`, i.e. the left barrier is the nearest *strictly smaller* element (`<`), and so is the right. That is the symmetric `<`/`<` convention. Let me trace it on the most tie-dense input I can think of: `a = [2, 2]`, `n = 2`. By hand the truth is: subarrays are `[0,0]` (min at 0), `[1,1]` (min at 1), `[0,1]` (min value 2, leftmost is index 0). So `c = [2, 1]`, sum 3 = 2*3/2. Now run my code.

Left pass, popping on `>=`: i=0, stack empty -> L[0]=-1, push 0. i=1, top is index 0 with a=2, `2 >= 2` true so pop it; stack empty -> L[1]=-1, push 1. So `L = [-1, -1]`.
Right pass, popping on `>=`: i=1, empty -> R[1]=2, push 1. i=0, top index 1 a=2, `2>=2` pop; empty -> R[0]=2, push 0. So `R = [2, 2]`.
Counts: c[0] = (0-(-1))*(2-0) = 1*2 = 2. c[1] = (1-(-1))*(2-1) = 2*1 = 2. Sum = 4.

**The bug.** The sum is `4`, but it must be `3 = n(n+1)/2`. I over-counted by one. The culprit is the subarray `[0,1]`: index 0 claims it (its rectangle extends right through the equal element at 1 because the right pass treated equals as not-a-barrier — `R[0]=2`), and index 1 *also* claims it (its rectangle extends left through the equal element at 0 because the left pass also treated equals as not-a-barrier — `L[1]=-1`). Both sides being strict (`<`) means an equal element blocks *neither* direction, so the subarray spanning two equal minima is counted twice. The leftmost-minimum rule says it must be counted once, by index 0 only. My derivation already told me this: the left barrier must use `<=`, not `<`. The left pass should *stop at* (not pop) an equal element, so that index 1's leftward rectangle is bounded by index 0 and does not reclaim `[0,1]`.

**Fix and re-verification.** Make the left pass pop only on `>` (so it stops at `<=`, i.e. the barrier is the nearest `a[j] <= a[i]`), and leave the right pass popping on `>=` (barrier is nearest `a[j] < a[i]`):

```
// left[i]: nearest j<i with a[j] <= a[i]  -> pop while a[top] > a[i]
while (!st.empty() && a[st.back()] > a[i]) st.pop_back();
// right[i]: nearest j>i with a[j] <  a[i]  -> pop while a[top] >= a[i]
while (!st.empty() && a[st.back()] >= a[i]) st.pop_back();
```

Re-trace `[2, 2]`. Left pass, pop on `>`: i=0 -> L[0]=-1, push 0. i=1: top index 0 a=2, `2 > 2` is false, so do NOT pop; L[1]=0, push 1. So `L = [-1, 0]`. Right pass, pop on `>=`: i=1 -> R[1]=2, push 1. i=0: top index 1 a=2, `2>=2` true pop; empty -> R[0]=2. So `R = [2, 2]`. Counts: c[0] = (0-(-1))*(2-0) = 2. c[1] = (1-0)*(2-1) = 1. Sum = 3. Correct, and `[0,1]` is now owned only by index 0. The over-count is gone, for exactly the reason I diagnosed.

Let me also re-trace the sample `[2,1,2,1,3]` through the fixed passes to be sure I did not break the distinct-value case. Left pass (pop on `>`):
- i=0 a=2: empty -> L=-1, push0. stack idx:[0].
- i=1 a=1: top0 a=2, 2>1 pop; empty -> L=-1, push1. stack:[1].
- i=2 a=2: top1 a=1, 1>2 false -> L=1, push2. stack:[1,2].
- i=3 a=1: top2 a=2, 2>1 pop; top1 a=1, 1>1 false -> L=1, push3. stack:[1,3].
- i=4 a=3: top3 a=1, 1>3 false -> L=3, push4.
`L = [-1,-1,1,1,3]`. Right pass (pop on `>=`), i from 4 down:
- i=4 a=3: empty -> R=5, push4. stack:[4].
- i=3 a=1: top4 a=3, 3>=1 pop; empty -> R=5, push3. stack:[3].
- i=2 a=2: top3 a=1, 1>=2 false -> R=3, push2. stack:[3,2].
- i=1 a=1: top2 a=2, 2>=1 pop; top3 a=1, 1>=1 pop; empty -> R=5, push1. stack:[1].
- i=0 a=2: top1 a=1, 1>=2 false -> R=1, push0.
`R = [1,5,3,5,5]`. Counts: c0=(0+1)*(1-0)=1; c1=(1+1)*(5-1)=8; c2=(2-1)*(3-2)=1; c3=(3-1)*(5-3)=4; c4=(4-3)*(5-4)=1. `c=[1,8,1,4,1]`, matching my paper derivation exactly. Good — the fix preserves the distinct case and repairs the tie case.

**A second, quieter bug — the weighting and the modulus.** With the geometry correct I turn to forming `S = sum_i i*c[i] mod p`. My first accumulation looked like:

```
int ans = 0;
for (int i = 0; i < n; i++) {
    long long cnt = (long long)(i - left_b[i]) * (right_b[i] - i);
    ans = (ans + i * cnt) % MOD;     // BUG: i*cnt and ans typed/placed wrong
}
```

Two defects hide here. First, `ans` is `int`; `% MOD` keeps it below `10^9+7` which is itself larger than `INT_MAX = 2.1*10^9`? No — `10^9+7 < 2.15*10^9`, so a value `< MOD` fits an `int` barely, but `ans + i*cnt` is evaluated *before* the `%`, and `i*cnt` can be `~2*10^5 * 2*10^10 = 4*10^15`, which overflows `int` long before the modulo runs. Even the multiplication `i * cnt`: `i` is `int`, `cnt` is `long long`, so that product is `long long` (fine), but adding to an `int` `ans` and storing back truncates. Let me trace a concrete overflow case to make the failure undeniable: a strictly increasing array `a = [1,2,3]`. Then every `i` owns `[i..r]` for all `r`, so `c[i] = n - i = [3,2,1]`, and `S = 0*3 + 1*2 + 2*1 = 4`. Tiny, no overflow — bad witness. I need large magnitudes. Take the increasing array of length `2*10^5`: `c[i] = n - i`, and `S = sum_i i*(n-i)`. With `n = 2*10^5` the dominant term is around `(n/2)*(n/2)*n ~ 10^5*10^5*2*10^5 = 2*10^15` summed... the true `S` before modulo is on the order of `n^3/6 ~ 1.3*10^15`. An `int` `ans` cannot hold partial sums that size; it wraps and gives garbage, while a `long long` accumulator reduced each step stays correct. So `ans` must be `long long`, and I should reduce `i*cnt` mod p *every* iteration, never letting an unreduced running sum grow.

Second defect: even the *count* itself, `(i-left)*(right-i)`, can reach `~2*10^10` and must be formed in 64-bit (it is, via the cast) and then reduced mod p before multiplying by `i`, otherwise `i*cnt` could reach `2*10^5 * 2*10^10 = 4*10^15` — that fits in `long long`, so it is safe even without pre-reducing `cnt`, but to be defensive and uniform I reduce `cnt` mod p first, then it is `< 10^9`, and `i % MOD` is `< 10^9`, so the product is `< 10^18`, comfortably inside `long long`. Fixed accumulation:

```
long long ans = 0;
for (int i = 0; i < n; i++) {
    long long cnt = (long long)(i - left_b[i]) * (long long)(right_b[i] - i);
    cnt %= MOD;
    ans = (ans + (long long)(i % MOD) * cnt) % MOD;
}
```

Re-trace the increasing `[1,2,3]`: c=[3,2,1]; ans = 0 + (0*3) + (1*2) + (2*1) = 4 mod p = 4. And the large increasing case now stays in range because `ans` is reduced every step. The off-by-one trap of weighting by a 1-based index instead of 0-based is also worth naming: the spec says `i` is 0-based, so index 0 contributes `0`. If I had looped `for i in 1..n` or used `(i+1)*c[i]` I would silently shift the whole answer; my sample check `S=26` with the `0*1` term being zero is exactly what pins the 0-based convention.

**Edge cases, deliberately.**
- `n = 0`: the loop never runs, `ans = 0`. Output `0`. (`n(n+1)/2 = 0` subarrays — correct.)
- `n = 1`, `a = [7]`: one subarray `[0,0]` owned by index 0; `S = 0 * 1 = 0`. Trace: L[0]=-1, R[0]=1, c0=(0+1)*(1-0)=1, S = 0*1 = 0. Output `0` — correct, the single element sits at index 0 so it contributes nothing to a position-weighted sum.
- All equal, `a = [5,5,5,5,5,5]`: only the leftmost index 0 can extend its left rectangle to `L=-1`; every other index `i` has `L = i-1` (the equal element immediately left is a `<=` barrier), and `R = n` for all (no value below 5). So `c[i] = (i-(i-1))*(n-i) = n - i` for `i>=1`, and `c[0] = (0+1)*(n-0) = n`. Sum = `n + sum_{i>=1}(n-i) = n + (n-1)+(n-2)+...+1 = n + n(n-1)/2 = n(n+1)/2`. Invariant holds. For `n=6` that is `21`, and `S = 0*6 + 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 0+5+8+9+8+5 = 35`. My code prints `35` on this input — matches the brute force.
- Strictly decreasing, `a = [3,2,1]`: each `a[i]` is smaller than everything to its left, so `L[i] = i-1` for `i>=1`? No: left barrier is nearest `<= a[i]` on the left; with strictly decreasing values everything to the left is *larger*, so there is no `<=` barrier and `L[i] = -1` for all `i`. Right barrier nearest `< a[i]` is `i+1`. So `c[i] = (i+1)*(1) = i+1`, sum = `1+2+3 = 6 = 3*4/2`. Correct.
- Overflow: counts formed in 64-bit, reduced mod p; `ans` is 64-bit reduced each step. The intermediate `(long long)(i%MOD)*cnt < 10^18 < 9.2*10^18`. Safe.
- Output: exactly one integer and newline; `cin >>` skips arbitrary whitespace so parsing is format-agnostic.

**Final solution.** I disproved the symmetric `<`/`<` convention by tracing `[2,2]` to an over-count of exactly the shared subarray, fixed it with the asymmetric `<=`-left / `<`-right barriers that encode "leftmost minimum wins," and separately caught the `int` accumulator / 0-based-weighting trap by reasoning about magnitudes on the large increasing array. The `sum_i c[i] = n(n+1)/2` invariant gated every step. This is what I ship — one self-contained `O(n)` file:

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // c[i] = number of subarrays whose minimum is a[i], with ties broken by the
    // LEFTMOST minimum index. a[i] owns [l, r] (l <= i <= r) iff:
    //   - no element < a[i] in [l, r]            (else that element's value is the min)
    //   - no element <= a[i] strictly left of i  (an equal element to the left is a
    //                                              more-left minimum => it owns instead)
    //   - elements equal to a[i] to the RIGHT are fine (i stays leftmost min).
    // So:  left[i]  = nearest j < i with a[j] <= a[i]   -> l in (left[i], i]
    //      right[i] = nearest j > i with a[j] <  a[i]   -> r in [i, right[i])
    //      c[i] = (i - left[i]) * (right[i] - i).
    // Mixing <= / < across the two sides is what prevents double counting equal mins.

    vector<int> left_b(n), right_b(n);
    vector<int> st;
    st.reserve(n);

    // left[i]: previous index with a[j] <= a[i]; pop while a[top] > a[i].
    for (int i = 0; i < n; i++) {
        while (!st.empty() && a[st.back()] > a[i]) st.pop_back();
        left_b[i] = st.empty() ? -1 : st.back();
        st.push_back(i);
    }
    st.clear();
    // right[i]: next index with a[j] < a[i]; pop while a[top] >= a[i].
    for (int i = n - 1; i >= 0; i--) {
        while (!st.empty() && a[st.back()] >= a[i]) st.pop_back();
        right_b[i] = st.empty() ? n : st.back();
        st.push_back(i);
    }

    long long ans = 0;
    for (int i = 0; i < n; i++) {
        long long cnt = (long long)(i - left_b[i]) * (long long)(right_b[i] - i); // exact, fits in 64-bit
        cnt %= MOD;
        ans = (ans + (long long)(i % MOD) * cnt) % MOD;
    }

    cout << ans << "\n";
    return 0;
}
```

**Causal recap.** The symmetric `<`/`<` nearest-smaller convention looked natural but a trace of `[2,2]` returned `4` against the forced invariant `n(n+1)/2 = 3`, exposing that an equal element blocking neither side lets the shared subarray be claimed by both equal minima; the leftmost-minimum rule and my three-condition derivation demand the asymmetric `<=`-on-the-left / `<`-on-the-right barriers, which credit each tie to exactly one owner, restoring the invariant; with the geometry correct, the remaining trap was numeric — counts up to `~2*10^10` and a position-weighted sum up to `~10^15` overflow an `int` accumulator, so counts are built in 64-bit and the running sum is reduced mod `10^9+7` every step, while the 0-based weighting (index 0 contributes `0`) is pinned by the sample `S = 26`.
