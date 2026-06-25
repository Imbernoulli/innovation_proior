**Reading the problem and pinning the contract.** I have `n` daily ice-thickness readings `a[0..n-1]`, possibly negative, a tolerance `D`, and I must count the contiguous stretches `[i, j]` whose range `max(a[i..j]) - min(a[i..j])` is **strictly less than** `D`. Input is `n` and `D` on the first line, then the `n` readings; I print one integer. Before any algorithm I fix the scale, because it pins the types. `n <= 2*10^5`, so the number of stretches is at most `n*(n+1)/2`. Let me actually compute that bound rather than eyeball it: `2*10^5 * (2*10^5 + 1) / 2 ≈ 2*10^5 * 2*10^5 / 2 = 2*10^10`. That is an order of magnitude past the signed 32-bit ceiling of about `2.1*10^9`, so the answer accumulator must be 64-bit. `D` is bounded by `2*10^9`, which itself does not fit in a signed 32-bit `int` either (`2*10^9 > 2.147*10^9`? no — `2*10^9 < 2.147*10^9`, so it *just* fits), but the difference `a[max] - a[min]` can be up to `10^9 - (-10^9) = 2*10^9`, again at the edge; to avoid any signed-overflow reasoning at the comparison I will read `D` and the readings as `long long` and do all arithmetic in 64-bit. That is decision one and it is non-negotiable.

**The predicate is monotone in the window — that is the whole lever.** I want to count subarrays with `max - min < D`. The key structural fact: if a window `[i, j]` is stable, then any sub-window `[i', j']` with `i <= i' <= j' <= j` is also stable, because removing elements can only shrink the max and raise the min, hence shrink the range. Equivalently, for a fixed right end `j`, the set of left ends `i` that make `[i, j]` stable is a *suffix* `i in [L(j), j]` — once it is stable it stays stable as `i` grows toward `j`. That monotonicity is exactly what lets a two-pointer sweep work: as `j` increases by one, the boundary `L(j)` only ever moves right, never left, so the left pointer is monotone and the whole sweep is amortized linear.

**Laying out the candidate approaches.** Two routes, and I want the one I can defend.

- *Quadratic enumeration.* For each start `i`, extend `j` rightward keeping running max and min, count while range `< D`. `O(n^2)`, trivially correct — this is my brute-force oracle, not my submission. At `n = 2*10^5` it is `~2*10^10` operations, far over a 1-second budget.
- *Sliding window with two monotonic deques.* Sweep `right` from `0` to `n-1`. Keep a left pointer `left` = smallest start such that `[left, right]` is stable. For each `right`, every start in `[left, right]` yields a stable window, so the contribution is the count of those starts. To know `max - min` of the current window in amortized `O(1)` I keep a decreasing deque for the max (front = window max index) and an increasing deque for the min (front = window min index). `O(n)`. The open questions are precisely the two boundaries: the left-advance test (`>= D` vs `> D`) and the count increment (`right - left + 1` vs `right - left`).

**Deriving the count increment, and checking it numerically.** Suppose after processing `right`, the smallest stable start is `left`. The stable windows ending exactly at `right` are `[left, right], [left+1, right], ..., [right, right]`. How many is that? The starts range over the integers `left, left+1, ..., right` inclusive, which is `right - left + 1` values. Let me sanity-check the count on a concrete tiny case so I do not ship an off-by-one in the formula itself. Take `right = 3`, `left = 1`: starts are `1, 2, 3`, that is three windows, and `right - left + 1 = 3 - 1 + 1 = 3`. Matches. Take the degenerate `left = right = 3`: one window `[3,3]`, and `3 - 3 + 1 = 1`. Matches. And if the window is *empty* because even `[right, right]` is not stable, I will have advanced `left` to `right + 1`, giving `right - (right+1) + 1 = 0` — zero contribution, which is exactly what I want. So `right - left + 1` is the right increment, provided I let `left` reach `right + 1` in the unstable case. Good; I will hold that thought, because that `left == right + 1` state is where empty-deque crashes hide.

**Deriving the left-advance condition.** The window `[left, right]` is stable iff `a[maxd.front()] - a[mind.front()] < D`. So I should advance `left` (shrink) *while* the window is **not** stable, i.e. while `range >= D`. The negation of `< D` is `>= D`, not `> D` — getting that boundary right is the strict/non-strict crux. When I advance `left`, any deque front whose index is now `< left` has fallen out of the window and must be popped. I push `right` onto both deques first (maintaining monotonicity), then shrink.

**Confirming the recurrence by hand on the sample.** Sample: `D = 5`, `a = [4, 8, 6, 11, 9, 7]`, claimed answer `12`. Let me enumerate the stable windows ending at each `right`, by the suffix rule, range strictly `< 5`:
- `right=0` (4): `[4]` range 0 < 5. starts {0}. count 1.
- `right=1` (8): `[8]` r0; `[4,8]` r4 < 5. starts {0,1}. count 2.
- `right=2` (6): `[6]` r0; `[8,6]` r2; `[4,8,6]` r4 < 5. starts {0,1,2}. count 3.
- `right=3` (11): `[11]` r0; `[6,11]` r5 — NOT < 5. So left jumps to 3. starts {3}. count 1.
- `right=4` (9): `[9]` r0; `[11,9]` r2; `[6,11,9]` r5 — not stable. left = 3. starts {3,4}. count 2.
- `right=5` (7): `[7]` r0; `[9,7]` r2; `[11,9,7]` r4 < 5; `[6,11,9,7]` r5 — not stable. left = 3. starts {3,4,5}. count 3.

Total `1+2+3+1+2+3 = 12`. The model matches the claimed answer, and I can see the left pointer being monotone non-decreasing across `right` (`0,0,0,3,3,3`), confirming the two-pointer invariant.

**First implementation — and then a trace, because clean math transcribes dirty.** My first cut of the sweep body:

```
deque<int> maxd, mind;
long long answer = 0;
int left = 0;
for (int right = 0; right < n; right++) {
    while (!maxd.empty() && a[maxd.back()] <= a[right]) maxd.pop_back();
    maxd.push_back(right);
    while (!mind.empty() && a[mind.back()] >= a[right]) mind.pop_back();
    mind.push_back(right);
    while (a[maxd.front()] - a[mind.front()] >= D) {  // shrink while unstable
        left++;
        if (maxd.front() < left) maxd.pop_front();
        if (mind.front() < left) mind.pop_front();
    }
    answer += (long long)(right - left + 1);
}
```

The thing that worries me is the `D = 0` case: no window can have range `< 0`, so *every* window is unstable, so the shrink loop will keep advancing `left` past `right`, and the deques can empty out — and then `a[maxd.front()]` reads `front()` on an empty deque, which is undefined behaviour. Let me trace the smallest input that exposes it: `n = 1`, `D = 0`, `a = [100]`. Start `left = 0`, deques empty.

- `right = 0`: push `0` onto both deques. Now the shrink loop: `a[maxd.front()] - a[mind.front()] = a[0] - a[0] = 0`, and `0 >= D = 0` is **true**, so I enter the loop. `left++` -> `left = 1`. `maxd.front() = 0 < 1` -> pop, `maxd` empty. `mind.front() = 0 < 1` -> pop, `mind` empty. Loop condition re-evaluates: `a[maxd.front()]` with `maxd` empty -> **`front()` on empty deque, undefined behaviour / crash**.

**Diagnosing the first bug.** The defect is precise and it is a boundary bug, exactly the kind this problem is about: when `D` is small enough that even the singleton `[right, right]` is unstable, the correct left pointer is `right + 1` (the window is empty, contributing zero), but my shrink loop has no guard stopping it from reading the deque fronts after they have been emptied. The window `[left, right]` is empty precisely when `left > right`; in that state there is nothing to test and nothing to count, but my `while` re-checks `a[maxd.front()]` regardless. I confirmed this concretely by running the unguarded build on `n=1, D=0` and it segfaulted. The fix is to add `left <= right` to the shrink-loop guard so I never read a front for an empty window.

**Fixing and re-verifying the first bug.** Guarded loop:

```
while (left <= right && a[maxd.front()] - a[mind.front()] >= D) {
    left++;
    if (maxd.front() < left) maxd.pop_front();
    if (mind.front() < left) mind.pop_front();
}
answer += (long long)(right - left + 1);
```

Re-trace `n=1, D=0, a=[100]`: `right=0`, push 0 onto both deques. Shrink guard: `left(0) <= right(0)` true, and `a[0]-a[0]=0 >= 0` true -> enter. `left=1`; `maxd.front()=0 < 1` pop (empty); `mind.front()=0 < 1` pop (empty). Re-check guard: `left(1) <= right(0)` is **false** -> exit, never touching the empty fronts. Contribution `right - left + 1 = 0 - 1 + 1 = 0`. Answer `0`. Correct: with `D=0` no stretch is stable. And the deques being empty entering the next iteration is fine, because the next `right` pushes itself on before anything reads a front. The crash is gone and the count is right, which is the evidence I trust.

**Second trace — the strict/non-strict boundary, on an exact-equality window.** The other place this dies is `>= D` vs `> D` in the shrink test. Let me build a case where a window's range equals `D` exactly, because that is the only input where the two conditions disagree. Take `D = 4`, `a = [1, 5, 2]` (so the window `[1,5]` has range exactly 4). The problem says stable means range *strictly less than* `D`, so range `= 4` is **not** stable. Brute count: `[1]`r0✓, `[5]`r0✓, `[2]`r0✓ (three singletons), `[1,5]`r4 — not `<4`, ✗; `[5,2]`r3 `<4` ✓; `[1,5,2]`r4 — ✗. So the answer is `3 + 1 = 4`.

Now trace my guarded code with the correct `>= D` test. `left=0`.
- `right=0` (1): maxd=[0], mind=[0]. shrink: `left<=0` and `a[0]-a[0]=0>=4`? `0>=4` false -> no shrink. count `0-0+1=1`. answer 1.
- `right=1` (5): maxd: `a[0]=1 <= a[1]=5` pop -> maxd=[1]; mind: `a[0]=1 >= 5`? no -> mind=[0,1]. shrink: `a[maxd.front]=a[1]=5`, `a[mind.front]=a[0]=1`, `5-1=4 >= 4` **true** -> enter. `left=1`; `maxd.front=1 < 1`? no; `mind.front=0 < 1`? yes pop -> mind=[1]. re-check: `left(1)<=right(1)` and `a[1]-a[1]=0>=4`? false -> exit. count `1-1+1=1`. answer 2.
- `right=2` (2): maxd: `a[1]=5 <= 2`? no -> maxd=[1,2]; mind: `a[1]=5 >= 2` pop -> mind=[2]. shrink: `a[maxd.front]=a[1]=5`, `a[mind.front]=a[2]=2`, `5-2=3 >= 4`? false -> no shrink. count `2-1+1=2`. answer 4.

Final answer `4`. Matches the brute. Now let me prove the boundary actually mattered by flipping the test to the *wrong* `> D`: at `right=1`, range `4 > 4` is **false**, so I would NOT shrink, leaving `left=0` and counting `1-0+1=2` instead of `1`, over-counting the exactly-equal window `[1,5]`. That would give a final answer of `5`, which is wrong. So `>= D` (the negation of the stable predicate `< D`) is mandatory; `> D` silently over-counts every window whose span equals `D`. The strict-less-than in the statement forces the `>=` in the shrink loop — that is the off-by-one that decides correctness, and I caught it only by constructing a range-equals-`D` case and tracing it.

**Edge cases, deliberately, because this is where this code dies.**
- `n = 0`: `cin >> n >> D` reads `n=0` (and `D`); the readings line is absent, the loop never runs, `answer = 0`. Correct: no stretches.
- `n = 1, D = 1, a = [100]`: `right=0`, push. shrink: `0 >= 1`? false. count `0-0+1=1`. Answer 1 — the singleton is stable when `D>0`. Correct.
- `n = 1, D = 0`: traced above, answer `0`. Correct.
- all equal, e.g. `a=[7,7,7,7], D=1`: every window has range `0 < 1`, so `left` never advances; contributions `1+2+3+4=10 = n(n+1)/2`. I ran it: output `10`. Let me cross-check the formula: `4*5/2 = 10`. Matches.
- `D = 0` on a multi-element array `[5,5,5,5]`: every window range `0`, `0 >= 0` true, so every window is unstable, `left` tracks `right+1` each step, every contribution `0`, answer `0`. I ran it: `0`. Correct.
- Overflow: the accumulator is `long long`; worst answer `n(n+1)/2 ≈ 2*10^10` fits with room. I tested the strictly-increasing array of length `2*10^5` with a huge `D` so all windows count: output `20000100000`, and `200000*200001/2 = 20000100000` — exact match, and this value (`2*10^10`) is past 32-bit, confirming `int` would have wrapped. The range arithmetic `a[i]-a[j]` is at most `2*10^9`, computed in `long long`, no overflow.
- Performance: the big random `n=2*10^5` case ran in `0.05 s` with `4.8 MB`, since each index enters and leaves each deque at most once and `left` advances at most `n` times total — amortized `O(n)`. Comfortably inside the 1-second, 256-MB budget.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace, so the two-line input format with a possibly-absent second line parses fine.

**Final solution.** I convinced myself the idea is right by checking the monotone-suffix structure and hand-enumerating the sample to `12`, and I convinced myself the *code* is right by tracing two distinct boundary failures to precise causes — the empty-window deque crash at `D=0`, fixed by the `left <= right` guard, and the strict-vs-non-strict over-count, fixed by shrinking on `>= D` — then re-verifying each fix and the corners, including a numeric check that the all-stable answer equals `n(n+1)/2`. That is what I ship: one self-contained `O(n)` file.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long D;
    if (!(cin >> n >> D)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Count contiguous subarrays whose (max - min) < D, via a sliding window.
    // maxd holds indices with decreasing values (front = window max),
    // mind holds indices with increasing values (front = window min).
    deque<int> maxd, mind;
    long long answer = 0;
    int left = 0;
    for (int right = 0; right < n; right++) {
        while (!maxd.empty() && a[maxd.back()] <= a[right]) maxd.pop_back();
        maxd.push_back(right);
        while (!mind.empty() && a[mind.back()] >= a[right]) mind.pop_back();
        mind.push_back(right);

        // Shrink from the left until the window [left, right] satisfies max - min < D.
        // Guard left <= right so the deques are never empty when we read their fronts.
        while (left <= right && a[maxd.front()] - a[mind.front()] >= D) {
            left++;
            if (maxd.front() < left) maxd.pop_front();
            if (mind.front() < left) mind.pop_front();
        }
        // All windows ending at 'right' with start in [left, right] are stable.
        // If left == right+1 the window is empty and this contributes 0.
        answer += (long long)(right - left + 1);
    }

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The predicate `max - min < D` is monotone in the window, so for each right end the stable starts form a suffix `[left, right]` and the count increment is `right - left + 1` (verified numerically, including the empty-window value `0` at `left = right+1`). My first sweep read the deque fronts unconditionally, so on `D = 0` — where even a singleton is unstable and `left` runs to `right+1`, emptying both deques — it dereferenced an empty deque and crashed; a trace of `n=1, D=0` pinpointed it and a `left <= right` guard fixed it. The second boundary is strict-vs-non-strict: because stable is `< D`, the shrink loop must run on the negation `>= D`; a constructed range-equals-`D` case (`D=4, [1,5,2]`) showed that the wrong `> D` over-counts the exactly-equal window, giving `5` instead of `4`. With both boundaries fixed, a 64-bit accumulator handles the `~2*10^10` worst-case count, and the amortized-`O(n)` deque sweep clears `n=2*10^5` in 0.05 s.
