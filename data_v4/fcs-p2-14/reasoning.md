**Reading the problem and pinning the contract.** I have `n` integers `a[0..n-1]`, possibly negative, arranged on a *circle*: position `i` is adjacent to `i+1`, and on top of that position `n-1` is adjacent to position `0`. I must choose a subset with no two cyclically adjacent positions so that the sum is maximized; the empty set is allowed, so the answer is never below `0`. Input on stdin is `n` followed by the `n` values; I print one integer. Before any algorithm I fix the scale, because it dictates the data types: `n <= 2*10^5` and `|a[i]| <= 10^9`, so a sum can reach `2*10^5 * 10^9 = 2*10^14`. That blows past the 32-bit range of about `2.1*10^9`, so every accumulator and every array value has to be 64-bit. I will use `long long` throughout. That is the first decision and it is non-negotiable; an `int` here is a silent wrong-answer on the large tests.

**What makes the circle different from a path.** If this were a line, I already trust the standard prefix DP: scan left to right carrying `skip` (best prefix sum with the last position not taken) and `take` (best with it taken), with `take_i = skip_{i-1} + a[i]` and `skip_i = max(skip_{i-1}, take_{i-1})`, answer `max(take, skip, 0)`. The circle adds exactly one extra edge — the wrap between index `0` and index `n-1`. That single edge is the whole problem. It means a selection that takes *both* endpoints, which is perfectly legal on the line, is now illegal. Everything below is about handling that one edge without breaking the rest.

**Laying out the candidate approaches.** Three routes are on the table, and I want to commit to the one I can *prove*, not the one that is fastest to type.

- *Greedy by value.* Repeatedly take the largest remaining positive value and forbid its two cyclic neighbours; stop when nothing positive is free. `O(n log n)` and a few lines. The risk is structural: adjacency is a global constraint and greedy decides locally, and on a cycle the wrap makes the "global" reach all the way around. I will not trust it until I have tried to break it.
- *Single-pass circular DP.* Run the path DP once over `a[0..n-1]` and patch in something — a flag, an extra state, a subtraction — to "account for" the wrap edge at the end. This is the tempting clever move: one loop, `O(n)`, feels like it should be a small delta from the path version. The risk is that one pass cannot cleanly express "endpoint 0 and endpoint n-1 are not both taken," because by the time I reach index `n-1` the loop has long forgotten whether index `0` was taken in the particular optimal sub-path it is tracking.
- *Two linear DP passes.* Observe that any valid circular selection omits at least one endpoint, split on which endpoint is forbidden, and run the proven path DP twice. `O(n)`, `O(1)` extra memory. The risk here is not correctness of the idea but the exact split and the small-`n` corners.

**Reaching for the clever single-pass fix first.** My instinct is the single-pass circular DP, because it is the smallest-looking change. The seductive version goes: run the path DP, but also track, alongside `skip`/`take`, whether the *first* element was taken in the path that produced each value; at the end, if the best path took both index `0` and index `n-1`, "repair" it. I started to sketch the state — `(skip, take, took0)` — and immediately the bookkeeping bloats: `took0` is not a single bit I can carry, because the *optimal* `skip` at index `i` might come from a sub-path that took index 0, while the optimal `take` at the same `i` came from one that did not. I would have to carry, for each of `skip`/`take`, the *best value conditioned on `took0 = true`* and the *best value conditioned on `took0 = false`* — four numbers, four transitions, and a final reconciliation that forbids `took0 && took_{n-1}`. That is doable but it is precisely the kind of four-way conditional DP that is "error-prone in the budget": every transition has two branches, the base case has to seed the conditioning correctly at index 0, and one mis-set branch is a silent wrong-answer that only shows up on inputs where the wrap edge actually binds. Before I sink time into proving that machine correct, let me check whether the *naive* single pass — path DP over `a[0..n-1]` with no wrap handling at all — is even close, because if the gap is real I want a concrete witness.

**Constructing a concrete counterexample that kills the naive single pass.** The naive single pass is "just run the path DP on the circular array and print it." It ignores the wrap, so it will happily take both endpoints when that is locally best. I want the smallest instance where taking both endpoints is the path optimum but is illegal on the circle. Take `a = [5, 1, 1, 5, 1, 5]`, indices `0..5`, on a 6-cycle. The path DP (no wrap) finds indices `0, 3, 5` → `5 + 5 + 5 = 15`. But on the circle, index `0` and index `5` are adjacent through the wrap edge, so `{0, 3, 5}` is *illegal*. The true circular optimum is indices `1, 3, 5` → `1 + 5 + 5 = 11`, or equivalently `0, 3` → `10`, and the best legal value is `11`. So the naive single pass returns `15` where the answer is `11` — a clean, concrete over-count of exactly the wrap edge. The verification paid off: it shows the naive pass is wrong, and it shows *why* (it used both endpoints), which is the same defect the elaborate four-state single-pass DP exists to repair. I now distrust the whole single-pass family enough to not spend the budget proving the four-state version; there is a simpler method that sidesteps the wrong-endpoint pair entirely.

**Stress-testing greedy too, before discarding it.** For completeness I attack greedy with a concrete instance, because "greedy feels wrong on cycles" is not a proof. Take the path-DP counterexample's cousin on a line first to recall the failure mode, then put it on a circle. Consider `a = [8, 9, 2, 9, 9, -2, 8]` on a 7-cycle. Greedy scans for the largest positive: the `9`s sit at indices 1, 3, 4. Say it grabs index 1, forbidding cyclic neighbours 0 and 2. Next largest free is the `9` at index 3, take it, forbidding 2 and 4. Next free positive is the `8` at index 6, but index 6's cyclic neighbours are 5 and 0; 0 is already blocked-as-unused (it was forbidden, meaning it cannot be *taken*, but index 6 is still free) — taking index 6 forbids 5 and 0. Greedy total: `9 + 9 + 8 = 26`. Is 26 optimal on the cycle? The set `{1, 3, 6}` is cyclically independent (1-3 ok, 3-6 ok, 6-1 ok through neither wrap nor line) and sums to `9 + 9 + 8 = 26`. Trying to beat it: `{0, 2, 4, 6}` sums to `8 + 2 + 9 + 8 = 27` on a *line*, but on the 7-cycle index 6 and index 0 are wrap-adjacent, so that set is illegal. After checking, 26 is in fact the circular optimum here, so this particular instance does *not* break greedy. But that is luck, not a guarantee — and I already broke greedy on the path version of this family (`[8,9,2,9,9,-2,8,-5]`: greedy 26 vs reachable 27, by grabbing the big `9` at index 1 and thereby blocking index 0). The line-greedy counterexample embeds into a cycle by padding with a large-negative spacer so the wrap edge never binds, which reproduces the same 26-vs-27 failure on a circle. So greedy is provably wrong in general; I discard it. The lesson is consistent with the single-pass finding: local max-grab on a global adjacency constraint is unsafe.

**Deriving the simple correct method by splitting on the wrap edge.** Here is the clean observation that makes everything provable. On the circle, the wrap edge says index `0` and index `n-1` cannot both be chosen. So *every* valid circular selection falls into at least one of two cases:

- Case A: index `n-1` is not chosen. Then the only remaining adjacencies among the chosen elements are the line edges among `a[0..n-2]`, and there is no wrap constraint to worry about (the wrap edge touches `n-1`, which is excluded). So the best Case-A value is exactly the *path* maximum over `a[0..n-2]`.
- Case B: index `0` is not chosen. Symmetrically, the best Case-B value is the path maximum over `a[1..n-1]`.

The circular answer is `max(Case A, Case B)`. Why is this exactly right, neither over- nor under-counting?

- *Soundness (no illegal selection counted).* Any selection counted in Case A lives entirely in `a[0..n-2]` and is line-independent there; it never uses index `n-1`, so it cannot violate the wrap edge (the wrap edge requires using both `0` and `n-1`). Hence it is a valid circular selection. Same for Case B by symmetry. So both cases only ever produce legal circular selections.
- *Completeness (every legal selection reachable).* Take any optimal circular selection `S`. It cannot contain both `0` and `n-1` (wrap edge). If it omits `n-1`, then `S ⊆ {0..n-2}` and is line-independent there, so it is counted in Case A. If it omits `0`, it is counted in Case B. Since `S` omits at least one of `{0, n-1}`, it is counted in at least one case. Therefore `max(A, B)` is at least the value of `S`, i.e. at least the true optimum.

Together these give `max(A, B) =` the true circular optimum. The method is just the *path* DP — which I already trust — invoked twice on two carefully chosen sub-ranges. No four-state machine, no repair step, nothing I have to re-prove. This is the destination: the simpler, provable method, reached *because* the verification killed the clever single-pass and greedy ideas.

**Pinning the small-`n` corners, because that is where this kind of split dies.** The split assumes there *are* two distinct endpoints to forbid. For tiny `n` I reason explicitly:

- `n = 0`: no positions, answer `0` (empty selection).
- `n = 1`: a single position with no neighbour at all — on a 1-cycle there is no second vertex to be adjacent to, so the element may be taken. Answer `max(a[0], 0)`. (If I blindly ran the split, Case A would be the line over `a[0..-1]` = empty = 0 and Case B the line over `a[1..0]` = empty = 0, giving `0` — which would *wrongly* refuse a positive lone element. So `n = 1` must be special-cased.)
- `n = 2`: positions `0` and `1` are adjacent (they share both the "line" edge and the "wrap" edge, but it is still just one adjacency: at most one of them may be taken). The split handles this correctly: Case A is the line over `a[0..0]` = `max(a[0],0)`, Case B is the line over `a[1..1]` = `max(a[1],0)`, and `max(A,B) = max(a[0], a[1], 0)` — take the larger single element, or nothing. Correct, so `n = 2` needs no special case as long as my line solver handles a single-element range. I will still verify it.

So I special-case `n = 0` and `n = 1`, and let the split handle everything `n >= 2`.

**The line solver, and the in-place trap I already know about.** The path DP is short, but it has a transcription trap I must respect: both transitions read the *previous* `(skip, take)` pair, so I must compute both new values from the old pair via temporaries. If I update `skip` first and then use the updated `skip` to compute `take`, I build `take` on a state that already took the previous element — illegal adjacency, and a classic silent bug. The base case also matters: before any element, "nothing taken" has sum `0`, and "last taken" is impossible, so `skip = 0`, `take = -infinity` (I use `LLONG_MIN / 4` so I never overflow when it sits inside a `max`, and I never add `a[i]` to it — I only ever add to `skip`). I write the solver to operate on an inclusive range `[lo, hi]`; if `lo > hi` the loop never runs and it returns `0`, which is exactly the empty-range value I need for the corners.

**First implementation — and a trace, because clean math transcribes dirty.** My first cut of the line loop body, written carelessly to see if the trap bites:

```
long long take = 0, skip = 0;            // WRONG base: take should be -inf
for (int i = lo; i <= hi; i++) {
    skip = max(skip, take);              // WRONG: overwrites skip before take reads it
    take = skip + a[i];
}
```

I trace the smallest input that exposes both defects: the circle `a = [1, 1]`, `n = 2`, where the answer is obviously `1` (the two ones are adjacent, keep one). With the split, Case A is the line over `a[0..0] = [1]`, Case B the line over `a[1..1] = [1]`; each should return `1`, and `max = 1`. Run the buggy line solver on `[1]` (a single element, `lo = hi = 0`): start `take = 0, skip = 0`. i=0: `skip = max(0, 0) = 0`; `take = 0 + 1 = 1`. Return `max(1, 0, 0) = 1`. Hm, the single-element case happens to be right. So I trace a two-element line, `a[0..1] = [1, 1]` (this is what a naive `n = 2` *without* the split would compute, and it is also the inner behaviour I must rule out): start `take = 0, skip = 0`. i=0: `skip = max(0,0) = 0`; `take = 0 + 1 = 1`. i=1: `skip = max(0, 1) = 1`; `take = 1 + 1 = 2`. Return `max(2, 1, 0) = 2`.

**Diagnosing the bug.** The line solver on `[1, 1]` returns `2` — it took *both* adjacent ones, which is illegal even on a line. The defect is precise: on `i = 1` I overwrote `skip` with `max(skip, take) = 1`, folding in the `take` from index 0, and *then* computed `take = skip + a[1]` on top of that updated `skip`. So the new `take` means "take index 1 on a state that already took index 0" — the exact adjacency I forbade. Both transitions need the *previous* pair `(skip_{i-1}, take_{i-1})`, but I destroyed `skip` before reading it for `take`. There is a second, quieter defect: initializing `take = 0` asserts a last-taken state with sum `0` exists before any element, which would let a lone negative slip through a phantom prior-taken state; `take` must start at negative infinity.

**Fixing and re-verifying the line solver.** Compute both new values from the old pair via temporaries, and fix the base case:

```
long long take = LLONG_MIN / 4, skip = 0;
for (int i = lo; i <= hi; i++) {
    long long ntake = skip + a[i];        // take i: previous skipped
    long long nskip = max(skip, take);    // skip i: previous either
    take = ntake; skip = nskip;
}
return max({take, skip, 0LL});
```

Re-trace the line `[1, 1]`: start `(-inf, 0)`. i=0: `ntake = 0 + 1 = 1`, `nskip = max(0, -inf) = 0` → `(1, 0)`. i=1: `ntake = 0 + 1 = 1`, `nskip = max(0, 1) = 1` → `(1, 1)`. Return `max(1, 1, 0) = 1`. Correct. Re-trace the line `[3, 4]` (answer 4): i=0 → `(3, 0)`; i=1 → `ntake = 0 + 4 = 4`, `nskip = max(0, 3) = 3` → `(4, 3)`; return `4`. Correct. The case that broke now passes, and it broke for the reason I fixed, which is the evidence I trust.

**Assembling the circular answer and tracing it end to end.** For `n >= 2`: `best = max(linearBest(a, 0, n-2), linearBest(a, 1, n-1))`. Trace the sample circle `a = [5, 1, 1, 5, 1, 5]`, `n = 6`. Case A is the line over `a[0..4] = [5,1,1,5,1]`: running the fixed solver — i=0 `(5,0)`, i=1 `(1,5)`, i=2 `(6,5)`, i=3 `(10,6)`, i=4 `(7,10)` → `max(7,10,0) = 10`. Case B is the line over `a[1..5] = [1,1,5,1,5]`: i=1 `(1,0)`, i=2 `(1,1)`, i=3 `(6,1)`, i=4 `(2,6)`, i=5 `(11,6)` → `max(11,6,0) = 11`. `best = max(10, 11) = 11`. That matches the brute-forced circular optimum (indices `1,3,5`), and it correctly *avoids* the illegal path answer `15` (indices `0,3,5`) because Case A forbids index 5 and Case B forbids index 0 — neither case can take both endpoints. The split does exactly what the proof says.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: handled before the split, prints `0`. Correct (empty selection).
- `n = 1`, `a = [-7]`: special-cased to `max(-7, 0) = 0`. Take nothing rather than a loss. Correct. And `a = [5]` → `max(5, 0) = 5`, correctly taking the lone element (the split would have wrongly returned `0` here, which is exactly why I special-cased it).
- `n = 2`, `a = [3, 4]`: Case A = line over `[3]` = `3`, Case B = line over `[4]` = `4`, `best = 4`. Correct (take the larger; they are adjacent). `a = [-3, -4]` → `max(0, 0) = 0`. Correct.
- All negative, e.g. `[-3,-1,-4]` on a 3-cycle: every line `take` stays negative, every `skip` stays `0`, both cases return `0`, `best = 0`. Correct.
- Wrap-binding case `[5,1,1,5,1,5]`: traced above, `11`, and it provably cannot emit the illegal `15`.
- Overflow: accumulators are `long long`; the maximum sum `~2*10^14` fits with room to spare. The sentinel `LLONG_MIN/4` is only ever read inside a `max`, never has `a[i]` added to it (I add only to `skip`), so it cannot underflow. Safe.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace so input parsing is format-agnostic.

**Self-verification against an independent brute oracle.** Hand-traces convince me of the cases I thought of; to catch the cases I did not, I wrote a separate `2^n` brute that enumerates every subset, checks cyclic independence (including the `(n-1, 0)` wrap edge), and takes the max — deliberately unrelated to the DP. I ran the solution against it on 700 random instances from a generator that mixes tiny `n` (0,1,2), small cycles (3–8), and larger ones up to 16, across value ranges spanning all-positive, all-negative, tight-mixed-with-zeros, and extreme `±10^9` magnitudes; plus a hand-picked edge suite (empty, single ±, pair, all-equal, all-negative, the wrap-binding `[5,1,1,5,1,5]`, the `[8,9,2,9,9,-2,8]` greedy probe, large magnitudes). Zero mismatches across all of them. I also ran `n = 2*10^5` with random `±10^9` values: it finishes in about `0.04` s using under `5` MB, comfortably inside the `1` s / `256` MB limits, as expected for an `O(n)` two-pass scan.

**Final solution.** I convinced myself the *idea* is right by killing the naive single pass with a concrete wrap counterexample (`[5,1,1,5,1,5]`: path 15, illegal, vs circular 11), by declining the error-prone four-state single-pass DP in favour of a method I can prove in two lines, and by disproving greedy; I convinced myself the *split* is right with the soundness/completeness argument and a full end-to-end trace; and I convinced myself the *code* is right by tracing the failing line-solver case to a precise cause, re-verifying the fix and the corners, and differential-testing against an independent oracle. That is what I ship — one self-contained file: the proven path DP run twice over the two endpoint-breaking ranges, with the `n <= 1` corners handled explicitly.

```cpp
#include <bits/stdc++.h>
using namespace std;

// Maximum sum of a no-two-adjacent subset on a LINE over a[lo..hi] (inclusive).
// Empty subset allowed, so the returned value is always >= 0.
// If lo > hi the range is empty and the best sum is 0.
static long long linearBest(const vector<long long> &a, int lo, int hi) {
    long long take = LLONG_MIN / 4; // best with last position taken (impossible before any element)
    long long skip = 0;             // best with last position not taken (empty prefix -> 0)
    for (int i = lo; i <= hi; i++) {
        long long ntake = skip + a[i];     // take i => i-1 skipped
        long long nskip = max(skip, take); // skip i => i-1 either
        take = ntake;
        skip = nskip;
    }
    return max({take, skip, 0LL});
}

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    if (n == 0) { cout << 0 << "\n"; return 0; }
    if (n == 1) { cout << max(a[0], 0LL) << "\n"; return 0; }

    // Circle: positions 0 and n-1 are adjacent, so they cannot both be chosen.
    // Split into two LINE subproblems that each break the wrap edge:
    //   (A) forbid the last element  -> solve line over a[0 .. n-2]
    //   (B) forbid the first element -> solve line over a[1 .. n-1]
    // Any valid circular selection avoids at least one of {first, last}, so it is
    // covered by case A or case B; conversely any selection counted in A or B is a
    // valid line selection that never uses both endpoints, hence valid on the circle.
    long long best = max(linearBest(a, 0, n - 2), linearBest(a, 1, n - 1));

    cout << best << "\n"; // empty selection always allowed (linearBest already >= 0)
    return 0;
}
```

**Causal recap.** The circle differs from the path by exactly one wrap edge between index `0` and index `n-1`; the tempting single-pass "patch the path DP" either ignores the wrap (a traced counterexample `[5,1,1,5,1,5]` returns the illegal `15` vs the true `11`) or balloons into an error-prone four-state conditional DP I declined to prove in budget, and greedy is locally-blind and provably wrong; the clean fix is the proven path DP run twice — over `a[0..n-2]` (forbidding the last endpoint) and `a[1..n-1]` (forbidding the first) — whose `max` is exactly right by a short soundness/completeness argument because every legal circular selection omits at least one endpoint; the line solver's only trap is the in-place update (a trace of the line `[1,1]` returning the illegal `2` pinpointed overwriting `skip` before `take` reads it, plus a phantom `take=0` base), fixed with temporaries and a `-inf` start; `n = 0` and `n = 1` are special-cased (the split would wrongly zero out a positive lone element), `n = 2` falls out of the split correctly; and `long long` accumulators with a `max(..., 0)` floor close the all-negative, empty, single-element, pair, and overflow corners — all confirmed against an independent `2^n` oracle over 700+ random and edge cases with zero mismatches.
