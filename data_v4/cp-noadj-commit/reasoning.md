**Reading the problem and pinning the contract.** I have `n` integers `a[0..n-1]`, possibly negative, and I must choose a subset with no two chosen positions adjacent so that the sum is maximized; the empty set is allowed, so the answer is never below `0`. Input on stdin is `n` followed by the `n` values; I print one integer. Let me get the scale fixed before any algorithm, because it dictates the data types: `n <= 2*10^5` and `|a[i]| <= 10^9`, so a sum can reach `2*10^5 * 10^9 = 2*10^14`. That blows past the 32-bit range of about `2.1*10^9`, so every accumulator and every array value has to be 64-bit. I will use `long long` throughout. That is the first decision and it is non-negotiable; an `int` here is a silent wrong-answer on the large tests.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one that is easiest to type.

- *Greedy by value.* Repeatedly take the largest remaining positive value and forbid its two neighbours; stop when nothing positive is free. It is `O(n log n)` and three lines. The risk is structural: adjacency is a global constraint and greedy decides locally, which is exactly the configuration where greedy tends to be wrong. I will not trust it until I have tried to break it.
- *Linear prefix DP.* Scan left to right and carry, for each prefix, the best sum split by whether the last position is taken. `O(n)`, `O(1)` memory. The risk here is not correctness of the idea but correctness of the *transcription* — the transitions are easy to write subtly wrong.

**Stress-testing greedy before committing.** Hand-waving "greedy feels right" is how wrong solutions get shipped, so let me actually attack it with a concrete instance. Take `a = [8, 9, 2, 9, 9, -2, 8, -5]`, indices `0..7`. Greedy scans for the largest positive: the `9`s sit at indices 1, 3, 4; say it grabs index 1 first. That forbids indices 0 and 2. The largest still-free positive is the `9` at index 3; take it, forbidding 2 and 4. The only free positive left is the `8` at index 6; take it, forbidding 5 and 7. Greedy's total is `9 + 9 + 8 = 26`.

Is 26 optimal? Let me hunt for something greedy structurally could not reach. Indices `0, 2, 4, 6` are pairwise non-adjacent, with values `8 + 2 + 9 + 8 = 27`. That is strictly better than 26. So greedy is wrong, and I now see *why*: by snatching the big `9` at index 1 it blocked index 0, and that single block propagated into a worse global choice. The verification paid off — it killed an approach I would otherwise have shipped. Greedy is out.

**Deriving the DP and checking the recurrence on paper.** I want the best sum over `a[0..i]` under the adjacency rule, and the only thing the future cares about is whether position `i` was taken (that is what restricts position `i+1`). So I keep two quantities per prefix:

- `skip` = best sum over `a[0..i]` with position `i` **not** taken,
- `take` = best sum over `a[0..i]` with position `i` taken.

If I do not take `i`, then `i-1` was unconstrained, so `skip_i = max(skip_{i-1}, take_{i-1})`. If I take `i`, then `i-1` must be untaken and I add `a[i]`, so `take_i = skip_{i-1} + a[i]`. The answer is `max(take_{n-1}, skip_{n-1}, 0)`; the `0` encodes the empty selection and is what makes all-negative inputs return `0` instead of a negative. Base case before any element: "nothing taken" has sum `0`, and "last taken" is impossible, so `skip = 0`, `take = -infinity`.

Let me confirm the recurrence by hand on the sample `a = [5, 1, 1, 5, 1, 5]`, answer `15` (indices 0, 3, 5). Start `(take, skip) = (-inf, 0)`. i=0 (5): `take=0+5=5`, `skip=max(0,-inf)=0`. i=1 (1): `take=0+1=1`, `skip=max(0,5)=5`. i=2 (1): `take=5+1=6`, `skip=max(5,1)=5`. i=3 (5): `take=5+5=10`, `skip=max(5,6)=6`. i=4 (1): `take=6+1=7`, `skip=max(6,10)=10`. i=5 (5): `take=10+5=15`, `skip=max(10,7)=10`. Answer `max(15,10,0)=15`. The recurrence is right.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the loop body:

```
long long take = 0, skip = 0;
for (int i = 0; i < n; i++) {
    skip = max(skip, take);   // best if i is not taken
    take = skip + a[i];       // best if i is taken
}
```

Something about updating `skip` and then using it in the same iteration looks dangerous, so I trace the smallest input that could expose it: `a = [1, 1]`, where the answer is obviously `1` (the two ones are adjacent, I can keep only one). Start `take=0, skip=0`. i=0: `skip=max(0,0)=0`; `take=0+1=1`. i=1: `skip=max(0,1)=1`; `take=1+1=2`. Final `max(2,1,0)=2`.

**Diagnosing the bug.** The code returns `2` — it took *both* adjacent ones, which is illegal. The defect is precise: on `i=1` I overwrote `skip` with `max(skip, take)=1`, folding in the `take` from index 0, and *then* computed `take = skip + a[1]` on top of that updated `skip`. So the new `take` means "take index 1 on a state that already took index 0" — the exact adjacency I forbade. Both transitions need the *previous* pair `(skip_{i-1}, take_{i-1})`, but I destroyed `skip` before reading it for `take`. There is a second, quieter bug too: initializing `take = 0` asserts a last-taken state with sum `0` exists before any element, which would let the model skip a lone negative through a phantom; `take` must start at negative infinity.

**Fixing and re-verifying.** Compute both new values from the old pair via temporaries, and fix the base case:

```
long long take = LLONG_MIN / 4, skip = 0;
for (int i = 0; i < n; i++) {
    long long ntake = skip + a[i];        // take i: previous skipped
    long long nskip = max(skip, take);    // skip i: previous either
    take = ntake; skip = nskip;
}
```

Re-trace `[1, 1]`: start `(-inf, 0)`. i=0: `ntake=0+1=1`, `nskip=max(0,-inf)=0` -> `(1, 0)`. i=1: `ntake=0+1=1`, `nskip=max(0,1)=1` -> `(1, 1)`. Answer `max(1,1,0)=1`. Correct. Re-trace `[3, 4]` (answer 4): i=0 -> `(3,0)`; i=1 -> `ntake=0+4=4`, `nskip=max(0,3)=3` -> `(4,3)`; answer `4`. Correct. The two cases that broke before now pass, and they broke for the reason I fixed, which is the evidence I trust.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: the loop never runs; `max(take, skip, 0) = max(-inf, 0, 0) = 0`. The empty selection — correct.
- `n = 1`, `a = [-7]`: i=0 -> `ntake = 0 + (-7) = -7`, `nskip = max(0,-inf) = 0` -> answer `max(-7, 0, 0) = 0`. Take nothing rather than a loss — correct.
- All negative, `[-3,-1,-4]`: every `take` stays negative, `skip` stays `0`, answer `0`. Correct.
- Overflow: accumulators are `long long`; the maximum sum `~2*10^14` fits with room to spare. The sentinel `LLONG_MIN/4` is only ever read inside a `max`, never has `a[i]` added to it (I only add to `skip`), so it cannot underflow. Safe.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace so the input parsing is format-agnostic.

**Final solution.** I convinced myself the idea is right by disproving greedy and hand-checking the recurrence, and I convinced myself the *code* is right by tracing the failing case to a precise cause and re-verifying the fix and the corners. That is what I ship — one self-contained file, the simple `O(n)` DP I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // skip = best prefix sum with the last position NOT taken (starts 0: empty set);
    // take = best with the last position taken (starts -inf: impossible before any element).
    long long take = LLONG_MIN / 4, skip = 0;
    for (int i = 0; i < n; i++) {
        long long ntake = skip + a[i];     // take i  => i-1 skipped
        long long nskip = max(skip, take); // skip i  => i-1 either
        take = ntake;
        skip = nskip;
    }

    cout << max({take, skip, 0LL}) << "\n"; // empty selection always allowed
    return 0;
}
```

**Causal recap.** Greedy looked right but a single traced counterexample (`[8,9,2,9,9,-2,8,-5]`: greedy 26 vs the reachable 27) showed a local max-grab blocks more than it gains, so I moved to the prefix DP and checked its recurrence on the sample; the DP's two transitions both read the previous `(skip, take)` pair, which my first in-place loop destroyed by overwriting `skip` first — a trace of `[1,1]` returning the illegal `2` pinpointed it; computing both new values from the old pair and starting `take` at negative infinity fixes it; and `max(..., 0)` over 64-bit accumulators closes out the all-negative, empty, single-element, and overflow corners.
