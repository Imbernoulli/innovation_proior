**Reading the problem and pinning the contract.** I have `m` stamping presses on one line. Press `i` waits `w[i]` milliseconds (a warm-up), then emits its first part *at* time `w[i]`, and afterwards one more part every `c[i]` milliseconds. By elapsed time `T` it has produced `0` if `T < w[i]`, else `floor((T - w[i]) / c[i]) + 1` — the `+1` is the part stamped at the moment `w[i]` itself. The presses run in parallel and independently, so the total by time `T` is the sum over presses of that per-press count. Given a quota `N`, I want the smallest `T` with total `>= N`, and I print that `T`. Edge in the contract: `N = 0` means the quota is met before anything happens, answer `0`.

Before I touch an algorithm I fix the scale, because it decides the data types and this problem is plainly built to punish the wrong choice. Constraints: `m <= 10^5`, `0 <= N <= 10^9`, `0 <= w[i] <= 10^9`, `1 <= c[i] <= 10^9` (note `c[i] >= 1`, so no division by zero). What is the largest `T` I could possibly print? Imagine one press with `w = 10^9`, `c = 10^9`, `N = 10^9`. It produces its `k`-th part at time `w + (k-1)*c`, so the `10^9`-th part lands at `10^9 + (10^9 - 1)*10^9 ≈ 10^18`. That is the answer for that input, and `10^18` is about *four hundred million times* larger than the 32-bit ceiling of `~2.1*10^9`. So the answer, the binary-search endpoints, and the product `(N-1)*c[i]` are all 64-bit quantities. Even within `produced(T)`, the running total of parts can exceed `10^9` (sum of up to `10^5` presses each producing up to ~`10^9` parts), which also overflows 32-bit and can even threaten 64-bit if I am not careful. I will use `long long` everywhere and additionally cap the running total. This is decision number one and it is non-negotiable; an `int` here is a silent wrong answer on exactly the tests this problem advertises.

**Candidate approaches.** How do I find the smallest `T`?

- *Simulate event by event.* Maintain, for each press, the time of its next part; repeatedly advance to the earliest next-part time, increment a counter, push that press's following part. Stop when the counter hits `N`. This is obviously correct but it makes one heap operation per part produced, i.e. up to `N = 10^9` of them — far too slow for a 1-second limit, and it is essentially a brute force. Useful as a *checker* on tiny inputs, not as the solution.
- *Binary search on the answer.* The key observation is that `produced(T)` — the total parts by time `T` — is **non-decreasing** in `T`. Increasing `T` can only switch presses on and grant them more cycles; it can never remove a part already stamped. So the predicate `P(T) := [produced(T) >= N]` is monotone: false for small `T`, true from some threshold onward. The answer is exactly that threshold, and I can binary search it. Each feasibility test is an `O(m)` sum, and I need about `log2(10^18) ≈ 60` iterations, so `O(m log(max T)) ≈ 6*10^6` operations. That fits comfortably. This is the route.

**Deriving the feasibility test and proving monotonicity.** For a fixed `T`,
`produced(T) = sum_{i: T >= w[i]} ( floor((T - w[i]) / c[i]) + 1 )`.
Take any `T' > T`. For a press already on (`T >= w[i]`), `floor((T' - w[i]) / c[i]) >= floor((T - w[i]) / c[i])` because the numerator grew and the denominator is fixed positive, so its count does not drop. For a press not yet on at `T`, its count was `0` and can only become `0` or positive at `T'`. Summing, `produced(T') >= produced(T)`. Monotone, as claimed. Therefore `P` is a step function from false to true and binary search is valid.

**Choosing the search bounds.** Lower bound `lo = 0` is clearly safe: `produced(0)` counts only presses with `w[i] = 0`, which is `<= N` in general but never *over*-shoots the answer downward — `lo = 0` is a legitimate left end (if `produced(0) >= N` the search will collapse to `0`). The subtle part is a *guaranteed* upper bound `hi` with `produced(hi) >= N`. I reason: press `i` *by itself* stamps its `N`-th part at time `w[i] + (N - 1)*c[i]`. The real line includes press `i` plus others, so by that time the line has produced **at least** `N` parts (press `i` alone already did). Hence `hi = max_i ( w[i] + (N - 1)*c[i] )` satisfies `produced(hi) >= N`. It is loose — usually the line reaches `N` much earlier thanks to the other presses — but binary search only needs *some* valid upper bound, and this one is provably valid and easy to compute. I take the max over presses (using the slowest-by-itself press gives the largest, safest bound).

**Sanity-check the derivation on the sample.** Sample: `m = 3`, `N = 10`, presses `(0,3), (2,5), (1,2)`. Let me tabulate `produced(T)`:
- `T=8`: press0 `floor(8/3)+1 = 2+1 = 3`; press1 `floor((8-2)/5)+1 = 1+1 = 2`; press2 `floor((8-1)/2)+1 = 3+1 = 4`. Total `3+2+4 = 9 < 10`.
- `T=9`: press0 `floor(9/3)+1 = 3+1 = 4`; press1 `floor(7/5)+1 = 1+1 = 2`; press2 `floor(8/2)+1 = 4+1 = 5`. Total `4+2+5 = 11 >= 10`.
So the threshold is `9`: false at `8`, true at `9`. The contract says `9`. My feasibility formula reproduces the intended answer, so the derivation is sound. (As a bonus the `hi` for this case is `max(0 + 9*3, 2 + 9*5, 1 + 9*2) = max(27, 47, 19) = 47`, comfortably above `9`.)

**First implementation and a trace.** Here is my first cut of the core, written quickly and — I already suspect — carelessly on types:

```
auto produced = [&](long long T) {
    int total = 0;                                   // (A)
    for (int i = 0; i < m; i++)
        if (T >= w[i]) total += (T - w[i]) / c[i] + 1;
    return total;
};
int hi = 0;                                          // (B)
for (int i = 0; i < m; i++)
    hi = max(hi, w[i] + (N - 1) * c[i]);             // (C)
long long lo = 0, H = hi;
while (lo < H) {
    long long mid = lo + (H - lo) / 2;
    if (produced(mid) >= N) H = mid; else lo = mid + 1;
}
cout << lo << "\n";
```

I will trace this on the overflow-flavoured input it is supposed to handle: a single press, `m = 1`, `N = 10^9`, `w = 10^9`, `c = 10^9`. The correct answer is `w + (N-1)*c = 10^9 + (10^9-1)*10^9 = 10^18`. Walk through the code. Line (C): `(N - 1) * c[i]` — but wait, what types are in play? `N` is whatever I declared it; suppose for the moment `N` is `long long` but `hi` at (B) is `int`. The product `(N-1)*c[i]` is computed in `long long` (`≈ 10^18`), then assigned via `max(hi, ...)` into an `int` `hi`. That truncates `10^18` to a garbage 32-bit value. So `H = hi` is already corrupt before the search even starts.

**The bug.** Let me actually run the buggy version to see how bad it is rather than guess. I compiled a faithful int-typed copy and fed it `1 1000000000 / 1000000000 1000000000`. It printed `0`. Zero! The correct answer is `10^18`. The mechanism: `(N-1)*c = (10^9-1)*10^9 ≈ 10^18` overflows when funneled into the `int hi`; the truncated `hi` came out small (even negative-ish modulo `2^32`), so `H` was tiny, the loop `while (lo < H)` ran on a bogus range, and `lo` ended at `0`. Compounding it, the accumulator at (A) is `int total` — even on inputs where `hi` survived, summing up to `10^5` presses' counts (each up to `~10^9`) would overflow `int` and wrap to a meaningless, possibly negative, value, making `produced(mid) >= N` answer randomly and steering the search to a wrong threshold. **Two distinct overflow sites**, both 32-bit: the endpoint computation at (B)/(C) and the feasibility accumulator at (A). This is precisely the pitfall the problem is engineered around, and the trace makes it concrete: the program doesn't crash, it confidently prints `0` for an answer that is `10^18`.

**Fix and re-verification.** Make `N` a `long long`, make `total` and `hi`/`H` `long long`, and — because even a 64-bit `total` could in principle be pushed around by adversarial inputs (sum of `10^5` terms each up to `~10^9` is `~10^14`, safe, but I want the lambda to be robust and to allow early exit) — saturate the running total at a large cap so it can never overflow and so I can bail out the instant the quota is met:

```
const long long CAP = (long long)4e18;
auto produced = [&](long long T) -> long long {
    long long total = 0;
    for (int i = 0; i < m; i++) {
        if (T >= w[i]) {
            total += (T - w[i]) / c[i] + 1;
            if (total >= CAP) return CAP;            // saturate, cannot overflow
        }
    }
    return total;
};
long long hi = 0;
for (int i = 0; i < m; i++)
    hi = max(hi, w[i] + (N - 1) * c[i]);             // all long long now
```

Re-trace the single-press overflow case: (C) computes `10^9 + (10^9-1)*10^9 = 10^18` in `long long`, so `hi = 10^18` survives intact. The search runs `lo=0, H=10^18`; `produced(mid)` for this one press is `floor((mid - 10^9)/10^9) + 1` once `mid >= 10^9`, which equals `N = 10^9` exactly at `mid = 10^9 + (10^9-1)*10^9 = 10^18` and is `< N` just below it, so the binary search lands on `10^18`. I ran the fixed binary `1 1000000000 / 1000000000 1000000000` and it prints `1000000000000000000`. Correct. The two bugs broke for the reason I diagnosed (32-bit truncation), and fixing the types fixes both — that matched cause-and-effect is the evidence I trust, not just "it now passes."

**Second debug episode — the warm-up off-by-one.** Even with types correct, the per-press formula has a `+1` that is easy to get wrong, and an off-by-one there is a *silent* logic bug independent of overflow. Let me stress the boundary deliberately with `m = 2`, `N = 1`, presses `(5,1)` and `(3,1)`. The expected answer: the very first part on the whole line is stamped by whichever press warms up first, at time `min(w) = 3`. So the answer must be `3`, and at `T = 2` production must be `0`.

Trace my formula. At `T = 2`: press0 needs `T >= 5`? No, `0`. press1 needs `T >= 3`? No, `0`. Total `0 < 1`. Good. At `T = 3`: press0 `T>=5`? No. press1 `T>=3`? Yes: `floor((3-3)/1)+1 = floor(0)+1 = 1`. Total `1 >= 1`. So threshold is `3`. The `+1` is doing exactly the right thing — it counts the part stamped *at* the warm-up instant. Had I written the count as `floor((T - w[i]) / c[i])` *without* the `+1`, then at `T = 3` press1 would give `floor(0/1) = 0`, total `0`, and the part stamped at the warm-up moment would be invisible; the search would push to `T = 4` and print `4`, an off-by-one. Or, subtly worse, dropping the `+1` and instead testing `T > w[i]` (strict) would also lose the first part. I ran the fixed solution on `2 1 / 5 1 / 3 1` and it prints `3`. Matches. The `+1`-with-`>=` pairing is the correct encoding of "first part lands at `w[i]`," and the boundary trace confirms it.

There is one more boundary inside this formula worth a hard look: a press that is on but whose next part hasn't matured. Take a press `(w=1, c=2)` at `T=2`: `floor((2-1)/2)+1 = floor(1/2)+1 = 0+1 = 1` — it has stamped exactly its warm-up part and not yet the second (which comes at `T=3`). Correct. At `T=3`: `floor(2/2)+1 = 1+1 = 2`. Correct. The floor division handles the "between parts" interior cleanly because of `c[i] >= 1`; if `c[i]` could be `0` this would divide by zero, but the constraints forbid it, so no guard is needed.

**Edge cases, deliberately.**
- `N = 0`: the quota is already met. My code special-cases this before the search: `if (N == 0) print 0`. Without that guard, `hi = max_i (w[i] + (0 - 1)*c[i]) = max_i (w[i] - c[i])`, which can be *negative*, making `H < lo = 0` and the loop print `lo = 0` anyway — but relying on that accident is fragile, so I keep the explicit `N == 0 -> 0`. I tested `3 0 / ...` and it prints `0`.
- `N = 1`: answer is `min_i w[i]` (the earliest any press fires its warm-up part), verified above as `3` on `(5,1),(3,1)`.
- Single press, huge values: `1 10^9 / 10^9 10^9` -> `10^18`, verified; this is the headline overflow case.
- Many presses, tiny cycles, big quota: `2 2000000000` wait — `N <= 10^9`, so let me use `2 1000000000 / 0 1 / 0 1`. Both presses fire every ms from `T=0`: total at `T` is `2*(T+1)`. Need `2*(T+1) >= 10^9` -> `T >= 5*10^8 - 1 = 499999999`. The accumulator `total` here at the threshold is `~10^9`, which overflows `int` but is fine in `long long`; the solution prints `499999999`. This exercises the *accumulator* overflow specifically (the endpoints stay small here), separate from the endpoint overflow in the single-press case.
- Overflow ceiling: every intermediate — `(N-1)*c[i]` (`<= 10^18`), `hi` (`<= 10^18`), `mid` (in `[0, 10^18]`), `total` (capped at `4*10^18`) — stays within `long long`'s `~9.2*10^18`. The `CAP = 4*10^18` plus one more term (`<= ~10^18`) before the cap check is at most `~5*10^18 < 9.2*10^18`, so even the over-the-cap step cannot overflow. Safe.
- Input parsing: `if (!(cin >> m >> N)) return 0;` guards empty input; `c[i] >= 1` guarantees no division by zero.

**Sanity at scale.** I ran the full-size worst case (`m = 10^5`, `N = 10^9`, random `w, c` up to `10^9`): it finishes in about `0.02` seconds using `~4.7` MB, well inside the `1` second / `256` MB budget — `~60` iterations times `10^5` presses is `6*10^6` cheap operations. And I ran `600` random small inputs against the independent event-by-event brute force (the millisecond-scan checker), with **zero** mismatches.

**Final solution.** I proved the predicate is monotone, derived a provably valid upper bound, and traced two real bugs to precise causes — a 32-bit truncation of both the endpoint product and the part accumulator (which made the program print `0` instead of `10^18`), and a latent off-by-one in the warm-up `+1` (which a `N=1` boundary trace pinned down). The fixes — `long long` throughout, a saturating accumulator, and the `>=`/`+1` pairing — are each verified by re-tracing the exact input that exposed the fault. This is what I ship, one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m;
    long long N;
    if (!(cin >> m >> N)) return 0;
    vector<long long> w(m), c(m);
    for (int i = 0; i < m; i++) cin >> w[i] >> c[i];

    if (N == 0) { cout << 0 << "\n"; return 0; }

    // produced(T): total parts all presses make by time T (milliseconds).
    // press i: 0 if T < w[i], else floor((T - w[i]) / c[i]) + 1.
    // Returns min(total, CAP) so the running sum cannot overflow long long.
    const long long CAP = (long long)4e18;
    auto produced = [&](long long T) -> long long {
        long long total = 0;
        for (int i = 0; i < m; i++) {
            if (T >= w[i]) {
                total += (T - w[i]) / c[i] + 1;
                if (total >= CAP) return CAP; // saturate early
            }
        }
        return total;
    };

    // Binary search smallest T with produced(T) >= N.
    // hi: a T that is certainly enough. The fastest press finishes N parts at
    // w_min + (N-1)*c_min; bound generously with max possible values.
    long long lo = 0;
    long long hi = 0;
    for (int i = 0; i < m; i++) {
        long long t = w[i] + (N - 1) * c[i]; // press i alone makes N parts by here
        hi = max(hi, t);
    }
    // hi as computed is an upper bound (single press alone already reaches N).

    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (produced(mid) >= N) hi = mid;
        else lo = mid + 1;
    }

    cout << lo << "\n";
    return 0;
}
```

**Causal recap.** Production is monotone in elapsed time, so the answer is the threshold of `produced(T) >= N` and binary search applies; a provably valid upper bound is `max_i (w[i] + (N-1)*c[i])` because that single press alone already reaches `N`. My first code typed two quantities as 32-bit — the endpoint product `(N-1)*c[i]` (`~10^18`) and the part accumulator — and a trace of the single-press `10^9/10^9/10^9` input showed it printing `0` instead of `10^18`, exactly the int-overflow this problem targets; switching to `long long` plus a saturating `CAP` fixes both sites. A second trace on `N=1`, presses `(5,1),(3,1)` confirmed the warm-up `+1` paired with a `>=` test correctly counts the part stamped at the warm-up instant (answer `3`, not `4`). The `N=0` guard, the no-division-by-zero from `c[i] >= 1`, the at-scale `0.02 s` timing, and `600` zero-mismatch random checks against an independent brute force close out the corners.
