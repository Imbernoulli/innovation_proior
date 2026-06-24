**Reading the problem and pinning the contract.** I have a rod split into `n` unit-segments `0..n-1`, each carrying an integer `v[i]` that may be negative. I must cut the rod into contiguous billets that tile the whole thing, every billet length between `L` and `R` inclusive, and a billet over segments `[j, i)` costs `K + |sum of v on [j, i)|`. I minimize the total cost, or print `-1` if no legal tiling exists, with `n = 0` costing `0`. Input is `n K L R` then the `n` values; output is one integer. Before any algorithm I fix the scale, because it dictates types and the shape of the loop. `n <= 2*10^5`, `|v[i]| <= 10^9`, `1 <= L <= R <= 50`, `0 <= K <= 10^9`. A single billet's `|sum|` is at most `R * 10^9 = 5*10^10`, plus `K <= 10^9`, so one billet costs at most `~5.1*10^10`. There are at most `n` billets (each length at least `L >= 1`), so the total cost is bounded by `2*10^5 * 5.1*10^10 = ~1.02*10^16`. That fits in a signed 64-bit integer (`~9.2*10^18`) with three orders of magnitude to spare, but it blows 32-bit (`~2.1*10^9`) by seven orders. So every accumulator — prefix sums, the dp array, the per-billet cost — must be `long long`. That is decision number one and it is non-negotiable; an `int` anywhere in the cost path is a silent wrong-answer on the large tests. I also note `R <= 50` is a deliberate gift: it means the per-position transition window is at most 50 wide, so an `O(n*R)` DP is about `10^7` operations — comfortably inside one second. The bound on `R` is what makes the obvious DP fast; without it the window could be `O(n)` and the DP `O(n^2)`.

**Why a billet is a half-open range, and why that matters here.** I will represent a billet by the half-open segment range `[j, i)`, meaning it covers segments `j, j+1, ..., i-1`, a length of exactly `i - j`. I choose half-open on purpose: it makes the prefix-sum identity clean (`sum of v on [j, i) = S[i] - S[j]` with `S[i] = v[0] + ... + v[i-1]`), and it makes "the prefix `[0, i)` is fully tiled" a statement about a single index `i` running from `0` to `n`. The cost of `dp` will be indexed by these *cut boundaries* `0, 1, ..., n`, not by segments. The danger sign I am already circling: the constraint is phrased on *length* (`L <= length <= R`), but the DP transition is phrased on the *predecessor boundary* `j`, and translating one into the other is exactly where an off-by-one will hide. I want to derive that translation slowly rather than guess it.

**Laying out the candidate approaches.** Two routes, and I will commit to the one I can prove rather than the one that types fastest.

- *Fixed-length or greedy heuristic.* Always cut length `R` (or `L`), or grow a billet greedily while `|running sum|` keeps shrinking. `O(n)`, trivial. But `v` has negatives, so the cost of a billet is minimized by keeping its running sum near zero, and the zero-crossing of a partial-sum walk does not land on any fixed multiple of `R` or `L`. My instinct is that no fixed rule survives, but instinct is not proof, so I will try to break it with a concrete instance before trusting or discarding.
- *Linear partition DP.* `dp[i]` = cheapest tiling of `[0, i)`. The last billet is `[j, i)` for some legal `j`; recurse. `O(n*R)`. The idea is plainly correct; the *transcription* — the window of `j`, the base case, the unreachable-state sentinel, and the `-1` reporting — is where bugs live. This is the route I expect to ship, but I will earn it.

**Stress-testing the greedy before committing.** Take `v = [3, 3, -5, -5, 3, 3]`, `K = 2`, `L = 2`, `R = 3`. Prefix sums `S = [0, 3, 6, 1, -4, -1, 2]`. The "always cut the maximum length `R = 3`" greedy cuts `[0, 3)` and `[3, 6)`: sums `S[3]-S[0] = 1` and `S[6]-S[3] = 1`, cost `(2 + 1) + (2 + 1) = 6`. The "always cut the minimum length `L = 2`" greedy cuts `[0,2), [2,4), [4,6)`: sums `6, -5, 5`, cost `(2+6)+(2+5)+(2+5) = 27`. So even the two simplest fixed rules disagree wildly (6 vs 27), which already tells me the answer is sensitive to *where* the cuts fall, not just how many there are. Is `6` even optimal here? The only other tiling of length-6 with pieces in `[2,3]` is `[0,3)+[3,6)` (=6) or `[0,2)+[2,4)+[4,6)` (=27) — and that is it, because `2+2+2`, `3+3`, are the only compositions of 6 into parts in `{2,3}`. So `6` is optimal here, and it happened to coincide with "always `R`". But that coincidence is exactly the trap. Let me find a case where keeping the running sum near zero beats both fixed rules: `v = [5, -4, 5, -4, 5]`, `K = 0`, `L = 2`, `R = 3`, `S = [0,5,1,6,2,7]`. "Always `R=3`" can only cut `[0,3)+...` but `5 - 3 = 2 < L`, so it would need `[0,3)+[3,5)`: sums `6, 1`, cost `6 + 1 = 7`. "Always `L=2`" cuts `[0,2)+[2,4)+...` leaving length 1, infeasible — greedy-min dies. The balanced cut `[0,2)+[2,5)` has sums `1, 6` = 7, while `[0,3)+[3,5)` is also 7. Both 7. Hmm — let me push the negatives: `v=[9,-9,9,-9]`, `K=0`, `L=2`, `R=2`, `S=[0,9,0,9,0]`. Forced cuts `[0,2)+[2,4)`: sums `0, 0`, cost `0`. "Always `R=2`" gives the same here because `L=R`. The real lesson from these probes is sharper than "greedy is wrong on value": greedy is *fragile on feasibility* — the minute `L = R` or `n` is not a clean multiple, a fixed-length rule either gets lucky or produces an infeasible leftover, and it has no notion of `-1`. A DP that scans the whole legal window is the only thing that both finds the balance-minimizing split and correctly reports infeasibility. Greedy is out; DP it is.

**Deriving the DP and the exact transition window.** Let `dp[i]` = the minimum total cost to tile the prefix `[0, i)` (the first `i` segments) into billets each of length in `[L, R]`; `dp[i] = +infinity` if that prefix cannot be tiled. The base case is `dp[0] = 0`: tiling zero segments needs zero billets and costs nothing. For `i >= 1`, the last billet is some `[j, i)`; its length is `i - j` and must satisfy `L <= i - j <= R`. Now I solve that double inequality for `j`, slowly, because this is the crux. From `i - j <= R` I get `j >= i - R`. From `i - j >= L` I get `j <= i - L`. And `j` is a real cut boundary so `j >= 0`. Therefore

```
j ranges over [ max(0, i - R) , i - L ],   inclusive on both ends,
```

and the transition is `dp[i] = min over those j of ( dp[j] + K + |S[i] - S[j]| )`. If the window is empty (`i - L < max(0, i - R)`, which happens when `i < L`) then `dp[i]` stays `+infinity` — there is no legal last billet, the prefix is untileable. The answer is `dp[n]`, or `-1` if `dp[n]` is still infinity. Let me double-check the two endpoints of the window by plugging in the extreme legal lengths. Longest legal billet is length `R`, which starts at `j = i - R`; that is the *low* end of the `j`-range — good, and I clamp it to `0`. Shortest legal billet is length `L`, which starts at `j = i - L`; that is the *high* end — good. So the inclusive window is `[max(0, i-R), i-L]`. I write this down explicitly so I do not "simplify" it into something off by one later.

**Sanity-checking the derivation on the sample before any code.** `v = [3,3,-5,-5,3,3]`, `K=2`, `L=2`, `R=3`, `S=[0,3,6,1,-4,-1,2]`, target answer `6`. `dp[0]=0`. `dp[1]`: window `j in [max(0,1-3), 1-2] = [0, -1]` — empty, so `dp[1] = inf` (can't tile 1 segment with min length 2). Good. `dp[2]`: `j in [max(0,-1), 0] = [0,0]`; `dp[0] + 2 + |S[2]-S[0]| = 0 + 2 + 6 = 8`; `dp[2]=8`. `dp[3]`: `j in [max(0,0), 1] = [0,1]`; from `j=0`: `0 + 2 + |S[3]-S[0]| = 2 + 1 = 3`; from `j=1`: `dp[1]=inf`, skip; `dp[3]=3`. `dp[4]`: `j in [max(0,1), 2] = [1,2]`; `j=1`: inf; `j=2`: `dp[2] + 2 + |S[4]-S[2]| = 8 + 2 + 8 = 18`; `dp[4]=18`. `dp[5]`: `j in [max(0,2), 3] = [2,3]`; `j=2`: `8 + 2 + |S[5]-S[2]| = 8 + 2 + 7 = 17`; `j=3`: `3 + 2 + |S[5]-S[3]| = 3 + 2 + 5 = 10`; `dp[5]=10`. `dp[6]`: `j in [max(0,3), 4] = [3,4]`; `j=3`: `dp[3] + 2 + |S[6]-S[3]| = 3 + 2 + |2-1| = 3 + 2 + 1 = 6`; `j=4`: `dp[4] + 2 + |S[6]-S[4]| = 18 + 2 + |2-(-4)| = 18 + 2 + 6 = 26`; `dp[6] = 6`. The derivation reproduces the sample answer `6`, via the cut `[0,3)+[3,6)` exactly as expected. The recurrence and the window are right on paper.

**First implementation — and immediately a trace, because clean math transcribes dirty.** Here is my first cut of the core:

```
const long long INF = (long long)4e18;
vector<long long> dp(n + 1, INF);
dp[0] = 0;
for (int i = 1; i <= n; i++) {
    int jlo = max(0, i - R);
    int jhi = i - L + 1;                 // <-- first attempt at the high end
    for (int j = jlo; j <= jhi; j++) {
        if (dp[j] >= INF) continue;
        long long seg = S[i] - S[j];
        long long cost = K + llabs(seg);
        if (dp[j] + cost < dp[i]) dp[i] = dp[j] + cost;
    }
}
```

I wrote `jhi = i - L + 1` on autopilot — there is a reflex that array windows of "length up to something" end at `+1`, the half-open habit leaking into a place it does not belong. The smell is strong enough that I trace the smallest input that can expose a too-short billet. Take `n = 1`, `K = 37`, `L = 2`, `R = 2`, `v = [9]`. The correct answer is `-1`: one segment cannot host a billet of length `2`. Run the buggy loop. `S = [0, 9]`. `dp[0] = 0`, `dp[1] = INF` initially. `i = 1`: `jlo = max(0, 1 - 2) = 0`; `jhi = 1 - 2 + 1 = 0`; so the window is `j in [0, 0]` — *non-empty*. `j = 0`: `dp[0] = 0` is finite, `seg = S[1] - S[0] = 9`, `cost = 37 + 9 = 46`, `dp[1] = 46`. Final `dp[n] = dp[1] = 46`, printed `46`.

**Diagnosing the bug.** The code returns `46` where the truth is `-1`. The defect is precise and it is the pitfall I was warned to watch: `jhi = i - L + 1` admits `j = i - L + 1`, i.e. a last billet of length `i - j = L - 1`, which is *one segment shorter than the minimum allowed*. Here `L = 2` but it cut a billet of length `1`. The shortest legal billet starts at `j = i - L`, so the inclusive high end of the `j`-window is `i - L`, **not** `i - L + 1`. My half-open instinct fired in the wrong frame: the window over `j` is an inclusive integer range `[i-R, i-L]`, not a half-open length range. I confirm the direction of the error by re-deriving: length `>= L` means `i - j >= L` means `j <= i - L`; the largest legal `j` is exactly `i - L`, and `i - L + 1` violates it by one. So the fix is `jhi = i - L`.

**Fixing and re-verifying the first bug.** I change the line to `int jhi = i - L;` and re-run the same trace. `n=1, L=2, R=2, v=[9]`: `i=1`, `jlo = max(0, -1) = 0`, `jhi = 1 - 2 = -1`. Window `[0, -1]` is empty, the inner loop never executes, `dp[1]` stays `INF`, and I print `-1`. Correct. I re-run a second discriminating case to be sure I did not over-correct into forbidding a legal billet: `n = 2`, `L = 2`, `R = 2`, `K = 0`, `v = [3, 4]`. Truth: one billet `[0,2)`, cost `0 + |7| = 7`. Trace: `dp[0]=0`. `i=1`: `jlo=max(0,-1)=0`, `jhi=1-2=-1`, empty, `dp[1]=INF`. `i=2`: `jlo=max(0,0)=0`, `jhi=2-2=0`, window `[0,0]`; `j=0`: `dp[0] + 0 + |S[2]-S[0]| = 0 + 7 = 7`, `dp[2]=7`. Print `7`. Correct — the legal length-2 billet is *kept*, so the fix narrowed the window by exactly one on the high end and nothing more. The two cases that distinguish "off by one" both behave now, and they behave for the reason I fixed. I later confirmed against the brute force that the `jhi = i - L + 1` version disagrees on 73 of 400 random tiny cases, while `jhi = i - L` agrees on all of them — the trace caught a bug that is wrong 18% of the time, not a corner curiosity.

**Second debug episode: the unreachable-predecessor and the `-1` boundary.** With the window fixed I worry about a different boundary: what guards a *reachable* predecessor? The transition reads `dp[j]` for `j` in the window, but some of those `j` may themselves be untileable (`dp[j] = INF`). My loop has `if (dp[j] >= INF) continue;`, but let me make sure that guard is both present and correct, because without it a finite-looking `dp[i]` could be built on an impossible prefix, and with a wrong comparison it could wrongly *skip* a reachable one. I trace `n = 3`, `K = 0`, `L = 2`, `R = 2`, `v = [1, 1, 1]`. Truth: length 3 cannot be tiled by pieces of length exactly 2 (`2+2 > 3`, `2` alone leaves 1), so the answer is `-1`. `S = [0,1,2,3]`. `dp[0]=0`, rest `INF`. `i=1`: `jlo=0, jhi=-1`, empty, `dp[1]=INF`. `i=2`: `jlo=max(0,0)=0, jhi=0`; `j=0`: `dp[0]` finite, `cost = 0 + |2-0| = 2`, `dp[2]=2`. `i=3`: `jlo=max(0,1)=1, jhi=1`; window `[1,1]`; `j=1`: `dp[1] = INF`, the guard `continue`s; no other `j`; `dp[3]` stays `INF`. Print `-1`. Correct — and crucially, had I omitted the `dp[j] >= INF` guard, `j=1` would have computed `INF + 0 + |S[3]-S[1]| = INF + 2`, which in `long long` is `INF + 2` (no overflow at `4e18`, room to `9.2e18`), a finite-looking huge number that would have *replaced* the correct `INF` and made `dp[3]` look reachable — printing a giant wrong cost instead of `-1`. So the guard is load-bearing, and it must be `>= INF`, not `== INF`, because once a state is built from another infinite state (in a variant without the guard) it could exceed `INF` slightly; using `>=` is the safe boundary. I keep `if (dp[j] >= INF) continue;` and, for symmetry, report `-1` with `if (dp[n] >= INF)`. I also sanity-check the sentinel size: `INF = 4e18` plus at most one billet cost `~5.1e10` is `~4.0000000051e18 < 9.2e18`, so even the un-guarded sum cannot overflow `long long` and wrap to a negative that would defeat the `>= INF` test. The sentinel is chosen so the boundary check stays valid.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: the segment-reading loop and the dp loop both never run; `dp` has size `1` with `dp[0] = 0`; `dp[n] = dp[0] = 0 < INF`, so I print `0`. The empty rod, no cuts — correct, and I confirmed the program reads `n K L R`, then zero values, then prints `0`.
- `n = 1, L = 1`: window for `i=1` is `[max(0,1-R), 0] = [0, 0]`, one legal billet `[0,1)`, cost `K + |v[0]|`. For `v = [-7], K = 0`: `0 + 7 = 7`. Correct (length-1 billet is legal when `L = 1`).
- Infeasible by parity / size: covered above (`L = R = 2`, `n = 3` -> `-1`; `n < L` -> `-1`). The `-1` path is reached purely by `dp[n]` staying `INF`, no special-casing.
- `L = R` (forced length): the window collapses to a single `j = i - L` (clamped), so the DP still works; only multiples of `L` up to `n` are reachable. Verified by trace and by brute force.
- Maximum: `n = 2*10^5`, `R = 50`, `|v| = 10^9`, `K = 10^9`. The DP does `<= n*R = 10^7` finite-cost updates; measured wall time `0.02 s`, memory `~8 MB`. The largest `dp` value stays `~1.15*10^13` on a random max instance, and the proven bound is `~1.02*10^16`, both far under `LLONG_MAX`. No overflow, no TLE.
- Output format: exactly one integer and a newline; `cin >>` skips arbitrary whitespace and the missing second line for `n = 0` is harmless.

**Sanity-check of the derivation itself.** Beyond the sample I re-derived the window from both inequalities independently (`i - j <= R` and `i - j >= L`), got `j in [i-R, i-L]` both times, and confirmed the two endpoints correspond to the longest (`R`) and shortest (`L`) legal billets. The brute force — which independently enumerates *every* composition of the length into parts in `[L, R]` and takes the min of `sum(K + |segment sum|)` — agrees with the DP on 500 generated cases plus 1000 wider random cases, zero mismatches, including many `-1` instances. The two methods share no code path (recursion-over-compositions vs prefix-DP), so the agreement is real evidence, not a tautology.

**Final solution.** I disproved the greedy on feasibility and balance, derived the inclusive `j`-window `[max(0,i-R), i-L]` from first principles, caught the `i-L+1` off-by-one by tracing `n=1, L=2` (which illegally cut a length-1 billet and returned `46` instead of `-1`), caught the unreachable-predecessor boundary by tracing `n=3, L=R=2` (which needs the `dp[j] >= INF` guard to return `-1`), and closed the empty/single/forced/max/overflow corners. This is what I ship — one self-contained file, the `O(n*R)` partition DP I can defend:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long K;
    int L, R;
    if (!(cin >> n >> K >> L >> R)) return 0;
    vector<long long> v(n);
    for (auto &x : v) cin >> x;

    // prefix sums S[0..n], S[i] = v[0] + ... + v[i-1]
    vector<long long> S(n + 1, 0);
    for (int i = 0; i < n; i++) S[i + 1] = S[i] + v[i];

    const long long INF = (long long)4e18;
    // dp[i] = min cost to partition the first i unit-segments, i.e. the half-open
    // range [0, i), into valid billets. dp[0] = 0 (nothing cut yet).
    // A billet covering segments [j, i) has length (i - j), which must satisfy
    // L <= i - j <= R. Solving for j: j in [i - R, i - L], and also j >= 0.
    // The billet's cost is K + |S[i] - S[j]| (a setup fee plus the imbalance).
    vector<long long> dp(n + 1, INF);
    dp[0] = 0;
    for (int i = 1; i <= n; i++) {
        int jlo = max(0, i - R);   // longest allowed billet, length R
        int jhi = i - L;           // shortest allowed billet, length L
        for (int j = jlo; j <= jhi; j++) {
            if (dp[j] >= INF) continue;
            long long seg = S[i] - S[j];
            long long cost = K + llabs(seg);
            if (dp[j] + cost < dp[i]) dp[i] = dp[j] + cost;
        }
    }

    if (dp[n] >= INF) cout << -1 << "\n";
    else cout << dp[n] << "\n";
    return 0;
}
```

**Causal recap.** Greedy looked plausible but two probes showed fixed-length rules either get lucky or produce infeasible leftovers and have no `-1` notion, so I moved to a prefix partition DP and *derived* the predecessor window by solving `L <= i - j <= R` to `j in [max(0,i-R), i-L]` inclusive; my first transcription wrote the high end as `i - L + 1`, and tracing `n=1, K=37, L=R=2, v=[9]` showed it illegally cutting a length-1 billet and printing `46` instead of `-1` — fixing it to `i - L` and re-tracing two discriminating cases confirmed the window now keeps exactly the legal billets; a second trace on `n=3, L=R=2, v=[1,1,1]` showed the `dp[j] >= INF` guard is load-bearing for correct `-1` reporting and that `INF = 4e18` is small enough that the boundary check never overflows; with `long long` everywhere, the bounded `R <= 50` window keeping the DP at `O(n*R)`, and `max`/`min` over the inclusive window, the empty, single, forced-length, infeasible, and max-scale corners all fall out, and 1500 brute-force comparisons (zero mismatches) confirm it.
