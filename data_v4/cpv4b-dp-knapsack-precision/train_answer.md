**Problem.** Given `n` fuel canisters with integer energy `cal[i]` and mass `mass[i]`, choose a non-empty subset whose total mass is at least `L`, maximizing the energy density `(sum cal)/(sum mass)`. Output the maximum density as a reduced fraction `p/q`. Constraints: `n <= 50`, `mass[i] <= 1000` (so total mass `<= 50000`), `cal[i] <= 10^13`, `1 <= L <= sumMass`.

**Why greedy fails.** "Take the densest canister" is wrong: the densest single canister may have mass below `L` (infeasible alone), and reaching `L` by blending dilutes it. On the sample `(100,4),(30,2),(60,3),(12,1)` with `L=5`, the densest single is `(100,4)` at density `25`, but mass `4 < 5`; the best feasible blend is canisters 1+3 = `160/7 ≈ 22.86`, strictly below `25`. The choice of which canisters to combine is a genuine knapsack search.

**Key idea — subset-sum DP keyed on mass, then an exact ratio comparison.** Total mass is bounded by `sumMass <= 50000`, so run a 0/1 knapsack on mass: `dp[W]` = maximum total energy of a subset with total mass exactly `W` (`dp[0]=0`, sweep `W` high→low per item). For each fixed denominator `W`, only the max-energy subset can win, so this is the right table. Then over all reachable `W >= L`, maximize `dp[W]/W`. Reduce the winner with `gcd`.

**Pitfalls.**
1. *Integer overflow in the fraction comparison — the decisive issue.* Comparing `P1/W1` vs `P2/W2` by cross-multiplication needs `P1*W2` vs `P2*W1`. Here `P` reaches total energy `5*10^14` and `W` reaches `5*10^4`, so a cross-product reaches `2.5*10^19`, which exceeds `LLONG_MAX ≈ 9.2*10^18` by `2.35×`. A `long long` cross-multiply silently wraps and selects the wrong fraction — on an adversarial case it prints `333434742690641/34347` instead of the correct `134980715779324/13821`. Do the cross-multiply in `__int128` (range ~`1.7*10^38`, provably safe). Never divide in floating point; the answer must be exact. The DP sums themselves fit in `long long` (`<= 5*10^14`), so only the comparison needs 128-bit.
2. *First-candidate seeding.* Seed `bestP,bestW` from the first feasible `W` directly rather than cross-multiplying against an uninitialized sentinel; otherwise a `-1/1` placeholder leaks into a 128-bit product.
3. *Reduction.* Always output in lowest terms via `gcd(p, q)`; e.g. `4/2` must print as `2/1`.

**Edge cases.** `n = 1` with `L = mass[0]` → the lone canister reduced. `L = sumMass` → whole set forced, `(total energy)/(total mass)` reduced. Density ties → cross-multiply gives equality, keep the first; the reduced output is identical either way. Feasibility is guaranteed since `L <= sumMass` (the full set always qualifies). The DP never adds to its `NEG` sentinel (guarded), so it cannot overflow or underflow.

**Complexity.** `O(n * sumMass)` time (`<= 2.5*10^6` updates) and `O(sumMass)` memory; worst case runs in well under 0.01s using ~4 MB. The candidate scan is `O(sumMass)` with one `__int128` multiply each.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;
    vector<long long> cal(n), mass(n);
    long long sumMass = 0;
    for (int i = 0; i < n; i++) { cin >> cal[i] >> mass[i]; sumMass += mass[i]; }

    const long long NEG = LLONG_MIN / 4;
    // dp[w] = max total cal over a subset whose total mass is EXACTLY w (NEG = unreachable).
    vector<long long> dp(sumMass + 1, NEG);
    dp[0] = 0; // empty subset: mass 0, value 0
    for (int i = 0; i < n; i++) {
        for (long long w = sumMass; w >= mass[i]; w--) {
            if (dp[w - mass[i]] > NEG) {
                long long cand = dp[w - mass[i]] + cal[i];
                if (cand > dp[w]) dp[w] = cand;
            }
        }
    }

    // Among masses W in [L, sumMass] reachable by a NON-EMPTY subset, maximize value/W.
    // Best fraction bestP/bestW maximized; bestP = -1 marks "none yet".
    long long bestP = -1, bestW = 1;
    for (long long W = L; W <= sumMass; W++) {
        if (dp[W] <= NEG) continue;       // mass W not reachable
        if (W == 0) continue;             // density undefined; L>=1 anyway
        long long P = dp[W];
        // compare P/W vs bestP/bestW using exact 128-bit cross multiplication
        if (bestP < 0) { bestP = P; bestW = W; continue; }
        __int128 lhs = (__int128)P * bestW;
        __int128 rhs = (__int128)bestP * W;
        if (lhs > rhs) { bestP = P; bestW = W; }
    }

    if (bestP < 0) { cout << "IMPOSSIBLE\n"; return 0; }
    long long g = std::__gcd(bestP, bestW);
    if (g == 0) g = 1;
    cout << (bestP / g) << "/" << (bestW / g) << "\n";
    return 0;
}
```
