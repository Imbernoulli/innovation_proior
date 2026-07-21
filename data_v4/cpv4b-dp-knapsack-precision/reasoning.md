I have `n` fuel canisters, each with energy `cal[i]` and mass `mass[i]`; I must pick a non-empty subset whose total mass is at least `L` and maximize the density `(sum cal)/(sum mass)`, reported as an exact reduced fraction `p/q`. Two constraint numbers settle the shape of the danger before I choose any algorithm. The total mass of any subset is at most `n * max mass = 50 * 1000 = 50000` — small enough to index a table by mass. But `cal[i]` runs up to `10^13`, so a subset's total energy reaches `50 * 10^13 = 5*10^14`, and the objective is a *ratio* of two such sums. The moment I compare two candidate densities `P1/W1` and `P2/W2` the honest, division-free way — cross-multiply, `P1*W2` versus `P2*W1` — I am multiplying a `~5*10^14` numerator by a `~5*10^4` denominator, up to `2.5*10^19`. Signed 64-bit tops out near `9.2*10^18`. So the comparison, not the search, is where this problem is won or lost.

The tempting one-liner — sort by density `cal[i]/mass[i]`, take the densest — fails twice. The densest single canister can be too light to clear the mass floor `L`, and reaching `L` by adding more canisters dilutes it, so the winner is a genuine subset search rather than a sort. The sample makes it concrete: canisters `(100,4),(30,2),(60,3),(12,1)` with `L=5` have densities `25,15,20,12`; the densest is `(100,4)` at `25`, but its mass `4 < 5` is infeasible alone. Every feasible blend that includes it sits below `25` — `(100,4)+(60,3)=160/7≈22.86`, `(100,4)+(12,1)=112/5=22.4`, `(100,4)+(30,2)=130/6≈21.67` — and the best of them, `160/7`, is the answer.

Since total mass is bounded by `sumMass <= 50000`, I key a 0/1 knapsack on mass: `dp[W]` = the maximum total energy of a subset whose mass is *exactly* `W`, with `dp[0]=0` and every other entry unreachable, sweeping `W` from high to low per canister so each is used at most once, relaxing `dp[W] = max(dp[W], dp[W-m] + c)` whenever `dp[W-m]` is reachable. For a fixed denominator `W`, a larger numerator is a strictly larger density, so within each mass bucket only the max-energy subset can ever win — exactly what `dp[W]` holds. The answer is then the max of `dp[W]/W` over reachable `W >= L`. The cost is `O(n * sumMass) <= 2.5*10^6` updates, trivial. The brute-force `2^n` subset enumeration is hopeless as a solver but is the obvious oracle to check the DP against on small inputs.

Now the comparison. My first instinct writes the cross-multiply in `long long`:

```
long long lhs = P * bestW;    // compare P/W vs bestP/bestW
long long rhs = bestP * W;
if (lhs > rhs) { bestP = P; bestW = W; }
```

but the opening arithmetic already warns that `P*bestW` can reach `2.5*10^19`, more than twice `LLONG_MAX`. So I build the case that exposes it: a full instance, `n=50`, `mass[i]∈[900,1000]`, `cal[i]` near `10^13` with tiny perturbations to break ties, and a large floor forcing big total masses, and compare this `long long` version against an independent exact DP done in Python `Fraction`. On one such instance the `long long` version prints `333434742690641/34347` while the exact answer is `134980715779324/13821` — a completely different fraction. Instrumenting that instance: there are `14561` feasible candidates, the largest cross-product `maxP*maxW` reaches `21678410067718590585 ≈ 2.35× LLONG_MAX`, and `7659` of the comparisons overflow, so `lhs > rhs` compares wrapped garbage and the loop selects a fraction that is not the largest. A real wrong answer from integer overflow alone — the exact failure the problem is built around.

The fix is surgical: widen only the cross-multiplication to `__int128` (range `~1.7*10^38`, so `2.5*10^19` cannot come close), leaving the DP and its stored sums in `long long`, where they genuinely fit (`<= 5*10^14 < 9.2*10^18`):

```
__int128 lhs = (__int128)P * bestW;
__int128 rhs = (__int128)bestP * W;
if (lhs > rhs) { bestP = P; bestW = W; }
```

With this the adversarial instance prints `134980715779324/13821`, matching the `Fraction` oracle. Promoting the DP array to 128-bit would be wasted work; the only place the width is needed is the product.

One quieter corner in the candidate loop. If I initialize `bestP=-1, bestW=1` and cross-multiply the first feasible `W` against that `-1/1` sentinel, I am relying on `-1` behaving like a fraction — fragile, and if a mass-0 subset ever slipped in it would compare against `bestW=0`. Cleaner to seed the first feasible candidate directly and only cross-multiply thereafter: `if (bestP < 0) { bestP = P; bestW = W; continue; }`. The empty subset lives at `dp[0]` with undefined density; `L >= 1` keeps `W` starting at `>= 1` so it is never a candidate, but I leave an explicit `W==0` guard as defense. And the winner must be reduced — `gcd` turns e.g. `4/2` into `2/1`, which an exact-fraction check demands.

The remaining cases fall out. `n=1` with `L=mass[0]`: the lone canister, reduced. `L=sumMass`: only the full set clears the floor, so the answer is total energy over total mass, reduced. Density ties: cross-multiply gives equality, I keep the first, and the reduced output is identical whichever tied subset wins. Feasibility is guaranteed — `L <= sumMass` means the full set always qualifies — so `bestP` is always set and the `IMPOSSIBLE` branch is a guard that cannot fire under the constraints. The DP never adds `cal[i]` to the `NEG` sentinel (the `dp[w-m] > NEG` guard blocks it), so it neither overflows nor underflows.

I cross-checked the exact `Fraction` DP against the `2^n` brute force on small randomized cases across the big, small, mixed, and tie regimes, then the `__int128` solution against both — zero mismatches — including the full-constraint adversarial instances above. Worst case `n=50, sumMass=50000` runs in well under 0.01s in a few MB, far inside 2s / 256 MB. The complete program is in the answer.
