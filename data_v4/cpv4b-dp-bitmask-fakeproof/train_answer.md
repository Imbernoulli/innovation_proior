**Problem.** A station has `N` modules (`N <= 16`). Each of `M` crews services exactly one fixed module-set `mask[j]` (a bitmask) for cost `c[j]`. Choose crews whose module-sets are pairwise disjoint and union to all of `{0,..,N-1}` (an exact partition), minimizing total cost; print `-1` if impossible. Read `N M` then `M` lines `mask c` from stdin; print the minimum cost.

**Key idea — subset DP over module-sets.** Let `cost1[s]` be the cheapest single crew whose mask equals `s` (else `INF`); this reduction also makes "each crew at most once" automatic, since disjoint masks never repeat. Let `best[m]` be the cheapest exact cover of set `m`. Then `best[0] = 0`, and for `m != 0` fix `low = lowest set bit of m` (the lowest uncovered module must be serviced by exactly one crew) and split:

- `best[m] = min over submasks s of m with (s & low) != 0 and cost1[s] < INF of  cost1[s] + best[m ^ s]`.

The answer is `best[2^N - 1]` (or `-1` if it stayed `INF`). Restricting `s` to contain `low` counts each partition once and keeps the per-mask work to submasks of `m`.

**Pitfalls.**
1. *Mis-estimating the cost (the real trap).* The work is `Σ_m 2^popcount(m)` pairs. The tempting "average popcount is `N/2`, so total ≈ `2^N · 2^(N/2) = 2^(1.5N)`" illegally replaces `E[2^X]` by `2^E[X]`; since `2^X` is convex this *under*-counts. The true closed form is `3^N` — each bit is independently out of `m`, in `m`-not-`s`, or in both (three states). Verify numerically: `Σ_m 2^popcount(m)` equals `81` at `N=4` and `6561` at `N=8`, exactly `3^N`, and strictly above the `2^(1.5N)` guess (`64`, `4096`). At `N=16` the real work is `3^16 ≈ 4.3·10^7`, fine for 2 s — but never assert the bound without this check.
2. *Unguarded INF addition.* Guard `best[m ^ s] < INF` (and `cost1[s] < INF`) **before** summing; otherwise impossible subproblems contribute a huge finite value and an uncoverable mask gets a bogus cost instead of staying `INF`.
3. *Overflow and `N = 0`.* Totals reach `1.6·10^10`, so use `long long`. For `N = 0` the empty partition costs `0`; special-case it.

**Edge cases.** `N = 0` -> `0`; roster that cannot tile the modules -> `-1`; duplicate masks -> cheapest wins via `cost1 = min(...)`; singletons-only roster forced to cover everything -> sum of singletons; zero-cost crews participate normally.

**Complexity.** `O(3^N)` time (proved and numerically checked), `O(2^N)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
  Relay Crews — exact-partition minimum cost via submask DP.

  N modules (0..N-1), full set FULL = (1<<N)-1.
  M crews; crew j covers module-set mask[j] (mask[j] != 0) at cost c[j] (>=0).
  Each crew may be used at most once. We must partition FULL into the masks of the
  chosen crews (every module serviced exactly once -> chosen masks pairwise disjoint
  and union = FULL). Minimize total cost. If impossible, print -1.

  best[m] = minimum total cost to EXACTLY cover module-set m using a subset of crews
            (each crew at most once, chosen masks pairwise disjoint, union == m).

  We first reduce crews: for each achievable single-crew mask keep the cheapest cost,
  cost1[m] = min c[j] over crews j with mask[j] == m (INF if none).
  Then a standard subset DP:
     best[0] = 0
     best[m] = min over nonempty submasks s of m with cost1[s] < INF of
               cost1[s] + best[m ^ s].
  To avoid double counting / ordering issues we fix the lowest set bit of m to lie in s,
  i.e. we only split off submasks s that contain lowbit(m). That makes each partition
  counted once and keeps the per-mask work to submasks-of-m.
*/

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if(!(cin >> n >> m)) return 0;
    const int FULL = (1<<n) - 1;
    const long long INF = (long long)4e18;

    vector<long long> cost1(1<<n, INF);
    for(int j=0;j<m;j++){
        int mk; long long c;
        cin >> mk >> c;
        // mk given as an integer in [1, 2^n - 1]; clamp guard not needed per contract
        if(mk>=1 && mk<=FULL){
            cost1[mk] = min(cost1[mk], c);
        }
    }

    if(n==0){
        // FULL == 0; empty partition has cost 0.
        cout << 0 << "\n";
        return 0;
    }

    vector<long long> best(1<<n, INF);
    best[0] = 0;
    for(int mask=1; mask<=FULL; mask++){
        int low = mask & (-mask);           // lowest set bit of mask
        int rest = mask ^ low;              // bits above the forced lowest bit
        long long bm = INF;
        // enumerate submasks s of mask that CONTAIN low: s = low | (submask of rest)
        int sub = rest;
        while(true){
            int s = low | sub;              // contains low, s subset of mask
            if(cost1[s] < INF){
                long long other = best[mask ^ s];
                if(other < INF){
                    bm = min(bm, cost1[s] + other);
                }
            }
            if(sub==0) break;
            sub = (sub-1) & rest;
        }
        best[mask] = bm;
    }

    long long ans = best[FULL];
    if(ans >= INF) cout << -1 << "\n";
    else cout << ans << "\n";
    return 0;
}
```
