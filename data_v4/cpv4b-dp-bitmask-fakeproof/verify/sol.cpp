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
