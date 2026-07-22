// TIER: greedy
// The obvious first idea: ignore the jammer entirely and repeatedly place the
// next tower AT the demand point that gives the largest immediate nominal gain
// in sum w_i*min(count_i,t_i)/t_i. This piles redundant towers straight onto
// whichever cluster/needle has the richest marginal value -- exactly where a
// single published jammer position is planted to catch them all at once.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, K, M, L, r, R;
vector<ll> px, py, pw; vector<int> pt;

int main(){
    if (scanf("%d %d %d %d %d %d", &N, &K, &M, &L, &r, &R) != 6) return 0;
    px.resize(N); py.resize(N); pw.resize(N); pt.resize(N);
    for (int i = 0; i < N; i++)
        scanf("%lld %lld %lld %d", &px[i], &py[i], &pw[i], &pt[i]);
    // jammer positions are not read -- the naive greedy is jammer-blind.

    ll r2 = (ll)r * r;
    vector<vector<int>> coverList(N);
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++){
            ll dx = px[j] - px[i], dy = py[j] - py[i];
            if (dx * dx + dy * dy <= r2) coverList[j].push_back(i);
        }

    vector<int> cnt(N, 0);
    for (int step = 0; step < K; step++){
        double best = -1.0; int bestJ = 0;
        for (int j = 0; j < N; j++){
            double gain = 0.0;
            for (int i : coverList[j]){
                int c = cnt[i];
                if (c < pt[i]) gain += (double)pw[i] * (min(c + 1, pt[i]) - c) / (double)pt[i];
            }
            if (gain > best){ best = gain; bestJ = j; }
        }
        for (int i : coverList[bestJ]) if (cnt[i] < pt[i]) cnt[i]++;
        printf("%lld %lld\n", px[bestJ], py[bestJ]);
    }
    return 0;
}
