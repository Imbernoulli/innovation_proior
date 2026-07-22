// TIER: greedy
// The obvious "aim the loudest mirrors" recipe: install the K highest-gain
// candidate reflectors. This is exactly what a gain-driven or continuous-optics
// heuristic does -- it grabs the loud Euclidean-ellipse decoys. On the integer
// lattice their true arrival ticks are scattered and their parities random, so
// they yield isolated single-cell peaks instead of a tall stacked echo, and they
// ignore the coupling across listeners. It beats do-nothing but stays far below a
// solution that equalizes integer arrival ticks.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    ll G, sx, sy, g0; int M, N; ll K;
    if (scanf("%lld %lld %lld %lld %d %d %lld", &G,&sx,&sy,&g0,&M,&N,&K) != 7) return 0;
    for (int i = 0; i < M; i++){ ll a,b; scanf("%lld %lld",&a,&b); }
    vector<ll> cg(N+1); vector<int> idx(N);
    for (int j = 1; j <= N; j++){
        ll x,y,g,p; scanf("%lld %lld %lld %lld",&x,&y,&g,&p);
        cg[j] = g; idx[j-1] = j;
    }
    // sort candidate indices by gain descending, deterministic tie-break by index
    sort(idx.begin(), idx.end(), [&](int a, int b){
        if (cg[a] != cg[b]) return cg[a] > cg[b];
        return a < b;
    });
    ll c = min<ll>(K, N);
    printf("%lld\n", c);
    for (ll i = 0; i < c; i++) printf("%d%c", idx[i], (i+1==c?'\n':' '));
    if (c == 0) printf("\n");
    return 0;
}
