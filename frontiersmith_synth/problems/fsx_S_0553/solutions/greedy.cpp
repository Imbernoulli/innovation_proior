// TIER: greedy
// The OBVIOUS approach: maximize placed value -- drop every part onto a TIGHT grid
// (cell = maxW x maxH, essentially touching), highest-value first, then apply a
// farthest-first COOLING cut order so recently-cut neighbours are far apart in time,
// and cut everything. TRAP: the persistent warp a touching neighbour leaves is
// time-independent, so no schedule removes it -- interior parts (2+ touching
// neighbours) warp and scrap, and each doomed cut still heats its neighbours. This
// is exactly the "zero-gap" corner of the space-time trade-off, cut blindly.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static ll gapDist(ll xi, ll yi, ll wi, ll hi, ll xj, ll yj, ll wj, ll hj){
    ll gx = max(0LL, max(xi - (xj + wj), xj - (xi + wi)));
    ll gy = max(0LL, max(yi - (yj + hj), yj - (yi + hi)));
    return gx + gy;
}

int main(){
    ll N, M, DA, DB, PW, PG, THR;
    if (!(cin >> N >> M >> DA >> DB >> PW >> PG >> THR)) return 0;
    vector<ll> w(M + 1), h(M + 1), v(M + 1), q(M + 1);
    ll maxW = 1, maxH = 1;
    for (int i = 1; i <= M; i++){
        cin >> w[i] >> h[i] >> v[i] >> q[i];
        maxW = max(maxW, w[i]); maxH = max(maxH, h[i]);
    }
    vector<int> ord(M);
    for (int i = 0; i < M; i++) ord[i] = i + 1;
    sort(ord.begin(), ord.end(), [&](int a, int b){ return v[a] > v[b]; });

    // tight grid: cell = maxW x maxH (zero gap between cells)
    ll cellW = maxW, cellH = maxH;
    ll cols = max<ll>(1, N / cellW), rows = max<ll>(1, N / cellH);
    ll cap = cols * rows;
    struct P { ll x, y; int id; };
    vector<P> placed;
    for (int t = 0; t < (int)ord.size() && (ll)placed.size() < cap; t++){
        ll k = placed.size(), col = k % cols, row = k / cols;
        placed.push_back({col * cellW, row * cellH, ord[t]});
    }
    int K = (int)placed.size();
    if (K == 0){ printf("0\n"); return 0; }

    // farthest-first cooling order (be clever about the schedule, keep dense pack)
    vector<int> order; vector<char> taken(K, 0); vector<ll> mind(K, LLONG_MAX);
    int cur = 0; order.push_back(cur); taken[cur] = 1;
    auto relax = [&](int c){
        for (int i = 0; i < K; i++) if (!taken[i]){
            ll d = gapDist(placed[c].x, placed[c].y, w[placed[c].id], h[placed[c].id],
                           placed[i].x, placed[i].y, w[placed[i].id], h[placed[i].id]);
            if (d < mind[i]) mind[i] = d;
        }
    };
    relax(cur);
    for (int step = 1; step < K; step++){
        int best = -1; ll bestd = -1;
        for (int i = 0; i < K; i++) if (!taken[i] && mind[i] > bestd){ bestd = mind[i]; best = i; }
        cur = best; taken[cur] = 1; order.push_back(cur); relax(cur);
    }
    printf("%d\n", K);
    for (int idx : order) printf("%d %lld %lld\n", placed[idx].id, placed[idx].x, placed[idx].y);
    return 0;
}
