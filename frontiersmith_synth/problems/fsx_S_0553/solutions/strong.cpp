// TIER: strong
// The INSIGHT: the decision variable is a SPACE-TIME layout, not a layout plus a
// tour. Sacrifice packing density (leave thermal gaps), schedule the remaining
// adjacencies apart with a cooling order, and CUT ONLY parts that will survive --
// a doomed cut only heats its neighbours. Co-design placement + schedule + subset.
//
// Heuristic: for several spacing choices, drop value-desc parts onto a spaced grid,
// order them farthest-first, then simulate the true field and iteratively drop the
// worst offenders until every cut part survives; keep the highest-value result.
// (An optimal solver can pack tighter / pick a better subset -> headroom remains.)
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

    struct P { ll x, y; int id; };

    // farthest-first cooling order over a placed set
    auto coolingOrder = [&](vector<P>& placed) -> vector<int> {
        int K = (int)placed.size();
        vector<int> order; if (K == 0) return order;
        vector<char> taken(K, 0); vector<ll> mind(K, LLONG_MAX);
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
        return order;
    };

    // given an ordered list, return which indices scrap (true) under the true field
    auto scrapMask = [&](vector<P>& seq) -> vector<char> {
        int K = (int)seq.size();
        vector<char> scrap(K, 0);
        for (int j = 0; j < K; j++){
            ll idj = seq[j].id, heat = q[idj];
            for (int i = 0; i < j && heat <= THR; i++){
                ll idi = seq[i].id, dt = j - i;
                ll gp = gapDist(seq[i].x, seq[i].y, w[idi], h[idi],
                                seq[j].x, seq[j].y, w[idj], h[idj]);
                heat += q[idi] / (1 + DA * dt + DB * gp);
                ll warp = PW - PG * gp; if (warp > 0) heat += warp;
            }
            if (heat > THR) scrap[j] = 1;
        }
        return scrap;
    };

    ll bestVal = -1;
    vector<P> bestSeq;

    // try several absolute cell gaps -- including the dense zero-gap point (what the
    // obvious approach uses) up to gaps that fully clear the persistent warp. For
    // each, order farthest-first and drop parts that still scrap. Keep the best
    // surviving value: co-designing space (gap) + time (order) + subset (drops).
    vector<ll> gaps = { 0, 1, 2, 3, 4, 5, 6, 8 };
    for (ll SG : gaps){
        ll cellW = maxW + SG, cellH = maxH + SG;
        ll cols = N / cellW, rows = N / cellH;
        ll cap = cols * rows;
        if (cap <= 0) continue;
        vector<P> placed;
        for (int t = 0; t < (int)ord.size() && (ll)placed.size() < cap; t++){
            int id = ord[t];
            ll k = placed.size(), col = k % cols, row = k / cols;
            placed.push_back({col * cellW, row * cellH, id});
        }
        // order farthest-first, then iteratively drop scrapped parts
        vector<int> order = coolingOrder(placed);
        vector<P> seq; for (int idx : order) seq.push_back(placed[idx]);
        for (int round = 0; round < 6; round++){
            vector<char> scrap = scrapMask(seq);
            bool any = false;
            vector<P> keep;
            for (int j = 0; j < (int)seq.size(); j++){
                if (scrap[j]) any = true; else keep.push_back(seq[j]);
            }
            seq.swap(keep);
            if (!any) break;
        }
        ll val = 0; for (auto& p : seq) val += v[p.id];
        if (val > bestVal){ bestVal = val; bestSeq = seq; }
    }

    if (bestSeq.empty()){ printf("0\n"); return 0; }
    printf("%d\n", (int)bestSeq.size());
    for (auto& p : bestSeq) printf("%d %lld %lld\n", p.id, p.x, p.y);
    return 0;
}
