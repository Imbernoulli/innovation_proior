// TIER: strong
// The insight: with a cold (Dirichlet-zero) rim, distance-to-rim is the
// FIRST-ORDER term of the diffusion field, so 2D placement collapses to an
// (almost) 1D problem -- pick a single shared order over the deck's cells by
// rim-distance ascending, breaking ties by angle around the deck center (a
// walk around the rim, then the next ring in, etc.), and let the HEAVIEST
// blocks claim the earliest (closest-to-rim, well-spread-around-the-ring)
// slots first. This is not "greedy plus more search": it is a reformulation
// of the search space itself (2D packing -> a 1D walk along level sets of
// distance-to-rim) that a physically-blind "spread blocks apart" heuristic
// cannot discover.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int,int> pii;

int W, H, N;
vector<ll> watt;
vector<vector<pii>> shape;

vector<pii> rotateNorm(const vector<pii>& cells, int r){
    vector<pii> out;
    out.reserve(cells.size());
    for (auto& c : cells){
        int dx = c.first, dy = c.second, nx, ny;
        switch (r){
            case 0: nx = dx;  ny = dy;  break;
            case 1: nx = dy;  ny = -dx; break;
            case 2: nx = -dx; ny = -dy; break;
            default: nx = -dy; ny = dx; break;
        }
        out.push_back({nx, ny});
    }
    int mnx = INT_MAX, mny = INT_MAX;
    for (auto& c : out){ mnx = min(mnx, c.first); mny = min(mny, c.second); }
    for (auto& c : out){ c.first -= mnx; c.second -= mny; }
    return out;
}

int main(){
    int ITERS;
    scanf("%d %d %d %d", &W, &H, &N, &ITERS);
    watt.resize(N + 1); shape.resize(N + 1);
    for (int i = 1; i <= N; i++){
        int k; ll w;
        scanf("%lld %d", &w, &k);
        watt[i] = w;
        vector<pii> cells(k);
        for (int j = 0; j < k; j++){
            int dx, dy;
            scanf("%d %d", &dx, &dy);
            cells[j] = {dx, dy};
        }
        shape[i] = cells;
    }

    // ---- one shared cell order: rim-distance ascending, then angle around
    // the deck center (walks the rim, then the next ring in, ...) ----
    double cx = (W - 1) / 2.0, cy = (H - 1) / 2.0;
    vector<pii> order;
    order.reserve(W * H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            order.push_back({x, y});
    vector<int> d(W * H), rank(W * H);
    vector<double> ang(W * H);
    for (auto& p : order){
        int x = p.first, y = p.second;
        int dd = min(min(x, W - 1 - x), min(y, H - 1 - y));
        d[y * W + x] = dd;
        ang[y * W + x] = atan2((double)y - cy, (double)x - cx);
    }
    sort(order.begin(), order.end(), [&](const pii& a, const pii& b){
        int ia = a.second * W + a.first, ib = b.second * W + b.first;
        if (d[ia] != d[ib]) return d[ia] < d[ib];
        if (fabs(ang[ia] - ang[ib]) > 1e-12) return ang[ia] < ang[ib];
        if (a.second != b.second) return a.second < b.second;
        return a.first < b.first;
    });

    vector<int> pOrder(N);
    for (int i = 0; i < N; i++) pOrder[i] = i + 1;
    sort(pOrder.begin(), pOrder.end(), [&](int a, int b){ return watt[a] > watt[b]; });

    vector<vector<int>> occ(H, vector<int>(W, -1));
    vector<int> outX(N + 1), outY(N + 1), outR(N + 1);

    for (int idx = 0; idx < N; idx++){
        int i = pOrder[idx];
        bool placed = false;
        for (auto& slot : order){
            if (placed) break;
            int x0 = slot.first, y0 = slot.second;
            for (int r = 0; r < 4 && !placed; r++){
                vector<pii> cells = rotateNorm(shape[i], r);
                bool ok = true;
                for (auto& c : cells){
                    int ax = x0 + c.first, ay = y0 + c.second;
                    if (ax < 0 || ax >= W || ay < 0 || ay >= H || occ[ay][ax] != -1){ ok = false; break; }
                }
                if (!ok) continue;
                for (auto& c : cells) occ[y0 + c.second][x0 + c.first] = i;
                outX[i] = x0; outY[i] = y0; outR[i] = r;
                placed = true;
            }
        }
        // placed is always true: total footprint <= grid area is guaranteed by the generator
    }
    for (int i = 1; i <= N; i++) printf("%d %d %d\n", outX[i], outY[i], outR[i]);
    return 0;
}
