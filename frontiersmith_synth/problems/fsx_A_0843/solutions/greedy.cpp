// TIER: greedy
// The obvious approach: sort blocks by wattage descending, then place each one
// to MAXIMIZE its distance from already-placed blocks (spread hot sources apart,
// checkerboard-style, to avoid clustering). Since nothing is placed yet when the
// heaviest block is chosen first, it defaults to the deck's geometric CENTER --
// the natural "start from the middle, spread outward" convention. This heuristic
// never looks at the cold rim at all, so on a "needle" instance (one dominant
// block among many tiny ones) it plants the hottest source at the worst possible
// spot.
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

    vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b){ return watt[a] > watt[b]; });

    vector<vector<int>> occ(H, vector<int>(W, -1));
    vector<int> outX(N + 1), outY(N + 1), outR(N + 1);
    vector<pair<double,double>> centroids;
    double cx = (W - 1) / 2.0, cy = (H - 1) / 2.0;

    for (int idx = 0; idx < N; idx++){
        int i = order[idx];
        bool have = false;
        double bestScore = 0; int bestX = 0, bestY = 0, bestR = 0;
        for (int r = 0; r < 4; r++){
            vector<pii> cells = rotateNorm(shape[i], r);
            int bw = 0, bh = 0;
            for (auto& c : cells){ bw = max(bw, c.first + 1); bh = max(bh, c.second + 1); }
            for (int y = 0; y + bh <= H; y++){
                for (int x = 0; x + bw <= W; x++){
                    bool ok = true;
                    for (auto& c : cells){
                        if (occ[y + c.second][x + c.first] != -1){ ok = false; break; }
                    }
                    if (!ok) continue;
                    double sx = 0, sy = 0;
                    for (auto& c : cells){ sx += x + c.first; sy += y + c.second; }
                    sx /= cells.size(); sy /= cells.size();
                    double score;
                    if (centroids.empty()){
                        double d = fabs(sx - cx) + fabs(sy - cy);
                        score = -d; // prefer the CENTER when nothing is placed yet
                    } else {
                        double mind = 1e18;
                        for (auto& pc : centroids)
                            mind = min(mind, fabs(sx - pc.first) + fabs(sy - pc.second));
                        score = mind; // maximize distance from everything already placed
                    }
                    if (!have || score > bestScore + 1e-9){
                        have = true; bestScore = score; bestX = x; bestY = y; bestR = r;
                    }
                }
            }
        }
        vector<pii> cells = rotateNorm(shape[i], bestR);
        double sx = 0, sy = 0;
        for (auto& c : cells){
            int ax = bestX + c.first, ay = bestY + c.second;
            occ[ay][ax] = i; sx += ax; sy += ay;
        }
        sx /= cells.size(); sy /= cells.size();
        centroids.push_back({sx, sy});
        outX[i] = bestX; outY[i] = bestY; outR[i] = bestR;
    }
    for (int i = 1; i <= N; i++) printf("%d %d %d\n", outX[i], outY[i], outR[i]);
    return 0;
}
