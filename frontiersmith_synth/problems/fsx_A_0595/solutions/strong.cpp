// TIER: strong
// Insight: the value of the whole plan is set by the LEAF-SIZE DISTRIBUTION of the
// tree. So choose each band's HEIGHT and its MIX of widths to maximize value-DENSITY
// under the global caps, instead of chasing per-piece value. For every candidate
// height, fill a band by a density-ordered (value per unit area) bounded fill across
// the width; commit the height whose band yields the most value; repeat down the
// sheet. This diversifies to match the demand distribution and ignores shiny needles
// that waste a band. (Still a per-band greedy, not a true 2D knapsack -> headroom.)
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int W, H, K, D;
vector<int> dw, dh, dv, dc;

void emitBand(int w, int bh, const vector<int>& pieces, int idx){
    if (idx >= (int)pieces.size()){ printf("0 0\n"); return; }
    int t = pieces[idx], pw = dw[t];
    if (pw == w && idx == (int)pieces.size() - 1){ printf("0 %d\n", t); return; }
    printf("1 0 %d\n", pw);
    printf("0 %d\n", t);
    emitBand(w - pw, bh, pieces, idx + 1);
}
void emitStack(int Hrem, const vector<pair<int,vector<int>>>& bands, int idx){
    if (idx >= (int)bands.size()){ printf("0 0\n"); return; }
    int bh = bands[idx].first;
    if (bh == Hrem && idx == (int)bands.size() - 1){ emitBand(W, bh, bands[idx].second, 0); return; }
    printf("1 1 %d\n", bh);
    emitBand(W, bh, bands[idx].second, 0);
    emitStack(Hrem - bh, bands, idx + 1);
}

int main(){
    scanf("%d %d %d %d", &W, &H, &K, &D);
    dw.assign(D + 1, 0); dh.assign(D + 1, 0); dv.assign(D + 1, 0); dc.assign(D + 1, 0);
    for (int i = 1; i <= D; i++) scanf("%d %d %d %d", &dw[i], &dh[i], &dv[i], &dc[i]);

    vector<ll> cap(D + 1);
    for (int i = 1; i <= D; i++) cap[i] = dc[i];

    // distinct heights present
    set<int> hs;
    for (int i = 1; i <= D; i++) if (dh[i] <= H && dw[i] <= W) hs.insert(dh[i]);

    int Hrem = H;
    vector<pair<int,vector<int>>> bands;
    while (Hrem > 0){
        int bestH = -1; ll bestVal = 0; vector<int> bestPieces; vector<pair<int,int>> bestUse;
        for (int hcur : hs){
            if (hcur > Hrem) continue;
            // candidate types of this exact height with remaining cap
            vector<int> cand;
            for (int i = 1; i <= D; i++)
                if (dh[i] == hcur && dw[i] <= W && cap[i] > 0) cand.push_back(i);
            if (cand.empty()) continue;
            // density order: value per unit area, desc
            sort(cand.begin(), cand.end(), [&](int a, int b){
                double da = (double)dv[a] / (double)(dw[a] * dh[a]);
                double db = (double)dv[b] / (double)(dw[b] * dh[b]);
                if (da != db) return da > db;
                return dv[a] > dv[b];
            });
            // bounded fill across the width, using a local copy of caps
            int rem = W; ll val = 0; vector<int> pieces; vector<pair<int,int>> use;
            for (int t : cand){
                int n = (int)min<ll>(cap[t], rem / dw[t]);
                if (n <= 0) continue;
                for (int k = 0; k < n; k++) pieces.push_back(t);
                rem -= n * dw[t];
                val += (ll)n * dv[t];
                use.push_back({t, n});
                if (rem <= 0) break;
            }
            if (val > bestVal){ bestVal = val; bestH = hcur; bestPieces = pieces; bestUse = use; }
        }
        if (bestH == -1 || bestVal <= 0) break;
        for (auto &pr : bestUse) cap[pr.first] -= pr.second;
        bands.push_back({bestH, bestPieces});
        Hrem -= bestH;
    }
    emitStack(H, bands, 0);
    return 0;
}
