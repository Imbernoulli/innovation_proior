// TIER: trivial
// Best single shelf: pick the type maximizing v*min(c, W/w) and slice one band of
// that many copies. This reproduces the checker baseline B -> ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int W, H, K, D;
vector<int> dw, dh, dv, dc;

// emit one band (rect width w, height bh) filled with the given piece types L->R
void emitBand(int w, int bh, const vector<int>& pieces, int idx){
    if (idx >= (int)pieces.size()){ printf("0 0\n"); return; }   // scrap
    int t = pieces[idx];
    int pw = dw[t];
    if (pw == w && idx == (int)pieces.size() - 1){ printf("0 %d\n", t); return; }
    // vertical cut: child1 = piece(pw,bh) leaf, child2 = rest
    printf("1 0 %d\n", pw);
    printf("0 %d\n", t);
    emitBand(w - pw, bh, pieces, idx + 1);
}

// emit vertical stack of bands over rect (W x Hrem)
void emitStack(int Hrem, const vector<pair<int,vector<int>>>& bands, int idx){
    if (idx >= (int)bands.size()){ printf("0 0\n"); return; }    // scrap band
    int bh = bands[idx].first;
    if (bh == Hrem && idx == (int)bands.size() - 1){ emitBand(W, bh, bands[idx].second, 0); return; }
    printf("1 1 %d\n", bh);                                      // horizontal cut
    emitBand(W, bh, bands[idx].second, 0);
    emitStack(Hrem - bh, bands, idx + 1);
}

int main(){
    scanf("%d %d %d %d", &W, &H, &K, &D);
    dw.assign(D + 1, 0); dh.assign(D + 1, 0); dv.assign(D + 1, 0); dc.assign(D + 1, 0);
    for (int i = 1; i <= D; i++) scanf("%d %d %d %d", &dw[i], &dh[i], &dv[i], &dc[i]);

    int best = -1; ll bestVal = -1;
    for (int i = 1; i <= D; i++){
        if (dw[i] <= W && dh[i] <= H){
            ll fit = min<ll>(dc[i], W / dw[i]);
            if (dv[i] * fit > bestVal){ bestVal = dv[i] * fit; best = i; }
        }
    }
    vector<pair<int,vector<int>>> bands;
    if (best != -1){
        int p = min<ll>(dc[best], W / dw[best]);
        vector<int> pieces(p, best);
        bands.push_back({dh[best], pieces});
    }
    emitStack(H, bands, 0);
    return 0;
}
