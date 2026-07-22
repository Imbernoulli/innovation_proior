// TIER: greedy
// The obvious first heuristic: shelf-pack top-down, each band filled with copies of
// the single type of highest PER-PIECE value that still fits and has cap. It chases
// raw value (grabbing shiny needles), uses one type per band, and never reasons
// about value-density or which band heights best fit the demand distribution.
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
    int Hrem = H;
    vector<pair<int,vector<int>>> bands;
    while (Hrem > 0){
        int best = -1;
        for (int i = 1; i <= D; i++){
            if (dh[i] <= Hrem && dw[i] <= W && cap[i] > 0 && W / dw[i] >= 1){
                if (best == -1 || dv[i] > dv[best] || (dv[i] == dv[best] && dw[i] > dw[best]))
                    best = i;
            }
        }
        if (best == -1) break;
        int p = (int)min<ll>(cap[best], W / dw[best]);
        if (p < 1){ cap[best] = 0; continue; }
        vector<int> pieces(p, best);
        bands.push_back({dh[best], pieces});
        cap[best] -= p;
        Hrem -= dh[best];
    }
    emitStack(H, bands, 0);
    return 0;
}
