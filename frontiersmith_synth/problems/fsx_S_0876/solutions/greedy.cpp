// TIER: greedy
// The obvious first pass: tile the panel area-first (largest shape that fits,
// scanned row-major), and cut each chosen piece from the sheet by simple
// first-fit (rotation 0 tried first, then position row-major) -- COMPLETELY
// IGNORING the grain-angle fields on both sides. This is a strong area-packer
// (low waste, low uncovered count) but a grain-blind one: on tests where the
// target flow field is a swirl, the sheet-cut rotation it happens to land on
// is essentially uncorrelated with what the panel needs there.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int Ws, Hs, Wp, Hp, K;
vector<vector<int>> src, tgt;
vector<string> mask;
vector<vector<pair<int,int>>> cat;
vector<vector<char>> sheetUsed;
vector<vector<char>> panelUsed;

int rotDx(int dx,int dy,int r){ switch(r&3){case 0:return dx;case 1:return -dy;case 2:return -dx;default:return dy;} }
int rotDy(int dx,int dy,int r){ switch(r&3){case 0:return dy;case 1:return dx;case 2:return -dy;default:return -dx;} }

// find a geometric fit on the panel so that shape's local cell k lands at (x0,y0)
bool tryPlacePanel(int shapeIdx, int x0, int y0, int &ax, int &ay, int &rpOut, vector<pair<int,int>> &cellsOut){
    auto &cells = cat[shapeIdx];
    int nc = (int)cells.size();
    for (int rp = 0; rp < 4; rp++){
        for (int k = 0; k < nc; k++){
            int rdx = rotDx(cells[k].first, cells[k].second, rp);
            int rdy = rotDy(cells[k].first, cells[k].second, rp);
            int aax = x0 - rdx, aay = y0 - rdy;
            bool ok = true;
            vector<pair<int,int>> tmp(nc);
            for (int j = 0; j < nc && ok; j++){
                int px = aax + rotDx(cells[j].first, cells[j].second, rp);
                int py = aay + rotDy(cells[j].first, cells[j].second, rp);
                if (px < 0 || px >= Wp || py < 0 || py >= Hp) ok = false;
                else if (mask[py][px] != '#' || panelUsed[py][px]) ok = false;
                else tmp[j] = {px, py};
            }
            if (ok){ ax = aax; ay = aay; rpOut = rp; cellsOut = tmp; return true; }
        }
    }
    return false;
}

// blind first-fit on the sheet: rs outer (priority to rs=0), then row-major position
bool tryPlaceSheetBlind(int shapeIdx, int &sxOut, int &syOut, int &rsOut, vector<pair<int,int>> &cellsOut){
    auto &cells = cat[shapeIdx];
    int nc = (int)cells.size();
    for (int rs = 0; rs < 4; rs++){
        for (int y = 0; y < Hs; y++){
            for (int x = 0; x < Ws; x++){
                bool ok = true;
                vector<pair<int,int>> tmp(nc);
                for (int j = 0; j < nc && ok; j++){
                    int sx = x + rotDx(cells[j].first, cells[j].second, rs);
                    int sy = y + rotDy(cells[j].first, cells[j].second, rs);
                    if (sx < 0 || sx >= Ws || sy < 0 || sy >= Hs) ok = false;
                    else if (sheetUsed[sy][sx]) ok = false;
                    else tmp[j] = {sx, sy};
                }
                if (ok){ sxOut = x; syOut = y; rsOut = rs; cellsOut = tmp; return true; }
            }
        }
    }
    return false;
}

int main(){
    cin >> Ws >> Hs;
    src.assign(Hs, vector<int>(Ws));
    for (int y = 0; y < Hs; y++) for (int x = 0; x < Ws; x++) cin >> src[y][x];
    cin >> Wp >> Hp;
    mask.assign(Hp, "");
    for (int y = 0; y < Hp; y++) cin >> mask[y];
    tgt.assign(Hp, vector<int>(Wp));
    for (int y = 0; y < Hp; y++) for (int x = 0; x < Wp; x++) cin >> tgt[y][x];
    cin >> K;
    cat.assign(K, {});
    for (int i = 0; i < K; i++){
        int nc; cin >> nc; cat[i].resize(nc);
        for (int j = 0; j < nc; j++) cin >> cat[i][j].first >> cat[i][j].second;
    }
    ll lambda, mu, Cu; cin >> lambda >> mu >> Cu;

    sheetUsed.assign(Hs, vector<char>(Ws, 0));
    panelUsed.assign(Hp, vector<char>(Wp, 0));

    vector<int> order(K); for (int i = 0; i < K; i++) order[i] = i;
    sort(order.begin(), order.end(), [&](int a, int b){ return cat[a].size() > cat[b].size(); });

    vector<array<int,7>> out; // shape_id xs ys rs xp yp rp

    for (int y0 = 0; y0 < Hp; y0++){
        for (int x0 = 0; x0 < Wp; x0++){
            if (mask[y0][x0] != '#' || panelUsed[y0][x0]) continue;
            bool placed = false;
            for (int oi = 0; oi < K && !placed; oi++){
                int shapeIdx = order[oi];
                int ax, ay, rp; vector<pair<int,int>> pcells;
                if (!tryPlacePanel(shapeIdx, x0, y0, ax, ay, rp, pcells)) continue;
                int sx, sy, rs; vector<pair<int,int>> scells;
                if (!tryPlaceSheetBlind(shapeIdx, sx, sy, rs, scells)) continue;
                for (auto &c : pcells) panelUsed[c.second][c.first] = 1;
                for (auto &c : scells) sheetUsed[c.second][c.first] = 1;
                out.push_back({shapeIdx, sx, sy, rs, ax, ay, rp});
                placed = true;
            }
            if (!placed){
                // fallback: monomino (shape 0 is always size 1 in this catalog)
                int mono = -1;
                for (int i = 0; i < K; i++) if (cat[i].size() == 1) { mono = i; break; }
                if (mono >= 0){
                    int sx, sy, rs; vector<pair<int,int>> scells;
                    if (tryPlaceSheetBlind(mono, sx, sy, rs, scells)){
                        panelUsed[y0][x0] = 1;
                        sheetUsed[sy][sx] = 1;
                        out.push_back({mono, sx, sy, rs, x0, y0, 0});
                    }
                }
            }
        }
    }

    cout << out.size() << "\n";
    for (auto &o : out) cout << o[0] << " " << o[1] << " " << o[2] << " " << o[3] << " " << o[4] << " " << o[5] << " " << o[6] << "\n";
    return 0;
}
