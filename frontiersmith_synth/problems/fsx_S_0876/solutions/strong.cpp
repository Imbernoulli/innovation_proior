// TIER: strong
// The insight: a piece is ONE object with two linked poses, so panel-tiling and
// sheet-cutting must not be solved as two independent packings. For every panel
// placement, this solution searches the sheet EXHAUSTIVELY over (position, sheet
// rotation) -- i.e. over the whole pose-PAIR, since only the rotation DIFFERENCE
// (panel rot - sheet rot) determines the glued-in grain angle -- and keeps the
// pair whose resulting angles best match the panel's desired flow field. It also
// gates which shape size is even attempted by how much the target field varies
// across the candidate footprint (spread): flat regions get big pieces (fewer
// seams to mismatch), curved regions fall back to small/diverse shapes that a
// single rotation CAN match well. Panel-side rotation is chosen only for
// geometric fit; the grain optimisation lives entirely in the sheet search.
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
int angDiff(int a,int b){ int d=abs(a-b)%180; return min(d,180-d); }

bool tryPlacePanel(int shapeIdx, int x0, int y0, int spreadCap,
                    int &ax, int &ay, int &rpOut, vector<pair<int,int>> &cellsOut){
    auto &cells = cat[shapeIdx];
    int nc = (int)cells.size();
    for (int rp = 0; rp < 4; rp++){
        for (int k = 0; k < nc; k++){
            int rdx = rotDx(cells[k].first, cells[k].second, rp);
            int rdy = rotDy(cells[k].first, cells[k].second, rp);
            int aax = x0 - rdx, aay = y0 - rdy;
            bool ok = true;
            vector<pair<int,int>> tmp(nc);
            int mn = 1000, mx = -1000;
            for (int j = 0; j < nc && ok; j++){
                int px = aax + rotDx(cells[j].first, cells[j].second, rp);
                int py = aay + rotDy(cells[j].first, cells[j].second, rp);
                if (px < 0 || px >= Wp || py < 0 || py >= Hp) ok = false;
                else if (mask[py][px] != '#' || panelUsed[py][px]) ok = false;
                else { tmp[j] = {px, py}; mn = min(mn, tgt[py][px]); mx = max(mx, tgt[py][px]); }
            }
            if (ok && nc > 1 && (mx - mn) > spreadCap) ok = false;
            if (ok){ ax = aax; ay = aay; rpOut = rp; cellsOut = tmp; return true; }
        }
    }
    return false;
}

// exhaustive search over ALL (position, rs) sheet pose-pairs for this panel
// placement; keeps the one minimizing summed angular mismatch to the target.
bool tryPlaceSheetBest(int shapeIdx, int rp, const vector<pair<int,int>> &panelCells,
                        int &sxOut, int &syOut, int &rsOut, vector<pair<int,int>> &cellsOut){
    auto &cells = cat[shapeIdx];
    int nc = (int)cells.size();
    bool found = false;
    ll bestCost = LLONG_MAX;
    int bestX=0,bestY=0,bestRs=0; vector<pair<int,int>> bestCells;
    for (int rs = 0; rs < 4; rs++){
        int netSteps = ((rp - rs) % 4 + 4) % 4;
        int netDeg = 90 * netSteps;
        for (int y = 0; y < Hs; y++){
            for (int x = 0; x < Ws; x++){
                bool ok = true;
                vector<pair<int,int>> tmp(nc);
                ll cost = 0;
                for (int j = 0; j < nc && ok; j++){
                    int sx = x + rotDx(cells[j].first, cells[j].second, rs);
                    int sy = y + rotDy(cells[j].first, cells[j].second, rs);
                    if (sx < 0 || sx >= Ws || sy < 0 || sy >= Hs) ok = false;
                    else if (sheetUsed[sy][sx]) ok = false;
                    else {
                        tmp[j] = {sx, sy};
                        int ang = (src[sy][sx] + netDeg) % 180;
                        cost += angDiff(ang, tgt[panelCells[j].second][panelCells[j].first]);
                    }
                }
                if (ok && cost < bestCost){
                    bestCost = cost; bestX = x; bestY = y; bestRs = rs; bestCells = tmp; found = true;
                }
            }
        }
    }
    if (found){ sxOut = bestX; syOut = bestY; rsOut = bestRs; cellsOut = bestCells; }
    return found;
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
    int mono = -1;
    for (int i = 0; i < K; i++) if (cat[i].size() == 1) { mono = i; break; }

    vector<array<int,7>> out;
    const int SPREAD_CAP = 45;

    for (int y0 = 0; y0 < Hp; y0++){
        for (int x0 = 0; x0 < Wp; x0++){
            if (mask[y0][x0] != '#' || panelUsed[y0][x0]) continue;
            bool placed = false;
            for (int oi = 0; oi < K && !placed; oi++){
                int shapeIdx = order[oi];
                int ax, ay, rp; vector<pair<int,int>> pcells;
                if (!tryPlacePanel(shapeIdx, x0, y0, SPREAD_CAP, ax, ay, rp, pcells)) continue;
                int sx, sy, rs; vector<pair<int,int>> scells;
                if (!tryPlaceSheetBest(shapeIdx, rp, pcells, sx, sy, rs, scells)) continue;
                for (auto &c : pcells) panelUsed[c.second][c.first] = 1;
                for (auto &c : scells) sheetUsed[c.second][c.first] = 1;
                out.push_back({shapeIdx, sx, sy, rs, ax, ay, rp});
                placed = true;
            }
            if (!placed && mono >= 0){
                int ax, ay, rp; vector<pair<int,int>> pcells;
                if (tryPlacePanel(mono, x0, y0, 1000000, ax, ay, rp, pcells)){
                    int sx, sy, rs; vector<pair<int,int>> scells;
                    if (tryPlaceSheetBest(mono, rp, pcells, sx, sy, rs, scells)){
                        panelUsed[y0][x0] = 1;
                        sheetUsed[sy][sx] = 1;
                        out.push_back({mono, sx, sy, rs, ax, ay, rp});
                    }
                }
            }
        }
    }

    cout << out.size() << "\n";
    for (auto &o : out) cout << o[0] << " " << o[1] << " " << o[2] << " " << o[3] << " " << o[4] << " " << o[5] << " " << o[6] << "\n";
    return 0;
}
