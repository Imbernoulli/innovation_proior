// TIER: strong
// The insight: symmetry belongs to the FOOTPRINT, not the hardware. A centred,
// axis-aligned rectangle's border is invariant under C2 by construction (no cell
// needs a partner search -- the shape simply maps onto itself); a centred SQUARE's
// border is invariant under the full D4 group (a non-square rectangle is not, since a
// 90-degree turn swaps its width and height). Build that shape with 4 corner mirrors,
// add one redirect mirror where the emitter's own straight column first meets the
// shape's edge, and let the single directed beam close onto it -- the traversal that
// gets it there is not symmetric at all, only the set of cells it lights up is. Search
// a few candidate sizes (all cheap: 4-5 mirrors regardless of size) and keep whichever
// scores best under the same Sym/Motif formula the checker uses.
#include <bits/stdc++.h>
using namespace std;

static int reflect(int dir, char mtype) {
    if (mtype == '/') { static const int m[4] = {1,0,3,2}; return m[dir]; }
    static const int m[4] = {3,2,1,0}; return m[dir];
}

static pair<int,int> applyCode(int code, int r, int c, int n) {
    switch (code) {
        case 0: return {r, c};
        case 1: return {c, n - 1 - r};
        case 2: return {n - 1 - r, n - 1 - c};
        case 3: return {n - 1 - c, r};
        case 4: return {n - 1 - r, c};
        case 5: return {r, n - 1 - c};
        case 6: return {c, r};
        default: return {n - 1 - c, n - 1 - r};
    }
}

static vector<vector<char>> simulate(int n, int ec, const vector<vector<char>>& mirror) {
    vector<vector<char>> illum(n, vector<char>(n, 0));
    vector<vector<array<bool,4>>> visited(n, vector<array<bool,4>>(n, {false,false,false,false}));
    int r = 0, c = ec, dir = 2;
    long long cap = 6LL * n * n + 50;
    for (long long step = 0; step < cap; step++) {
        if (r < 0 || r >= n || c < 0 || c >= n) break;
        illum[r][c] = 1;
        char mt = mirror[r][c];
        int nd = (mt == '/' || mt == '\\') ? reflect(dir, mt) : dir;
        if (visited[r][c][nd]) break;
        visited[r][c][nd] = true;
        r += (nd==0?-1:nd==2?1:0);
        c += (nd==1?1:nd==3?-1:0);
        dir = nd;
    }
    return illum;
}

static long long scoreF(int n, const vector<int>& codes, const vector<vector<char>>& illum) {
    vector<vector<int>> orbitId(n, vector<int>(n, -1));
    vector<int> orbitSize;
    for (int r = 0; r < n; r++) for (int c = 0; c < n; c++) {
        if (orbitId[r][c] != -1) continue;
        vector<pair<int,int>> members;
        for (int code : codes) {
            auto p = applyCode(code, r, c, n);
            bool dup=false; for (auto&m:members) if (m==p){dup=true;break;}
            if (!dup) members.push_back(p);
        }
        int id = (int)orbitSize.size();
        for (auto&p: members) orbitId[p.first][p.second]=id;
        orbitSize.push_back((int)members.size());
    }
    vector<int> touchedCount(orbitSize.size(),0);
    vector<char> touched(orbitSize.size(),0);
    for (int r=0;r<n;r++) for (int c=0;c<n;c++) if (illum[r][c]) {
        int id = orbitId[r][c]; touchedCount[id]++; touched[id]=1;
    }
    double symSum=0.0; int nT=0;
    for (size_t id=0; id<orbitSize.size(); id++) if (touched[id]) {
        symSum += (double)touchedCount[id]/(double)orbitSize[id]; nT++;
    }
    double sym = nT>0 ? symSum/nT : 0.0;
    bool seenMask[512] = {false};
    int motif=0;
    for (int r0=0;r0+3<=n;r0++) for (int c0=0;c0+3<=n;c0++) {
        int mask=0,bit=0; bool any1=false;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
            if (illum[r0+i][c0+j]) { mask|=(1<<bit); any1=true; }
            bit++;
        }
        if (any1 && !seenMask[mask]) { seenMask[mask]=true; motif++; }
    }
    int motifCapped = min(motif,15);
    return (long long)llround(sym*100.0) * (1+motifCapped);
}

int main() {
    int n, M, ec;
    char group[8];
    scanf("%d %d %7s", &n, &M, group);
    scanf("%d", &ec);

    vector<int> codes;
    if (string(group) == "C2") codes = {0,2};
    else codes = {0,1,2,3,4,5,6,7};

    long long bestF = -1;
    vector<pair<pair<int,int>,char>> bestMirrors;

    for (int half = 2; half <= n/2; half++) {
        int c0 = n/2 - half, c1 = n/2 - 1 + half;
        int r0 = c0, r1 = c1;
        if (!(c0 <= ec && ec <= c1)) continue;

        vector<pair<pair<int,int>,char>> mirrors;
        mirrors.push_back({{r0,c0}, '/'});
        mirrors.push_back({{r0,c1}, '\\'});
        mirrors.push_back({{r1,c0}, '\\'});
        mirrors.push_back({{r1,c1}, '/'});
        if (ec != c0 && ec != c1) mirrors.push_back({{r0,ec}, '\\'});
        else {
            // lead-in coincides with a corner: overwrite that corner so the
            // south-arriving beam still turns onto the top edge correctly.
            for (auto& m : mirrors) if (m.first == make_pair(r0, ec)) m.second = '\\';
        }
        if ((int)mirrors.size() > M) continue;

        vector<vector<char>> grid(n, vector<char>(n, 0));
        for (auto& m : mirrors) grid[m.first.first][m.first.second] = m.second;
        auto illum = simulate(n, ec, grid);
        long long F = scoreF(n, codes, illum);
        if (F > bestF) { bestF = F; bestMirrors = mirrors; }
    }

    printf("%d\n", (int)bestMirrors.size());
    for (auto& m : bestMirrors) printf("%d %d %c\n", m.first.first, m.first.second, m.second);
    return 0;
}
