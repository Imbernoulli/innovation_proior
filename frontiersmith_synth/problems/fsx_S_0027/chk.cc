#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int,int> pii;
typedef vector<pii> Shape;

static Shape normalize(Shape s) {
    int mnx = INT_MAX, mny = INT_MAX;
    for (auto &p : s) { mnx = min(mnx, p.first); mny = min(mny, p.second); }
    for (auto &p : s) { p.first -= mnx; p.second -= mny; }
    sort(s.begin(), s.end());
    return s;
}
static string canon(const Shape &s0) {
    Shape s = normalize(s0);
    string r;
    for (auto &p : s) { r += to_string(p.first); r += ','; r += to_string(p.second); r += ';'; }
    return r;
}
// all distinct orientations (rot x reflect), normalized+sorted, in canonical order
static vector<Shape> orientations(const Shape &base) {
    set<string> seen;
    vector<pair<string,Shape>> tmp;
    for (int refl = 0; refl < 2; refl++) {
        Shape s = base;
        if (refl) for (auto &p : s) p.first = -p.first;
        for (int rot = 0; rot < 4; rot++) {
            Shape t = s;
            for (int r = 0; r < rot; r++)
                for (auto &p : t) { int x = p.first, y = p.second; p.first = -y; p.second = x; }
            Shape nt = normalize(t);
            string c = canon(nt);
            if (!seen.count(c)) { seen.insert(c); tmp.push_back({c, nt}); }
        }
    }
    sort(tmp.begin(), tmp.end(), [](const pair<string,Shape>&a, const pair<string,Shape>&b){ return a.first < b.first; });
    vector<Shape> out;
    for (auto &pr : tmp) out.push_back(pr.second);
    return out;
}
// anchor = row-major-smallest cell (min y, tie min x) of a normalized shape
static pii anchorOf(const Shape &s) {
    int by = INT_MAX, bx = INT_MAX;
    for (auto &p : s)
        if (p.second < by || (p.second == by && p.first < bx)) { by = p.second; bx = p.first; }
    return {bx, by};
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int W = inf.readInt();
    int H = inf.readInt();
    int P = inf.readInt();

    vector<int> stock(P), sz(P);
    vector<vector<Shape>> oris(P);
    vector<set<string>> okset(P);
    for (int t = 0; t < P; t++) {
        stock[t] = inf.readInt();
        sz[t]    = inf.readInt();
        Shape base;
        for (int i = 0; i < sz[t]; i++) { int dx = inf.readInt(); int dy = inf.readInt(); base.push_back({dx,dy}); }
        oris[t] = orientations(base);
        for (auto &o : oris[t]) okset[t].insert(canon(o));
    }
    vector<vector<int>> V(H, vector<int>(W));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) V[y][x] = inf.readInt();

    // ---- checker baseline B: type-0 anchored greedy ----
    ll B = 0;
    {
        vector<vector<char>> cov(H, vector<char>(W, 0));
        int used = 0;
        bool stop = false;
        for (int y = 0; y < H && !stop; y++)
            for (int x = 0; x < W && !stop; x++) {
                if (V[y][x] == 0 || cov[y][x]) continue;
                if (used >= stock[0]) { stop = true; break; }
                for (auto &o : oris[0]) {
                    pii a = anchorOf(o);
                    int ox = x - a.first, oy = y - a.second;
                    bool fit = true;
                    for (auto &p : o) {
                        int cx = ox + p.first, cy = oy + p.second;
                        if (cx < 0 || cx >= W || cy < 0 || cy >= H || V[cy][cx] == 0 || cov[cy][cx]) { fit = false; break; }
                    }
                    if (fit) {
                        for (auto &p : o) { int cx = ox + p.first, cy = oy + p.second; cov[cy][cx] = 1; B += V[cy][cx]; }
                        used++;
                        break;
                    }
                }
            }
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // ---- read & validate participant output ----
    int M = ouf.readInt(0, W * H, "M");
    vector<vector<char>> used(H, vector<char>(W, 0));
    vector<int> cnt(P, 0);
    ll F = 0;
    for (int m = 0; m < M; m++) {
        int t = ouf.readInt(0, P - 1, "type");
        cnt[t]++;
        if (cnt[t] > stock[t]) quitf(_wa, "type %d installed more than its stock %d", t, stock[t]);
        Shape cells;
        set<pii> seen;
        for (int i = 0; i < sz[t]; i++) {
            int x = ouf.readInt(0, W - 1, "x");
            int y = ouf.readInt(0, H - 1, "y");
            if (V[y][x] == 0) quitf(_wa, "copy %d covers crevasse cell (%d,%d)", m, x, y);
            if (used[y][x])   quitf(_wa, "copy %d overlaps already-covered cell (%d,%d)", m, x, y);
            if (seen.count({x,y})) quitf(_wa, "copy %d lists duplicate cell (%d,%d)", m, x, y);
            seen.insert({x,y});
            cells.push_back({x,y});
        }
        if (okset[t].find(canon(cells)) == okset[t].end())
            quitf(_wa, "copy %d is not a legal orientation of type %d", m, t);
        for (auto &p : cells) { used[p.second][p.first] = 1; F += V[p.second][p.first]; }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
