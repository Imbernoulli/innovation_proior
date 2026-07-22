// Shared helpers for the "hedge-maze-veiled-symmetry" solutions (trivial/greedy/strong).
// Grid is n x n, 0-indexed, n odd. Entrance = (0,0). Center = (m,m), m=(n-1)/2.
// A "wall set" is stored as open[e] booleans over the 2*n*(n-1) internal edges:
//   horiz edge (r,c)-(r,c+1): id = r*(n-1)+c                       [0 .. n*(n-1)-1]
//   vert  edge (r,c)-(r+1,c): id = n*(n-1) + r*n + c               [n*(n-1) .. 2*n*(n-1)-1]
#pragma once
#include <bits/stdc++.h>
using namespace std;

typedef pair<int,int> Cell;

static inline int edgeIdOf(int n, int r1, int c1, int r2, int c2) {
    if (r1 == r2) { int c = min(c1, c2); return r1 * (n - 1) + c; }
    int r = min(r1, r2); return n * (n - 1) + r * n + c1;
}

// D4 transforms g=0..6 (identity excluded): 3 rotations + 4 reflections.
static inline Cell applyG(int n, int r, int c, int g) {
    switch (g) {
        case 0: return {c, n - 1 - r};             // rot90 cw
        case 1: return {n - 1 - r, n - 1 - c};      // rot180
        case 2: return {n - 1 - c, r};              // rot270 cw
        case 3: return {r, n - 1 - c};              // flip horizontal (mirror L-R)
        case 4: return {n - 1 - r, c};               // flip vertical (mirror T-B)
        case 5: return {c, r};                       // transpose (main diagonal)
        default: return {n - 1 - c, n - 1 - r};       // anti-transpose (anti diagonal)
    }
}

// Linear part of g acting on a direction vector (dr,dc) (same matrix, translation
// dropped -- valid since all g above are isometries of the square).
static inline void applyGDir(int dr, int dc, int g, int& odr, int& odc) {
    switch (g) {
        case 0: odr = dc; odc = -dr; break;
        case 1: odr = -dr; odc = -dc; break;
        case 2: odr = -dc; odc = dr; break;
        case 3: odr = dr; odc = -dc; break;
        case 4: odr = -dr; odc = dc; break;
        case 5: odr = dc; odc = dr; break;
        default: odr = -dc; odc = -dr; break;
    }
}

static inline int vec2code(int dr, int dc) {
    if (dr == 1) return 0; if (dr == -1) return 1;
    if (dc == 1) return 2; return 3; // dc==-1
}
static const int CODE_DR[4] = {1, -1, 0, 0};
static const int CODE_DC[4] = {0, 0, 1, -1};

// Build image[e][g] = id of edge e's image under symmetry g, for g=0..6.
static vector<array<int,7>> buildImageTable(int n) {
    int E = 2 * n * (n - 1);
    vector<array<int,7>> image(E);
    for (int r = 0; r < n; r++)
        for (int c = 0; c + 1 < n; c++) {
            int id = edgeIdOf(n, r, c, r, c + 1);
            for (int g = 0; g < 7; g++) {
                Cell a = applyG(n, r, c, g), b = applyG(n, r, c + 1, g);
                image[id][g] = edgeIdOf(n, a.first, a.second, b.first, b.second);
            }
        }
    for (int r = 0; r + 1 < n; r++)
        for (int c = 0; c < n; c++) {
            int id = edgeIdOf(n, r, c, r + 1, c);
            for (int g = 0; g < 7; g++) {
                Cell a = applyG(n, r, c, g), b = applyG(n, r + 1, c, g);
                image[id][g] = edgeIdOf(n, a.first, a.second, b.first, b.second);
            }
        }
    return image;
}

// Classic spiral-matrix traversal: starts (0,0), ends at the center cell for odd n.
static vector<Cell> spiralPath(int n) {
    vector<Cell> path; path.reserve((size_t)n * n);
    int top = 0, bottom = n - 1, left = 0, right = n - 1;
    while (top <= bottom && left <= right) {
        for (int c = left; c <= right; c++) path.push_back({top, c});
        top++;
        for (int r = top; r <= bottom; r++) path.push_back({r, right});
        right--;
        if (top <= bottom) {
            for (int c = right; c >= left; c--) path.push_back({bottom, c});
            bottom--;
        }
        if (left <= right) {
            for (int r = bottom; r >= top; r--) path.push_back({r, left});
            left++;
        }
    }
    return path;
}

struct Score { double sym, turn, osc, comp; };

// CAP for oscillation normalization -- must match chk.cc exactly.
static inline double oscCap(int n) { return (double)n * (double)n / 5.0; }

static Score computeScore(const vector<Cell>& path, int n,
                           const vector<array<int,7>>& image,
                           long long wSym, long long wTurn, long long wOsc) {
    int L = (int)path.size();
    int E = 2 * n * (n - 1);
    int m = (n - 1) / 2;
    // dirState[e] = -1 (closed/wall) or 0..3 (open, actual travel direction code).
    vector<int8_t> dirState(E, -1);
    for (int i = 0; i + 1 < L; i++) {
        auto [r1, c1] = path[i]; auto [r2, c2] = path[i + 1];
        int e = edgeIdOf(n, r1, c1, r2, c2);
        dirState[e] = (int8_t)vec2code(r2 - r1, c2 - c1);
    }
    long long matchCount = 0;
    for (int e = 0; e < E; e++) {
        for (int g = 0; g < 7; g++) {
            int im = image[e][g];
            if (dirState[e] < 0) { if (dirState[im] < 0) matchCount++; continue; }
            int odr, odc; applyGDir(CODE_DR[dirState[e]], CODE_DC[dirState[e]], g, odr, odc);
            int expect = vec2code(odr, odc);
            if (dirState[im] == expect) matchCount++;
        }
    }
    double sym = (double)matchCount / ((double)E * 7.0);

    long long nS = 0, nL = 0, nR = 0;
    for (int i = 1; i + 1 < L; i++) {
        int dr1 = path[i].first - path[i - 1].first, dc1 = path[i].second - path[i - 1].second;
        int dr2 = path[i + 1].first - path[i].first, dc2 = path[i + 1].second - path[i].second;
        if (dr1 == dr2 && dc1 == dc2) nS++;
        else {
            long long cross = (long long)dr1 * dc2 - (long long)dc1 * dr2;
            if (cross > 0) nL++; else nR++;
        }
    }
    long long totalTurns = nS + nL + nR;
    double turn = 0.0;
    if (totalTurns > 0) {
        double H = 0.0;
        for (long long cnt : {nS, nL, nR}) if (cnt > 0) {
            double p = (double)cnt / (double)totalTurns;
            H -= p * log2(p);
        }
        turn = H / log2(3.0);
    }

    long long TV = 0;
    vector<int> depth(L);
    for (int i = 0; i < L; i++) depth[i] = max(abs(path[i].first - m), abs(path[i].second - m));
    for (int i = 0; i + 1 < L; i++) TV += llabs((long long)depth[i + 1] - depth[i]);
    long long minTV = depth[0];
    double oscExcess = (double)(TV - minTV);
    double osc = min(1.0, oscExcess / oscCap(n));

    Score s;
    s.sym = sym; s.turn = turn; s.osc = osc;
    double wsum = (double)(wSym + wTurn + wOsc);
    s.comp = ((double)wSym * sym + (double)wTurn * turn + (double)wOsc * osc) / wsum;
    return s;
}

static inline bool adjacent(const Cell& a, const Cell& b) {
    return abs(a.first - b.first) + abs(a.second - b.second) == 1;
}

// Bounded 2-opt local search over grid-adjacency-preserving path reversals.
// useComposite=true  -> strong: accepts a move iff it raises the FULL weighted score,
//   so it explicitly weighs every candidate splice's effect on symScore, turnScore AND
//   oscScore together (per the actual input weights) before taking it.
// useComposite=false -> greedy: accepts a move iff it raises turnScore ALONE -- it is
//   never steering toward a symmetry or oscillation gain, and its only safety rail is
//   refusing a move that would drop the (unweighted-by-intent) composite below where it
//   started. Because grid-adjacency 2-opt splices on a spiral are structurally biased
//   toward ring-crossing moves (a single ring has almost no internal slack -- it is a
//   1-cell-wide path), greedy's turn-only moves routinely displace symScore/oscScore as
//   an unplanned side effect, sometimes helping them, often not: it has no mechanism to
//   tell whether a given splice is good or bad for the OTHER two mechanisms, so on any
//   test where they carry real weight it fares far worse than the joint search below.
static void localSearch(vector<Cell>& path, int n, const vector<array<int,7>>& image,
                         long long wSym, long long wTurn, long long wOsc,
                         int window, int sweeps, bool useComposite) {
    int L = (int)path.size();
    Score start = computeScore(path, n, image, wSym, wTurn, wOsc);
    Score cur = start;
    auto metric = [&](const Score& s) { return useComposite ? s.comp : s.turn; };
    for (int sw = 0; sw < sweeps; sw++) {
        bool improved = false;
        for (int i = 0; i <= L - 3; i++) {
            int jmax = min(L - 2, i + window);
            for (int j = i + 2; j <= jmax; j++) {
                if (!adjacent(path[i], path[j])) continue;
                if (!adjacent(path[i + 1], path[j + 1])) continue;
                vector<Cell> cand = path;
                reverse(cand.begin() + i + 1, cand.begin() + j + 1);
                Score cs = computeScore(cand, n, image, wSym, wTurn, wOsc);
                if (metric(cs) <= metric(cur) + 1e-12) continue;
                if (!useComposite && cs.comp < start.comp - 1e-12) continue;
                path.swap(cand); cur = cs; improved = true;
            }
        }
        if (!improved) break;
    }
}

static void readInput(istream& in, int& n, long long& wSym, long long& wTurn, long long& wOsc) {
    in >> n >> wSym >> wTurn >> wOsc;
}

static void printMaze(ostream& out, const vector<Cell>& path, int n) {
    int E = 2 * n * (n - 1);
    vector<char> open(E, 0);
    int L = (int)path.size();
    for (int i = 0; i + 1 < L; i++) {
        auto [r1, c1] = path[i]; auto [r2, c2] = path[i + 1];
        open[edgeIdOf(n, r1, c1, r2, c2)] = 1;
    }
    for (int r = 0; r < n; r++) {
        for (int c = 0; c + 1 < n; c++) {
            out << (open[edgeIdOf(n, r, c, r, c + 1)] ? 0 : 1);
            if (c + 2 < n) out << ' ';
        }
        out << '\n';
    }
    for (int r = 0; r + 1 < n; r++) {
        for (int c = 0; c < n; c++) {
            out << (open[edgeIdOf(n, r, c, r + 1, c)] ? 0 : 1);
            if (c + 1 < n) out << ' ';
        }
        out << '\n';
    }
}
