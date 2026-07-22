#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

struct Grid {
    int R, C, M, Wr, CovMin, CovMax, Sc;
    vector<string> g;
};

static Grid readGridFromInf(InStream &in) {
    Grid gr;
    gr.R = in.readInt(1, 100000, "R");
    gr.C = in.readInt(1, 100000, "C");
    gr.M = in.readInt(1, 100000, "M");
    gr.Wr = in.readInt(1, 1000000, "Wr");
    gr.CovMin = in.readInt(0, 1000, "CovMin");
    gr.CovMax = in.readInt(0, 1000, "CovMax");
    gr.Sc = in.readInt(1, 1000000, "Sc");
    gr.g.resize(gr.R);
    for (int i = 0; i < gr.R; i++) gr.g[i] = in.readToken();
    return gr;
}

static const int CH_DR[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
static const int CH_DC[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

static vector<int> distToRock(const Grid &gr) {
    int R = gr.R, C = gr.C;
    vector<int> dist(R * C, -1);
    queue<int> q;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (gr.g[r][c] == '#') {
                for (int k = 0; k < 8; k++) {
                    int nr = r + CH_DR[k], nc = c + CH_DC[k];
                    if (nr >= 0 && nr < R && nc >= 0 && nc < C && gr.g[nr][nc] == '.' &&
                        dist[nr * C + nc] == -1) {
                        dist[nr * C + nc] = 1;
                        q.push(nr * C + nc);
                    }
                }
            }
    while (!q.empty()) {
        int cur = q.front(); q.pop();
        int r = cur / C, c = cur % C;
        for (int k = 0; k < 8; k++) {
            int nr = r + CH_DR[k], nc = c + CH_DC[k];
            if (nr >= 0 && nr < R && nc >= 0 && nc < C && gr.g[nr][nc] == '.' &&
                dist[nr * C + nc] == -1) {
                dist[nr * C + nc] = dist[cur] + 1;
                q.push(nr * C + nc);
            }
        }
    }
    int CAP = R + C + 5;
    for (auto &x : dist) if (x == -1) x = CAP;
    return dist;
}

// Objective: weighted uncovered sand (rings around obstacles cost Wr per cell,
// open sand costs 1 per cell) plus Sc * (population variance of the Chebyshev-ish
// graph-gap from every sand cell to its nearest covered/raked cell). Minimized.
static double computeF(const Grid &gr, const vector<char> &covered, const vector<int> &D) {
    int R = gr.R, C = gr.C;
    long long weightedUncov = 0;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (gr.g[r][c] == '.') {
                int idx = r * C + c;
                if (!covered[idx]) weightedUncov += (D[idx] <= gr.M ? gr.Wr : 1);
            }

    vector<int> gap(R * C, -1);
    queue<int> q;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) {
            int idx = r * C + c;
            if (gr.g[r][c] == '.' && covered[idx]) { gap[idx] = 0; q.push(idx); }
        }
    while (!q.empty()) {
        int cur = q.front(); q.pop();
        int r = cur / C, c = cur % C;
        for (int k = 0; k < 8; k++) {
            int nr = r + CH_DR[k], nc = c + CH_DC[k];
            if (nr >= 0 && nr < R && nc >= 0 && nc < C && gr.g[nr][nc] == '.') {
                int nidx = nr * C + nc;
                if (gap[nidx] == -1) { gap[nidx] = gap[cur] + 1; q.push(nidx); }
            }
        }
    }
    int CAP = R + C + 5;
    double sum = 0, sumsq = 0;
    long long n = 0;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (gr.g[r][c] == '.') {
                int idx = r * C + c;
                int gv = (gap[idx] == -1) ? CAP : gap[idx];
                sum += gv; sumsq += (double)gv * gv; n++;
            }
    double mean = n ? sum / n : 0.0;
    double var = n ? (sumsq / n - mean * mean) : 0.0;
    if (var < 0) var = 0;
    return (double)weightedUncov + (double)gr.Sc * var;
}

// The checker's own simple reference construction (must match solutions/trivial.cpp
// exactly): fill sand cells in row-major reading order until the coverage floor of
// the band is met, then drop any leftover length-1 row runs.
static vector<char> trivialCovered(const Grid &gr) {
    int R = gr.R, C = gr.C;
    long long totalSand = 0;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (gr.g[r][c] == '.') totalSand++;
    long long target = (long long)llround(totalSand * (gr.CovMin + gr.CovMax) / 2000.0);

    vector<char> mask(R * C, 0);
    long long picked = 0;
    for (int r = 0; r < R && picked < target; r++)
        for (int c = 0; c < C && picked < target; c++)
            if (gr.g[r][c] == '.') { mask[r * C + c] = 1; picked++; }

    for (int r = 0; r < R; r++) {
        int c = 0;
        while (c < C) {
            if (gr.g[r][c] == '.' && mask[r * C + c]) {
                int c0 = c;
                while (c < C && gr.g[r][c] == '.' && mask[r * C + c]) c++;
                if (c - c0 < 2) mask[r * C + c0] = 0;
            } else c++;
        }
    }
    return mask;
}

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);
    Grid gr = readGridFromInf(inf);
    int R = gr.R, C = gr.C;

    long long totalSand = 0;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (gr.g[r][c] == '.') totalSand++;

    int P = ouf.readInt(0, R * C, "P");
    vector<char> covered(R * C, 0);
    long long coveredCount = 0;
    for (int i = 0; i < P; i++) {
        int L = ouf.readInt(2, R * C, "L");
        int pr = -1, pc = -1;
        for (int j = 0; j < L; j++) {
            int r = ouf.readInt(1, R, "r");
            int c = ouf.readInt(1, C, "c");
            if (gr.g[r - 1][c - 1] != '.')
                quitf(_wa, "line %d cell %d = (%d,%d) is not a sand cell", i, j, r, c);
            int idx = (r - 1) * C + (c - 1);
            if (covered[idx])
                quitf(_wa, "cell (%d,%d) used more than once across lines", r, c);
            if (j > 0) {
                int ddr = abs(r - pr), ddc = abs(c - pc);
                if (ddr + ddc != 1)
                    quitf(_wa, "line %d: cells %d and %d are not edge-adjacent", i, j - 1, j);
            }
            covered[idx] = 1;
            coveredCount++;
            pr = r; pc = c;
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    if (totalSand > 0) {
        if (coveredCount * 1000LL < (long long)gr.CovMin * totalSand)
            quitf(_wa, "coverage %lld/%lld sand cells below CovMin=%d permille",
                  coveredCount, totalSand, gr.CovMin);
        if (coveredCount * 1000LL > (long long)gr.CovMax * totalSand)
            quitf(_wa, "coverage %lld/%lld sand cells above CovMax=%d permille",
                  coveredCount, totalSand, gr.CovMax);
    }

    vector<int> D = distToRock(gr);
    double F = computeF(gr, covered, D);

    vector<char> trivMask = trivialCovered(gr);
    double B = computeF(gr, trivMask, D);
    if (B < 1e-9) B = 1e-9;

    double sc = min(1000.0, 100.0 * B / max(1e-9, F));
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f covered=%lld/%lld Ratio: %.6f",
          F, B, coveredCount, totalSand, sc / 1000.0);
    return 0;
}
