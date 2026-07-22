#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// ---------------------------------------------------------------------------
// Checker/scorer for "Polyphonic Facade Bands"  (family: frieze-band-polyphony)
//
// Reads B bands (period p + h x p primitive tile).  Scores
//   F = Groups * Seam + Motif
// where Groups = #distinct frieze signatures, Seam = sum of divisibility-gated
// seam harmonies, Motif = #distinct tiles.  Baseline B0 = value of the monochrome
// period-1 facade.  ratio = min(1, F/(10*B0)).  Monochrome facade -> 0.1.
// ---------------------------------------------------------------------------

int B, h, P, Q;
vector<vector<int>> M; // P x P harmony

// signature bits: V=1, H=2, G=4, R=8
int sigOf(const vector<vector<int>>& T, int L) {
    // H: horizontal mirror (row r == row h-1-r, no shift)
    bool mH = true;
    for (int r = 0; r < h && mH; r++)
        for (int x = 0; x < L; x++)
            if (T[r][x] != T[h - 1 - r][x]) { mH = false; break; }
    // G: glide (row r == row h-1-r shifted by L/2), needs L even
    bool mG = false;
    if (L % 2 == 0) {
        mG = true;
        int s = L / 2;
        for (int r = 0; r < h && mG; r++)
            for (int x = 0; x < L; x++)
                if (T[r][x] != T[h - 1 - r][(x + s) % L]) { mG = false; break; }
    }
    // V: nontrivial vertical mirror axis a (x-map non-identity)
    bool mV = false;
    for (int a = 0; a < L && !mV; a++) {
        bool nonid = false;
        for (int x = 0; x < L; x++) { int xr = ((a - x) % L + L) % L; if (xr != x) { nonid = true; break; } }
        if (!nonid) continue;
        bool okk = true;
        for (int r = 0; r < h && okk; r++)
            for (int x = 0; x < L; x++) { int xr = ((a - x) % L + L) % L; if (T[r][x] != T[r][xr]) { okk = false; break; } }
        if (okk) mV = true;
    }
    // R: 180 rotation center a (x-map non-identity, rows swapped)
    bool mR = false;
    for (int a = 0; a < L && !mR; a++) {
        bool nonid = false;
        for (int x = 0; x < L; x++) { int xr = ((a - x) % L + L) % L; if (xr != x) { nonid = true; break; } }
        if (!nonid) continue;
        bool okk = true;
        for (int r = 0; r < h && okk; r++)
            for (int x = 0; x < L; x++) { int xr = ((a - x) % L + L) % L; if (T[r][x] != T[h - 1 - r][xr]) { okk = false; break; } }
        if (okk) mR = true;
    }
    return (mV ? 1 : 0) | (mH ? 2 : 0) | (mG ? 4 : 0) | (mR ? 8 : 0);
}

bool primitive(const vector<vector<int>>& T, int L) {
    for (int d = 1; d < L; d++) {
        if (L % d) continue;
        bool per = true;
        for (int r = 0; r < h && per; r++)
            for (int x = 0; x < L; x++)
                if (T[r][x] != T[r][(x + d) % L]) { per = false; break; }
        if (per) return false; // repeats with smaller period d
    }
    return true;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);
    B = inf.readInt();
    h = inf.readInt();
    P = inf.readInt();
    Q = inf.readInt();
    M.assign(P, vector<int>(P));
    for (int i = 0; i < P; i++)
        for (int j = 0; j < P; j++)
            M[i][j] = inf.readInt();

    vector<int> per(B);
    vector<vector<vector<int>>> T(B);
    for (int b = 0; b < B; b++) {
        int p = ouf.readInt(1, Q, "period");
        per[b] = p;
        T[b].assign(h, vector<int>(p));
        for (int r = 0; r < h; r++)
            for (int x = 0; x < p; x++)
                T[b][r][x] = ouf.readInt(0, P - 1, "color");
        if (!primitive(T[b], p))
            quitf(_wa, "band %d tile is not primitive (minimal period < declared %d)", b, p);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // Groups
    set<int> sigs;
    for (int b = 0; b < B; b++) sigs.insert(sigOf(T[b], per[b]));
    int Groups = (int)sigs.size();

    // Motif (distinct tiles)
    set<string> tiles;
    for (int b = 0; b < B; b++) {
        string s = to_string(per[b]) + ":";
        for (int r = 0; r < h; r++) { for (int x = 0; x < per[b]; x++) { s += to_string(T[b][r][x]); s += ','; } s += '|'; }
        tiles.insert(s);
    }
    int Motif = (int)tiles.size();

    // Seam harmony (divisibility gated)
    double Seam = 0.0;
    for (int b = 0; b + 1 < B; b++) {
        int p1 = per[b], p2 = per[b + 1];
        if (p1 % p2 == 0 || p2 % p1 == 0) {
            int L = max(p1, p2);
            ll s = 0;
            for (int x = 0; x < L; x++)
                s += M[T[b][h - 1][x % p1]][T[b + 1][0][x % p2]];
            Seam += (double)s / (double)L;
        }
    }

    double F = (double)Groups * Seam + 1.0 * (double)Motif;
    double B0 = (double)(B - 1) * (double)M[0][0] + 1.0;
    if (B0 < 1e-9) B0 = 1.0;

    double sc = min(1000.0, 100.0 * F / max(1e-9, B0));
    quitp(sc / 1000.0, "OK Groups=%d Seam=%.4f Motif=%d F=%.4f B0=%.4f Ratio: %.6f",
          Groups, Seam, Motif, F, B0, sc / 1000.0);
    return 0;
}
