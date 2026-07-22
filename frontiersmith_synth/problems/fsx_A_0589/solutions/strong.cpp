// TIER: strong
// Insight: the score is Groups * Seam + Motif. Groups needs orbit periods > 2
// (only 4 signatures exist at period <= 2), but mixed periods create
// divisibility-incompatible seams that zero out harmony -- UNLESS the bands are
// ORDERED so neighboring periods divide one another (a divisibility chain), and
// colors are co-designed so the scarce high-harmony pairs land on those seams.
// This solver runs a deterministic simulated-annealing co-search over
// (period, tile, stacking order, colors) with an explicit divisibility-reorder
// move -- reformulating the task as "pick a diverse multiset of frieze groups,
// then thread them onto a divisibility chain."
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int B, h, P, Q;
vector<vector<int>> M;
vector<vector<int>> divs; // proper divisors of each p (1..Q), including 1..p-1 that divide p

struct Band { int p; vector<vector<int>> T; }; // T: h x p

mt19937 rng(987654321u);
int ri(int lo, int hi) { return lo + (int)(rng() % (unsigned)(hi - lo + 1)); }
double rd() { return (rng() >> 8) / (double)(1u << 24); }

bool primitive(const vector<vector<int>>& T, int p) {
    for (int di = 0; di < (int)divs[p].size(); di++) {
        int d = divs[p][di];
        if (d >= p) continue;
        bool per = true;
        for (int r = 0; r < h && per; r++)
            for (int x = 0; x < p; x++)
                if (T[r][x] != T[r][(x + d) % p]) { per = false; break; }
        if (per) return false;
    }
    return true;
}

int sigOf(const vector<vector<int>>& T, int L) {
    bool mH = true;
    for (int r = 0; r < h && mH; r++)
        for (int x = 0; x < L; x++)
            if (T[r][x] != T[h - 1 - r][x]) { mH = false; break; }
    bool mG = false;
    if (L % 2 == 0) {
        mG = true; int s = L / 2;
        for (int r = 0; r < h && mG; r++)
            for (int x = 0; x < L; x++)
                if (T[r][x] != T[h - 1 - r][(x + s) % L]) { mG = false; break; }
    }
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

double evalF(const vector<Band>& bs) {
    set<int> sigs; set<string> tiles;
    for (auto& b : bs) {
        sigs.insert(sigOf(b.T, b.p));
        string s = to_string(b.p) + ":";
        for (int r = 0; r < h; r++) { for (int x = 0; x < b.p; x++) { s += to_string(b.T[r][x]); s += ','; } s += '|'; }
        tiles.insert(s);
    }
    double Seam = 0.0;
    for (int i = 0; i + 1 < (int)bs.size(); i++) {
        int p1 = bs[i].p, p2 = bs[i + 1].p;
        if (p1 % p2 == 0 || p2 % p1 == 0) {
            int L = max(p1, p2); ll s = 0;
            for (int x = 0; x < L; x++)
                s += M[bs[i].T[h - 1][x % p1]][bs[i + 1].T[0][x % p2]];
            Seam += (double)s / (double)L;
        }
    }
    return (double)sigs.size() * Seam + 1.0 * (double)tiles.size();
}

void randTile(Band& b, int p) {
    b.p = p; b.T.assign(h, vector<int>(p));
    for (int t = 0; t < 40; t++) {
        for (int r = 0; r < h; r++) for (int x = 0; x < p; x++) b.T[r][x] = ri(0, P - 1);
        if (primitive(b.T, p)) return;
    }
    // fallback: guaranteed primitive
    for (int r = 0; r < h; r++) for (int x = 0; x < p; x++) b.T[r][x] = 0;
    b.T[1][0] = 1 % P; if (P > 1) b.T[1][0] = 1; else b.T[1][0] = 0;
}

int main() {
    if (scanf("%d %d %d %d", &B, &h, &P, &Q) != 4) return 0;
    M.assign(P, vector<int>(P));
    for (int i = 0; i < P; i++) for (int j = 0; j < P; j++) scanf("%d", &M[i][j]);

    divs.assign(Q + 1, {});
    for (int p = 1; p <= Q; p++) for (int d = 1; d < p; d++) if (p % d == 0) divs[p].push_back(d);

    // period palette: small values plus a spread up to Q
    vector<int> plist;
    for (int p = 1; p <= min(Q, 4); p++) plist.push_back(p);
    for (int p = 5; p <= Q; p++) plist.push_back(p);

    // init: diverse random bands
    vector<Band> cur(B);
    for (int b = 0; b < B; b++) randTile(cur[b], plist[ri(0, (int)plist.size() - 1)]);
    double curF = evalF(cur);
    vector<Band> best = cur; double bestF = curF;

    // iteration budget tuned to stay well under the time limit
    long ITER = 90000;
    double T0 = 6.0, T1 = 0.03;

    for (long it = 0; it < ITER; it++) {
        double frac = (double)it / ITER;
        double temp = T0 * pow(T1 / T0, frac);
        int mv = ri(0, 99);
        // snapshot minimal info to revert
        if (mv < 35) {
            // reassign a band's period+tile
            int b = ri(0, B - 1);
            Band old = cur[b];
            randTile(cur[b], plist[ri(0, (int)plist.size() - 1)]);
            double nf = evalF(cur);
            if (nf >= curF || rd() < exp((nf - curF) / temp)) curF = nf;
            else cur[b] = old;
        } else if (mv < 70) {
            // mutate one cell (keep primitive)
            int b = ri(0, B - 1);
            int r = ri(0, h - 1), x = ri(0, cur[b].p - 1);
            int oldc = cur[b].T[r][x];
            int nc = ri(0, P - 1);
            cur[b].T[r][x] = nc;
            if (!primitive(cur[b].T, cur[b].p)) { cur[b].T[r][x] = oldc; }
            else {
                double nf = evalF(cur);
                if (nf >= curF || rd() < exp((nf - curF) / temp)) curF = nf;
                else cur[b].T[r][x] = oldc;
            }
        } else if (mv < 90) {
            // swap two bands (reorder)
            int i = ri(0, B - 1), j = ri(0, B - 1);
            if (i == j) continue;
            swap(cur[i], cur[j]);
            double nf = evalF(cur);
            if (nf >= curF || rd() < exp((nf - curF) / temp)) curF = nf;
            else swap(cur[i], cur[j]);
        } else {
            // divisibility reorder: sort by period ascending (chains 1|2|4.., groups equals)
            vector<Band> snap = cur;
            stable_sort(cur.begin(), cur.end(), [](const Band& a, const Band& b) { return a.p < b.p; });
            double nf = evalF(cur);
            if (nf >= curF) curF = nf;
            else cur = snap;
        }
        if (curF > bestF) { bestF = curF; best = cur; }
    }

    for (auto& b : best) {
        printf("%d\n", b.p);
        for (int r = 0; r < h; r++) {
            for (int x = 0; x < b.p; x++) printf("%d%c", b.T[r][x], x + 1 < b.p ? ' ' : '\n');
        }
    }
    return 0;
}
