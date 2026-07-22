#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// ---------------------------------------------------------------------------
// Tailor Matching Plaid Across Every Seam -- generator.
//
// Builds N pieces (w_i,h_i) with average width ~= W/10 (so a width-blind
// single-column packer wastes ~90% of the fabric -- calibrates the trivial
// baseline). Pieces are partitioned into disjoint SEAM CHAINS: chain
// [c_0..c_{k-1}] connects c_t's TOP edge to c_{t+1}'s BOTTOM edge (k=1 is an
// isolated piece with no seam). Every chain is PLANTED so a valid assignment
// of rotations f_t in {0,1} and positions exists: walking the chain from
// c_0 (forced f=0) we pick a FREE random target residue for c_0, then for
// each subsequent piece pick a random f_t in {0,1} and derive req_t so that
// this f_t exactly reproduces the residue the seam requires. This plants a
// genuine lattice-coset structure: an obvious "assume f=0 everywhere" reader
// only reproduces the plant by chance (~2^-(k-1) per chain), so long chains
// are TRAP cases where blind rotation choice provably cannot be patched by
// translation alone (translation preserves y mod r, but the wrong f pins y
// mod r to the wrong residue class entirely) -- only recomputing the whole
// chain's required residues (quotienting by the phase lattice) recovers it.
// Short/no chains (early, easy test ids) let a lucky guess succeed often, so
// the trap is concentrated in later, chain-heavy test ids (>=3 of the 10).
// ---------------------------------------------------------------------------

struct Piece { long long w, h, req; };

static long long tgt(long long req, long long h, long long r, int f) {
    if (f == 0) return ((-req) % r + r) % r;
    return ((req - h) % r + r) % r;
}

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int id = atoi(argv[1]);
    if (id < 1 || id > 10) id = 1;

    // per-test: N, W, r, avgWpct(of W /100 scaled x1000 -> avgW=W*avgWpct/1000),
    // hMinMult, hMaxMult (of r), chain length spec, trapHeavy(0/1)
    struct P { int N; int W; long long r; int chainMode; };
    // chainMode: 0 = mostly isolated/short (easy), 1 = mixed short+medium,
    //            2 = long chains dominate (TRAP), 3 = one giant chain + noise (TRAP)
    static const P tbl[11] = {
        {0,0,0,0},
        {8,   50,  4,  0},   // 1: tiny sanity, isolated pieces only
        {20,  120, 6,  0},   // 2: small, mostly isolated + a couple pairs
        {40,  200, 10, 1},   // 3: medium, short chains (len 2-3)
        {60,  260, 15, 1},   // 4: medium, short/medium chains
        {90,  340, 12, 2},   // 5: TRAP -- long chains dominate
        {110, 400, 8,  1},   // 6: larger, mixed (mostly easy)
        {160, 540, 25, 2},   // 7: TRAP -- long chains
        {220, 680, 30, 3},   // 8: TRAP -- one giant chain + noise
        {280, 820, 18, 2},   // 9: TRAP -- long chains, larger r
        {360, 980, 40, 3},   // 10: largest, TRAP -- giant chain + noise, fills envelope
    };
    P p = tbl[id];
    int N = p.N; int W = p.W; long long r = p.r;

    long long avgW = max(2LL, (long long)W / 6);
    long long wlo = max(1LL, avgW * 6 / 10), whi = min((long long)W, avgW * 14 / 10);
    if (whi < wlo) whi = wlo;

    vector<Piece> pc(N);
    for (int i = 0; i < N; i++) {
        pc[i].w = rnd.next(wlo, whi);
        long long hlo = max(1LL, r), hhi = max(hlo, 4 * r);
        pc[i].h = rnd.next(hlo, hhi);
        pc[i].req = 0; // filled in during chain planting
    }

    // build a random partition of [0,N) into chains according to chainMode
    vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i;
    for (int i = N - 1; i > 0; i--) { int j = rnd.next(0, i); swap(order[i], order[j]); }

    vector<vector<int>> chains;
    int pos = 0;
    while (pos < N) {
        int remain = N - pos;
        int len;
        if (p.chainMode == 0) {
            len = (rnd.next(0, 99) < 15 && remain >= 2) ? 2 : 1;
        } else if (p.chainMode == 1) {
            int r100 = rnd.next(0, 99);
            if (r100 < 40) len = 1;
            else if (r100 < 75) len = min(remain, rnd.next(2, 3));
            else len = min(remain, rnd.next(3, 5));
        } else if (p.chainMode == 2) {
            int r100 = rnd.next(0, 99);
            if (r100 < 15) len = 1;
            else len = min(remain, rnd.next(6, 16));
        } else { // 3: one giant chain first, rest shorter/noise
            if (chains.empty()) {
                len = max(2, (int)(remain * 0.65));
                len = min(remain, len);
            } else {
                int r100 = rnd.next(0, 99);
                if (r100 < 30) len = 1;
                else len = min(remain, rnd.next(4, 10));
            }
        }
        len = max(1, min(len, remain));
        vector<int> c;
        for (int t = 0; t < len; t++) c.push_back(order[pos + t]);
        chains.push_back(c);
        pos += len;
    }

    // plant req_i for each chain so a valid rotation/position assignment exists
    vector<pair<pair<int,int>, pair<int,int>>> seams; // ((piece,edge),(piece,edge))
    for (auto& c : chains) {
        int k = (int)c.size();
        // c[0]: forced f=0, free target residue
        long long R0 = rnd.next(0LL, r - 1);
        pc[c[0]].req = ((-R0) % r + r) % r; // tgt(req,h,r,0) == R0
        long long need = (R0 + pc[c[0]].h) % r; // outgoing TOP phase
        for (int t = 1; t < k; t++) {
            int ft = rnd.next(0, 1);
            long long h_t = pc[c[t]].h;
            long long reqv;
            if (ft == 0) reqv = ((-need) % r + r) % r;
            else reqv = ((need + h_t) % r + r) % r;
            pc[c[t]].req = reqv;
            seams.push_back({{c[t-1], 1}, {c[t], 0}});
            if (t < k - 1) need = (need + h_t) % r;
        }
    }

    printf("%d %d %lld\n", N, W, r);
    string buf;
    for (int i = 0; i < N; i++) {
        buf += to_string(pc[i].w); buf += ' ';
        buf += to_string(pc[i].h); buf += ' ';
        buf += to_string(pc[i].req); buf += '\n';
        if (buf.size() > (1u << 20)) { fputs(buf.c_str(), stdout); buf.clear(); }
    }
    fputs(buf.c_str(), stdout);
    printf("%d\n", (int)seams.size());
    buf.clear();
    for (auto& s : seams) {
        buf += to_string(s.first.first); buf += ' ';
        buf += to_string(s.first.second); buf += ' ';
        buf += to_string(s.second.first); buf += ' ';
        buf += to_string(s.second.second); buf += '\n';
        if (buf.size() > (1u << 20)) { fputs(buf.c_str(), stdout); buf.clear(); }
    }
    fputs(buf.c_str(), stdout);
    return 0;
}
