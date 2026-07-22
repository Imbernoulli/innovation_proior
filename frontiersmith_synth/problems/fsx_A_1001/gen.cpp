#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// ============================================================================
// Family: stuckat-test-compression (theme: find every broken wire with the
// fewest probes).  Mechanisms composed: fault-dominance-collapse +
// symmetry-aware-covering.
//
// A fault i is a CUBE: a small set of (control-line position, required bit)
// pairs.  A probe (a full K-bit vector) DETECTS fault i iff it agrees with
// every pair in the cube.  Minimize the number of probes so every fault is
// detected by some probe.
//
// Planted structure per test:
//   * CLIQUE block (b control lines): m_clique faults, each a DISTINCT full
//     b-bit pattern over the SAME b lines.  Any two distinct patterns must
//     disagree somewhere on this block, so no probe can satisfy two of them
//     at once -> this component genuinely NEEDS m_clique dedicated probes
//     (a hard floor, keeps the score ceiling open).
//   * ORBIT clusters: groups of faults on pairwise-DISJOINT single lines, all
//     requiring bit=1.  Because their lines never overlap they are ALWAYS
//     mutually compatible, so ONE probe (all those lines set to 1) detects an
//     entire orbit.  A naive per-fault "dedicate a 0-filled probe to each
//     fault" approach never notices this (0-filling a probe for member A
//     leaves member B's required line at 0 while B needs 1), and needs one
//     probe per member -- the trap.
//   * DOMINANCE chains: a "superset" cube plus several "dominated" cubes that
//     are literal position/value subsets of the superset.  Any probe that
//     satisfies the superset automatically satisfies every dominated cube
//     (classic ATPG fault dominance) -- an easy win a simple greedy DOES find
//     via real coverage simulation.
//   * FILLER: small random cubes on a shared noise region (generic texture,
//     occasional accidental overlap).
//
// testId is a difficulty ladder: N and K grow (mildly super-linearly) from
// testId=1 (tiny, example scale) to testId=10 (large, fills the constraint
// envelope).  All randomness is testlib's rnd, seeded solely by testId.
// ============================================================================

struct Fault { vector<pair<int,int>> req; };

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    double f = (t - 1) / 9.0;
    int Ntarget = (int)llround(15.0 * pow(120.0, f));   // 15 .. 1800
    if (Ntarget < 8) Ntarget = 8;
    if (Ntarget > 1800) Ntarget = 1800;

    double cliqueFrac = 0.15, domFrac = 0.30, orbitFrac = 0.32, fillerFrac = 0.23;

    int m_clique = max(2, (int)llround(Ntarget * cliqueFrac));
    int domTotalTarget  = (int)llround(Ntarget * domFrac);
    int orbitTotalTarget = (int)llround(Ntarget * orbitFrac);

    // ---- clique block size b: 2^b >= 4*m_clique, clamped ----
    int b = 2;
    while ((1 << b) < m_clique * 4 && b < 16) b++;
    int maxCodes = (1 << b);
    if (m_clique > maxCodes / 2) m_clique = max(2, maxCodes / 2);

    // ---- dominance chain params: fewer, larger chains for bigger tests
    //      (keeps total dedicated positions from growing too fast) ----
    int domPerChain = 3 + (t - 1);              // dominated faults per chain
    int domSupersetSize = 4 + (t % 3);           // 4..6
    int chainSize = 1 + domPerChain;
    int numDomChains = max(1, domTotalTarget / chainSize);
    int domTotal = numDomChains * chainSize;

    // ---- orbit params ----
    int numOrbits = 1 + t / 4;                   // 1..3
    vector<int> orbitSizes(numOrbits);
    {
        int rem = orbitTotalTarget;
        for (int i = 0; i < numOrbits; i++) {
            int sz = rem / (numOrbits - i);
            if (sz < 3) sz = 3;
            orbitSizes[i] = sz;
            rem -= sz;
        }
    }
    int orbitTotal = 0; for (int s : orbitSizes) orbitTotal += s;

    int fillerCount = Ntarget - m_clique - domTotal - orbitTotal;
    if (fillerCount < 3) fillerCount = 3;

    int noiseRegionSize = 8;

    // ---- position allocation (disjoint contiguous regions per mechanism) ----
    int nextPos = 0;
    int cliqueBase = nextPos; nextPos += b;
    vector<int> orbitBase(numOrbits);
    for (int i = 0; i < numOrbits; i++) { orbitBase[i] = nextPos; nextPos += orbitSizes[i]; }
    vector<int> domBase(numDomChains);
    for (int i = 0; i < numDomChains; i++) { domBase[i] = nextPos; nextPos += domSupersetSize; }
    int noiseBase = nextPos; nextPos += noiseRegionSize;
    int K = nextPos;

    vector<Fault> faults;
    faults.reserve(m_clique + orbitTotal + domTotal + fillerCount + 8);

    // ---- clique: m_clique distinct full b-bit patterns ----
    {
        set<int> used;
        while ((int)used.size() < m_clique) used.insert(rnd.next(0, maxCodes - 1));
        for (int code : used) {
            Fault fl;
            for (int k = 0; k < b; k++) fl.req.push_back({cliqueBase + k, (code >> k) & 1});
            faults.push_back(fl);
        }
    }

    // ---- orbits: disjoint single lines, all requiring bit=1 ----
    for (int i = 0; i < numOrbits; i++)
        for (int j = 0; j < orbitSizes[i]; j++) {
            Fault fl; fl.req.push_back({orbitBase[i] + j, 1});
            faults.push_back(fl);
        }

    // ---- dominance chains: superset + several proper-subset dominated cubes ----
    for (int c = 0; c < numDomChains; c++) {
        int base = domBase[c];
        vector<int> vals(domSupersetSize);
        for (int k = 0; k < domSupersetSize; k++) vals[k] = rnd.next(0, 1);
        Fault sup;
        for (int k = 0; k < domSupersetSize; k++) sup.req.push_back({base + k, vals[k]});
        faults.push_back(sup);
        for (int d = 0; d < domPerChain; d++) {
            int subsize = 1 + rnd.next(0, max(0, domSupersetSize - 2));
            vector<int> idx(domSupersetSize); iota(idx.begin(), idx.end(), 0);
            for (int x = (int)idx.size() - 1; x > 0; x--) { int y = rnd.next(0, x); swap(idx[x], idx[y]); }
            Fault fl;
            for (int k = 0; k < subsize; k++) { int p = idx[k]; fl.req.push_back({base + p, vals[p]}); }
            faults.push_back(fl);
        }
    }

    // ---- filler: small random cubes on the shared noise region ----
    for (int i = 0; i < fillerCount; i++) {
        int csize = 2 + rnd.next(0, min(3, noiseRegionSize - 2));
        vector<int> idx(noiseRegionSize); iota(idx.begin(), idx.end(), 0);
        for (int x = (int)idx.size() - 1; x > 0; x--) { int y = rnd.next(0, x); swap(idx[x], idx[y]); }
        Fault fl;
        for (int k = 0; k < csize; k++) fl.req.push_back({noiseBase + idx[k], rnd.next(0, 1)});
        faults.push_back(fl);
    }

    int N = (int)faults.size();

    // ---- shuffle fault order so mechanisms are interleaved (Fisher-Yates via rnd) ----
    for (int x = N - 1; x > 0; x--) { int y = rnd.next(0, x); swap(faults[x], faults[y]); }

    // ---- emit ----
    string out;
    out.reserve((size_t)N * 24 + 64);
    char line[64];
    int hlen = sprintf(line, "%d %d\n", K, N);
    out.append(line, hlen);
    for (auto &fl : faults) {
        int clen = sprintf(line, "%d", (int)fl.req.size());
        out.append(line, clen);
        for (auto &pr : fl.req) {
            clen = sprintf(line, " %d %d", pr.first, pr.second);
            out.append(line, clen);
        }
        out.push_back('\n');
    }
    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
