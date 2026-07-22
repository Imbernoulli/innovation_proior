#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// ============================================================================
// Family: purge-carryover-scheduling  (theme: rainbow 3D print that bleeds
// between colors)
//
// L printed layers, each a set of regions. Every region carries a TOLERANCE
// CLASS: a short list (2-3) of concrete nozzle colors that satisfy that
// region's design spec -- the RECOLORING degree of freedom. The solver
// chooses one concrete color per region AND a print order within each
// layer; layers themselves print in fixed order 1..L (the plate's build
// direction cannot be reordered). Cost is charged by an ASYMMETRIC purge
// matrix P for every consecutive nozzle-color transition in the WHOLE job,
// including the CARRYOVER transition from a layer's last-printed color into
// the next layer's first-printed color.
//
// The purge matrix always has one drastically expensive direction: crossing
// from the DARK half of the palette (color ids >= C/2) into the LIGHT half
// (color ids < C/2) costs a huge constant BIG plus a small linear term;
// every other direction (light->light, light->dark, dark->dark) is cheap.
// This is deliberately asymmetric and matches the theme: a dark pigment
// residue bleeding into a light color ruins it, so it needs a full purge,
// while a light residue bleeding into a darker color is invisible.
//
// Innovation hook: a handful of ANCHOR regions per test are hard-locked to
// one tier (their tolerance class lists ONLY dark, or ONLY light, concrete
// colors) -- these force a small, BOUNDED number of genuine dark<-light
// pivots regardless of strategy. Every other region is FLEX: its tolerance
// class deliberately contains one option from EACH tier, so the solver can
// RECOLOR it to whichever tier keeps the current print run on one side of
// the expensive purge. A bounded "trap" subset of flex regions lists the
// WRONG tier first in its class (a plausible-looking nominal color that is
// actually mis-aligned with the run currently in effect) -- a solver that
// just takes "the tolerance class's first listed color" (never exploiting
// the recoloring freedom) will silently eat that trap's extra dark<->light
// purges over and over, while a solver that reasons about which tier the
// CURRENT run is committed to -- and recolors flex regions into that tier
// regardless of what is listed first -- pays only the unavoidable anchor
// pivots. This is the intended lever: reshaping the transition graph via
// discrete recoloring, not touring the fixed graph better.
// ============================================================================

struct RegionSpec { vector<int> cls; int layer, slot; bool isFlex; };

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- per-test size/structure ladder ----
    int Ltab[11]      = {0,   3,   6,  10,  20,  30,  15,  50, 100, 200, 300};
    int Ctab[11]      = {0,   4,   6,   6,   8,   8,   6,  10,  10,  12,  12};
    int RloTab[11]    = {0,   2,   3,   3,   4,   4,   3,   5,   5,   5,  10};
    int RhiTab[11]    = {0,   3,   5,   6,   8,  10,   6,  12,  15,  20,  25};
    int Atab[11]       = {0,   1,   2,   3,   4,   5,   1,   6,   8,  10,  12};
    int TrapTab[11]    = {0,   1,   2,   2,   4,   5,   8,   6,   8,  10,  12};

    int L = Ltab[testId];
    int C = Ctab[testId];
    int Rlo = RloTab[testId], Rhi = RhiTab[testId];
    int A = Atab[testId];
    int trapBudget = TrapTab[testId];
    int half = C / 2; // colors [0,half-1]=LIGHT, [half,C-1]=DARK

    // ---- purge matrix ----
    ll BIG = 350 + rnd.next(0, 150);
    vector<vector<ll>> P(C, vector<ll>(C, 0));
    for (int i = 0; i < C; i++)
        for (int j = 0; j < C; j++) {
            if (i == j) { P[i][j] = 0; continue; }
            bool darkToLight = (i >= half) && (j < half);
            if (darkToLight) P[i][j] = BIG + 6 * (ll)abs(i - j);
            else             P[i][j] = 3 + (ll)abs(i - j);
        }

    // ---- decide anchor layers (evenly spread, alternating forced tier) ----
    // anchor i (0-indexed): even -> forces DARK, odd -> forces LIGHT
    set<int> anchorLayerSet;
    vector<int> anchorLayer(A), anchorTier(A); // tier: 0=LIGHT,1=DARK
    for (int i = 0; i < A; i++) {
        int lay = (int)((ll)(i + 1) * L / (A + 1));
        if (lay < 0) lay = 0;
        if (lay >= L) lay = L - 1;
        while (anchorLayerSet.count(lay) && lay + 1 < L) lay++;
        anchorLayerSet.insert(lay);
        anchorLayer[i] = lay;
        anchorTier[i] = (i % 2 == 0) ? 1 : 0; // DARK, LIGHT, DARK, LIGHT, ...
    }
    map<int,int> layerAnchorTier; // layer -> forced tier
    for (int i = 0; i < A; i++) layerAnchorTier[anchorLayer[i]] = anchorTier[i];

    // ---- build all layers ----
    vector<vector<vector<int>>> layerRegions(L); // layerRegions[l][r] = class list
    vector<pair<int,int>> flexCandidates; // (layer, region) that are flex -> eligible for trap flip

    int phase = 0; // 0=LIGHT, 1=DARK, running commitment used only to pick a SENSIBLE nominal
    for (int l = 0; l < L; l++) {
        int Rl = rnd.next(Rlo, Rhi);
        layerRegions[l].resize(Rl);

        bool isAnchorLayer = layerAnchorTier.count(l);
        int forcedTier = isAnchorLayer ? layerAnchorTier[l] : -1;
        int nominal = phase; // tier used for this layer's flex regions (old phase, pre-pivot)

        int startR = 0;
        if (isAnchorLayer) {
            // region 0 = forced anchor, entirely within forcedTier
            int lo = forcedTier ? half : 0;
            int hi = forcedTier ? (C - 1) : (half - 1);
            int poolSize = hi - lo + 1;
            int k = min(rnd.next(2, 3), poolSize);
            set<int> chosen;
            while ((int)chosen.size() < k) chosen.insert(rnd.next(lo, hi));
            vector<int> cls(chosen.begin(), chosen.end());
            shuffle(cls.begin(), cls.end());
            layerRegions[l][0] = cls;
            startR = 1;
            phase = forcedTier; // commitment flips for subsequent layers
        }

        for (int r = startR; r < Rl; r++) {
            int loN = nominal ? half : 0, hiN = nominal ? (C - 1) : (half - 1);
            int loO = nominal ? 0 : half, hiO = nominal ? (half - 1) : (C - 1);
            int primary = rnd.next(loN, hiN);
            int secondary = rnd.next(loO, hiO);
            vector<int> cls = {primary, secondary};
            int k = rnd.next(2, 3);
            if (k == 3 && hiN > loN) {
                // extra same-tier option, guaranteed distinct from primary
                int extra;
                do { extra = rnd.next(loN, hiN); } while (extra == primary);
                cls.push_back(extra);
            }
            layerRegions[l][r] = cls; // primary (nominal-aligned) listed FIRST by default
            flexCandidates.push_back({l, r});
        }
    }

    // ---- flip a bounded random subset of flex regions to list the WRONG
    //      (opposite-tier) color first -- the trap ----
    shuffle(flexCandidates.begin(), flexCandidates.end());
    int nTrap = min((int)flexCandidates.size(), trapBudget);
    for (int i = 0; i < nTrap; i++) {
        auto [l, r] = flexCandidates[i];
        vector<int>& cls = layerRegions[l][r];
        swap(cls[0], cls[1]); // secondary (opposite tier) now listed first
    }

    // ---- shuffle each layer's region LISTING order (so a naive "print in
    //      input order" baseline gets no free structure) ----
    for (int l = 0; l < L; l++) {
        // shuffle region slots but keep class assignments attached
        vector<int> idx(layerRegions[l].size());
        iota(idx.begin(), idx.end(), 0);
        shuffle(idx.begin(), idx.end());
        vector<vector<int>> tmp(layerRegions[l].size());
        for (size_t i = 0; i < idx.size(); i++) tmp[i] = layerRegions[l][idx[i]];
        layerRegions[l] = tmp;
    }

    // ---- emit ----
    string out;
    out.reserve(1 << 20);
    char buf[256];
    int len = sprintf(buf, "%d %d\n", L, C);
    out.append(buf, len);
    for (int i = 0; i < C; i++) {
        string row;
        for (int j = 0; j < C; j++) {
            len = sprintf(buf, j + 1 == C ? "%lld\n" : "%lld ", P[i][j]);
            row.append(buf, len);
        }
        out += row;
    }
    for (int l = 0; l < L; l++) {
        int Rl = (int)layerRegions[l].size();
        len = sprintf(buf, "%d\n", Rl);
        out.append(buf, len);
        for (int r = 0; r < Rl; r++) {
            auto& cls = layerRegions[l][r];
            len = sprintf(buf, "%d", (int)cls.size());
            out.append(buf, len);
            for (int c : cls) { len = sprintf(buf, " %d", c); out.append(buf, len); }
            out += "\n";
        }
    }
    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
