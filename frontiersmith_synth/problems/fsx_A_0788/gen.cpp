#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Warren-truss (no verticals) geometry generator for "Size truss members for a
// target deflection". Builds a statically-determinate base truss (M+R=2N, always
// rigid for any positive bar areas), then optionally grafts short "decoy" braces
// at the load point(s): a nearby auxiliary node connected by three SHORT bars to
// the truss. A decoy bar carries real force (funnels the applied load) but its
// short length gives it tiny elongation/strain-energy even at minimum area, so
// enlarging it barely helps match the target deflection -- a force-only sizing
// rule (fully-stressed design) wastes budget there; a global energy/sensitivity
// view correctly ignores it.

struct TestParams {
    int P;            // panel count of the base Warren truss
    int decoys;        // number of decoy braces to graft on
    int mainLoads;     // number of "main" (non-decoy) loaded top-chord nodes
    int regime;        // 0 normal, 1 cost-heavy, 2 disp-heavy, 3 mixed
};

TestParams paramsFor(int t) {
    switch (t) {
        case 1: return {2, 0, 1, 0};
        case 2: return {3, 0, 1, 0};
        case 3: return {5, 1, 1, 0};
        case 4: return {6, 0, 2, 0};
        case 5: return {8, 2, 1, 0};
        case 6: return {10, 1, 1, 0};
        case 7: return {14, 1, 1, 1};
        case 8: return {18, 1, 1, 2};
        case 9: return {24, 3, 2, 0};
        default: return {30, 3, 2, 3};
    }
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    TestParams tp = paramsFor(testId);
    int P = tp.P;

    int dx = 40 + testId * 6 + rnd.next(0, 10);
    int h = 25 + testId * 4 + rnd.next(0, 8);

    // ---- base Warren truss ----
    // bottom nodes b[0..P], top nodes t[0..P-1]
    vector<ll> X, Y; vector<int> SUP;
    vector<int> bId(P + 1), tId(P);
    for (int i = 0; i <= P; i++) {
        bId[i] = (int)X.size();
        X.push_back((ll)i * dx);
        Y.push_back(0);
        if (i == 0) SUP.push_back(1);       // pinned
        else if (i == P) SUP.push_back(2);  // roller (fix y)
        else SUP.push_back(0);
    }
    for (int i = 0; i < P; i++) {
        tId[i] = (int)X.size();
        X.push_back((ll)i * dx + dx / 2);
        Y.push_back(h);
        SUP.push_back(0);
    }

    vector<int> BA, BB;
    for (int i = 0; i < P; i++) { BA.push_back(bId[i]); BB.push_back(bId[i + 1]); }       // bottom chord
    for (int i = 0; i < P; i++) { BA.push_back(bId[i]); BB.push_back(tId[i]); }           // diagonal /
    for (int i = 0; i < P; i++) { BA.push_back(tId[i]); BB.push_back(bId[i + 1]); }       // diagonal \.
    for (int i = 0; i + 1 < P; i++) { BA.push_back(tId[i]); BB.push_back(tId[i + 1]); }   // top chord

    // ---- loaded nodes: main loads + decoy braces ----
    struct LoadSpec { int node; ll fx, fy, tx, ty; };
    vector<LoadSpec> loads;

    // choose main load top-chord nodes (spread near mid-span)
    vector<int> mainNodes;
    if (tp.mainLoads == 1) {
        mainNodes.push_back(tId[P / 2]);
    } else {
        int a = max(0, P / 2 - 1), b = min(P - 1, P / 2 + 1);
        mainNodes.push_back(tId[a]);
        mainNodes.push_back(tId[b]);
    }

    // decoy anchor top-chord indices, spread across span, avoiding mainNodes' indices
    vector<int> decoyAnchors;
    if (tp.decoys > 0) {
        vector<int> cand;
        for (int i = 0; i < P; i++) {
            bool used = false;
            for (int mn : mainNodes) if (tId[i] == mn) used = true;
            if (!used) cand.push_back(i);
        }
        shuffle(cand.begin(), cand.end());
        for (int k = 0; k < tp.decoys && k < (int)cand.size(); k++) decoyAnchors.push_back(cand[k]);
    }

    ll K_ = 5; // catalog size
    ll baseArea = 8 + testId + rnd.next(0, 3);
    vector<ll> AREA(K_), COST(K_);
    for (int k = 0; k < K_; k++) {
        double a = (double)baseArea * pow(1.7, (double)k);
        AREA[k] = max<ll>(1, llround(a));
    }
    sort(AREA.begin(), AREA.end());
    for (int k = 1; k < K_; k++) if (AREA[k] <= AREA[k - 1]) AREA[k] = AREA[k - 1] + 1;
    double costScale = 0.35 + 0.05 * (rnd.next(0, 6));
    for (int k = 0; k < K_; k++) {
        double c = costScale * pow((double)AREA[k], 1.35);
        COST[k] = max<ll>(1, llround(c));
    }
    sort(COST.begin(), COST.end());
    for (int k = 1; k < K_; k++) if (COST[k] <= COST[k - 1]) COST[k] = COST[k - 1] + 1;

    ll E = 500 + testId * 40 + rnd.next(0, 200);

    ll Lest = dx; // typical global bar length scale for target estimation

    auto estimateTarget = [&](ll Fy, ll Fx, ll &Ty, ll &Tx) {
        double soft_y = (double)Fy * (double)Lest / ((double)E * (double)AREA[0]);
        double stiff_y = (double)Fy * (double)Lest / ((double)E * (double)AREA[K_ - 1]);
        double mid_y = (soft_y + stiff_y) / 2.0;
        Ty = llround(mid_y);
        if (Fx != 0) {
            double soft_x = (double)Fx * (double)Lest / ((double)E * (double)AREA[0]);
            double stiff_x = (double)Fx * (double)Lest / ((double)E * (double)AREA[K_ - 1]);
            Tx = llround((soft_x + stiff_x) / 2.0);
        } else Tx = 0;
    };

    for (int mn : mainNodes) {
        ll Fy = -(600 + 80 * testId + rnd.next(0, 200));
        ll Fx = (testId >= 4) ? (ll)(rnd.next(-100, 100)) : 0;
        ll Tx, Ty;
        estimateTarget(Fy, Fx, Ty, Tx);
        loads.push_back({mn, Fx, Fy, Tx, Ty});
    }

    for (int anchorIdx : decoyAnchors) {
        int anchor = tId[anchorIdx];
        ll offx = 2 + rnd.next(0, max(1, dx / 12));
        ll sy = rnd.next(0, 1) ? 1 : -1;
        ll offy = sy * (1 + rnd.next(0, max(1, h / 14)));
        int aux = (int)X.size();
        X.push_back(X[anchor] + offx);
        Y.push_back(Y[anchor] + offy);
        SUP.push_back(0);
        BA.push_back(aux); BB.push_back(anchor);
        BA.push_back(aux); BB.push_back(bId[anchorIdx]);
        BA.push_back(aux); BB.push_back(bId[anchorIdx + 1]);

        ll Fy = -(400 + rnd.next(0, 400));
        ll Fx = rnd.next(-80, 80);
        ll Tx, Ty;
        estimateTarget(Fy, Fx, Ty, Tx);
        loads.push_back({aux, Fx, Fy, Tx, Ty});
    }

    int N = (int)X.size();
    int M = (int)BA.size();
    int Lc = (int)loads.size();

    // ---- balance Wdisp against Wcost so neither term trivially dominates ----
    // Baseline (all bars at catalog index 0) cost is exactly COST[0]*totalLen; pick
    // Wdisp so that Wdisp*typicalTargetSwing sits at a chosen fraction of that cost,
    // making the target-mismatch and cost terms genuinely comparable in scale.
    double totalLen = 0.0;
    for (int i = 0; i < M; i++) {
        double dxv = (double)(X[BB[i]] - X[BA[i]]), dyv = (double)(Y[BB[i]] - Y[BA[i]]);
        totalLen += sqrt(dxv * dxv + dyv * dyv);
    }
    double baseCostTotal = (double)COST[0] * totalLen;
    double swingSum = 0.0;
    for (auto &ls : loads) swingSum += fabs((double)ls.ty) + fabs((double)ls.tx);
    double typicalSwing = max(1.0, swingSum / max(1, Lc));

    double costFraction = 0.6;
    if (tp.regime == 1) costFraction = 0.12;       // cost-heavy regime
    else if (tp.regime == 2) costFraction = 3.0;    // displacement-heavy regime
    else if (tp.regime == 3) costFraction = 1.0;    // mixed regime

    ll Wdisp = max<ll>(1, llround(baseCostTotal * costFraction / typicalSwing));
    ll Wcost = 1;

    printf("%d %d %d\n", N, M, (int)K_);
    for (int i = 0; i < N; i++) printf("%lld %lld %d\n", X[i], Y[i], SUP[i]);
    for (int i = 0; i < M; i++) printf("%d %d\n", BA[i] + 1, BB[i] + 1);
    for (int k = 0; k < K_; k++) printf("%lld %lld\n", AREA[k], COST[k]);
    printf("%lld\n", E);
    printf("%lld %lld\n", Wdisp, Wcost);
    printf("%d\n", Lc);
    for (auto &ls : loads) printf("%d %lld %lld %lld %lld\n", ls.node + 1, ls.fx, ls.fy, ls.tx, ls.ty);

    return 0;
}
