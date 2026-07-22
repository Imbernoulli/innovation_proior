#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int K, M;
ll S, R;
vector<int> inT, outT, outR, byT, byR, fp, cnt;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    K = inf.readInt();
    M = inf.readInt();
    S = inf.readLong();
    R = inf.readLong();

    inT.resize(M); outT.resize(M); outR.resize(M); byT.resize(M); byR.resize(M); fp.resize(M); cnt.resize(M);
    for (int i = 0; i < M; i++) {
        inT[i] = inf.readInt();
        outT[i] = inf.readInt();
        outR[i] = inf.readInt();
        byT[i] = inf.readInt();
        byR[i] = inf.readInt();
        fp[i] = inf.readInt();
        cnt[i] = inf.readInt();
    }

    // ---- internal baseline B: the "obvious" single-instance chain, cheapest module
    //      id found per hop (0->1->...->K-1), byproducts entirely ignored. ----
    double bRatio = 1.0;
    for (int t = 0; t < K - 1; t++) {
        int best = -1;
        for (int i = 0; i < M; i++) if (inT[i] == t && outT[i] == t + 1) { best = i; break; }
        if (best < 0) quitf(_fail, "bad instance: no module for hop %d->%d", t, t + 1);
        bRatio *= outR[best] / 100.0;
    }
    double B = (double)R * bRatio;
    if (!(B > 0)) quitf(_fail, "bad instance: B=%.6f", B);

    // ---- read participant output ----
    int P = ouf.readInt(0, 4000, "P");
    int E = ouf.readInt(0, 40000, "E");

    vector<int> mtype(P);
    vector<ll> footprintUsed(0);
    vector<int> countUsed(M, 0);
    ll totalFp = 0;
    for (int j = 0; j < P; j++) {
        mtype[j] = ouf.readInt(0, M - 1, "moduleType");
        countUsed[mtype[j]]++;
        totalFp += fp[mtype[j]];
    }
    if (totalFp > S) quitf(_wa, "footprint budget exceeded: used=%lld budget=%lld", totalFp, S);
    for (int i = 0; i < M; i++)
        if (countUsed[i] > cnt[i]) quitf(_wa, "module type %d used %d times, limit %d", i, countUsed[i], cnt[i]);

    vector<double> totalIn(P, 0.0), totalOutMain(P, 0.0), totalOutBy(P, 0.0);
    double rawUsed = 0.0, F = 0.0;

    const double EPS_ABS = 1e-6;
    const double EPS_REL = 1e-6;

    for (int e = 0; e < E; e++) {
        int sk = ouf.readInt(0, 1, "sk");
        int sid = ouf.readInt(-1, P - 1, "sid");
        int port = ouf.readInt(0, 1, "port");
        int dk = ouf.readInt(0, 1, "dk");
        int did = ouf.readInt(-1, P - 1, "did");
        double amount = ouf.readDouble(0.0, 1e7, "amount");
        if (!isfinite(amount)) quitf(_wa, "non-finite amount on edge %d", e);

        int edgeType;
        if (sk == 0) {
            edgeType = 0; // RAW supplies type 0 only
            rawUsed += amount;
        } else {
            if (sid < 0 || sid >= P) quitf(_wa, "edge %d: invalid source instance id", e);
            int mt = mtype[sid];
            if (port == 1) {
                if (byT[mt] < 0) quitf(_wa, "edge %d: instance %d's module has no byproduct port", e, sid);
                edgeType = byT[mt];
                totalOutBy[sid] += amount;
            } else {
                edgeType = outT[mt];
                totalOutMain[sid] += amount;
            }
        }

        if (dk == 0) {
            if (edgeType != K - 1) quitf(_wa, "edge %d: only type %d may reach the sink, got type %d", e, K - 1, edgeType);
            F += amount;
        } else {
            if (did < 0 || did >= P) quitf(_wa, "edge %d: invalid destination instance id", e);
            int mt = mtype[did];
            if (edgeType != inT[mt]) quitf(_wa, "edge %d: type %d cannot feed instance %d (needs type %d)", e, edgeType, did, inT[mt]);
            totalIn[did] += amount;
        }
    }

    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    if (rawUsed > R + EPS_ABS + EPS_REL * R) quitf(_wa, "raw feedstock over-used: used=%.6f budget=%lld", rawUsed, R);

    for (int j = 0; j < P; j++) {
        int mt = mtype[j];
        double allowMain = totalIn[j] * (outR[mt] / 100.0);
        double tolMain = EPS_ABS + EPS_REL * fabs(allowMain);
        if (totalOutMain[j] > allowMain + tolMain)
            quitf(_wa, "instance %d over-produces main output: out=%.6f allowed=%.6f", j, totalOutMain[j], allowMain);
        if (byT[mt] >= 0) {
            double allowBy = totalIn[j] * (byR[mt] / 100.0);
            double tolBy = EPS_ABS + EPS_REL * fabs(allowBy);
            if (totalOutBy[j] > allowBy + tolBy)
                quitf(_wa, "instance %d over-produces byproduct: out=%.6f allowed=%.6f", j, totalOutBy[j], allowBy);
        } else {
            if (totalOutBy[j] > EPS_ABS)
                quitf(_wa, "instance %d has no byproduct port", j);
        }
    }

    double sc = min(1000.0, 100.0 * F / max(1e-9, B));
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
