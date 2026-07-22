#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Ferry Ropes Across the Two Riverbank Towns".
//
// Input:  M Q B ; then M lane blocks (n_c, then n_c strictly increasing longs) ;
//         then Q query blocks (k, then k distinct 1-indexed lane ids, k in {2,3}).
// Output: for c=1..M: k_c, then k_c pairs "pos target" (0-indexed lane positions,
//         0 <= pos < target <= n_c-1). Global sum of k_c over all lanes <= B.
//
// The FIXED dispatcher (identical logic used for baseline and for the
// participant's ropes): for a query over lanes c_1..c_k with pointers p_i=0,
// repeat: if any p_i==n_{c_i} the convoy ends. Let v=max(L[c_i][p_i]). If all
// equal v: it's a joint layover -- advance every p_i by 1 (k probes). Else, for
// every i with L[c_i][p_i] < v: if a rope exists at position p_i on lane c_i
// with target t and L[c_i][t] <= v (does not overshoot), jump p_i = t (1 probe);
// otherwise walk p_i += 1 (1 probe).
//
// Objective (MIN): F = total probes summed over all Q convoys.
// Baseline B (checker-computed): the SAME dispatcher run with NO ropes at all
//   (the do-nothing construction -- exactly what the trivial reference submits).
// Score (min): sc = min(1000, 100*Bval/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static vector<vector<ll>> lane;         // lane[c] = sorted markers, 1-indexed c
static vector<vector<int>> qch;         // qch[q] = lane ids in that convoy

static ll simulateAll(const vector<vector<ll>> &ropeTarget /* ropeTarget[c][pos] or -1, size n_c; empty vector = no ropes for that lane */) {
    ll totalProbes = 0;
    for (auto &ch : qch) {
        int k = (int)ch.size();
        vector<ll> p(k, 0);
        vector<ll> n(k);
        for (int i = 0; i < k; i++) n[i] = (ll)lane[ch[i]].size();
        ll roundCap = 1;
        for (int i = 0; i < k; i++) roundCap += n[i];
        ll rounds = 0;
        while (true) {
            bool exhausted = false;
            for (int i = 0; i < k; i++) if (p[i] >= n[i]) { exhausted = true; break; }
            if (exhausted) break;
            if (++rounds > roundCap + 5) quitf(_wa, "internal: dispatcher did not terminate (bug)");
            ll v = -1;
            for (int i = 0; i < k; i++) v = max(v, lane[ch[i]][p[i]]);
            bool allEq = true;
            for (int i = 0; i < k; i++) if (lane[ch[i]][p[i]] != v) { allEq = false; break; }
            if (allEq) {
                for (int i = 0; i < k; i++) p[i]++;
                totalProbes += k;
                continue;
            }
            for (int i = 0; i < k; i++) {
                if (lane[ch[i]][p[i]] == v) continue; // this pointer holds the max, doesn't move this round
                int c = ch[i];
                ll pos = p[i];
                ll used = -1;
                if (!ropeTarget.empty() && !ropeTarget[c].empty() && ropeTarget[c][pos] >= 0) {
                    ll t = ropeTarget[c][pos];
                    if (lane[c][t] <= v) used = t;
                }
                if (used >= 0) p[i] = used; else p[i] = pos + 1;
                totalProbes++;
            }
        }
    }
    return totalProbes;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int M = inf.readInt();
    int Q = inf.readInt();
    int Bbudget = inf.readInt();
    lane.assign(M + 1, {});
    for (int c = 1; c <= M; c++) {
        int n = inf.readInt();
        lane[c].resize(n);
        for (int j = 0; j < n; j++) lane[c][j] = inf.readLong();
    }
    qch.assign(Q, {});
    for (int q = 0; q < Q; q++) {
        int k = inf.readInt();
        qch[q].resize(k);
        for (int i = 0; i < k; i++) qch[q][i] = inf.readInt();
    }

    // ---- read + validate participant's ropes ----
    vector<vector<ll>> ropeTarget(M + 1);
    ll totalRopes = 0;
    for (int c = 1; c <= M; c++) {
        int n = (int)lane[c].size();
        int kc = ouf.readInt(0, n, "k_c");
        ropeTarget[c].assign(n, -1);
        set<ll> seenPos;
        for (int r = 0; r < kc; r++) {
            ll pos = ouf.readLong(0, (ll)n - 1, "pos");
            if (seenPos.count(pos)) quitf(_wa, "lane %d: duplicate rope position %lld", c, pos);
            seenPos.insert(pos);
            ll lo = pos + 1, hi = (ll)n - 1;
            if (lo > hi) quitf(_wa, "lane %d: rope at pos %lld has no valid target (lane too short)", c, pos);
            ll target = ouf.readLong(lo, hi, "target");
            ropeTarget[c][pos] = target;
        }
        totalRopes += kc;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");
    if (totalRopes > Bbudget)
        quitf(_wa, "rope budget exceeded: used %lld > B=%d", totalRopes, Bbudget);

    ll F = simulateAll(ropeTarget);
    vector<vector<ll>> none(M + 1);
    ll Bval = simulateAll(none);
    if (Bval <= 0) Bval = 1;

    double sc = min(1000.0, 100.0 * (double)Bval / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld ropes=%lld/%d Ratio: %.6f",
          F, Bval, totalRopes, Bbudget, sc / 1000.0);
    return 0;
}
