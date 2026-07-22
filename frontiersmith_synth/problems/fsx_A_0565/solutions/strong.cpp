// TIER: strong
// Insight: the non-crossing rule makes every feasible assignment a monotone partition of the
// bay axis into contiguous zones, so the whole search collapses to "where do the K-1 cuts
// go?".  Two things the load-balancing greedy misses:
//   (1) a cut placed inside a congested cluster (bay gap < S) forces the two cranes to
//       SERIALIZE -> the makespan T balloons; cutting at a WIDE gap keeps zones parallel;
//   (2) given the zone durations, T is fixed, and idle energy is minimized by matching high
//       standby-power cranes to the busiest zones -- an increasing-subsequence assignment DP.
// Strong seeds several partitions (work / count quantiles, snapped to clean gaps) then
// hill-climbs the cut positions against the EXACT objective, choosing the best crane subset
// by DP at every step.  Deterministic RNG.
#include <bits/stdc++.h>
using namespace std;

static int K, M;
static long long S, ALPHA, GAMMA, SUMP;
static vector<long long> P, pos, work, prefW;   // pos sorted
static vector<int> cleanGaps;                    // ranks r with pos[r]-pos[r-1] >= S

// zone [l,r): duration
static inline long long zoneD(int l, int r) {
    if (l >= r) return 0;
    return (prefW[r] - prefW[l]) + ALPHA * (pos[r - 1] - pos[l]);
}

// Given cut ranks (size z-1, sorted, strictly in (0,M)), compute F under the optimal crane
// assignment.  Returns F; if fillAssign != nullptr also fills zoneCrane[] (crane per zone).
static long long evalCuts(const vector<int>& cut, vector<int>* fillAssign) {
    int z = (int)cut.size() + 1;
    vector<int> bnd; bnd.push_back(0);
    for (int c : cut) bnd.push_back(c);
    bnd.push_back(M);
    vector<long long> Dz(z), mnp(z), mxp(z);
    for (int j = 0; j < z; j++) {
        Dz[j] = zoneD(bnd[j], bnd[j + 1]);
        mnp[j] = pos[bnd[j]];
        mxp[j] = pos[bnd[j + 1] - 1];
    }
    // congested stretches -> makespan T
    long long T = 0, run = Dz[0];
    for (int j = 1; j < z; j++) {
        long long gap = mnp[j] - mxp[j - 1];
        if (gap < S) run += Dz[j];
        else { T = max(T, run); run = Dz[j]; }
    }
    T = max(T, run);
    // crane assignment: maximize sum p_c * Dz over increasing subsequence of length z
    static long long f[9][9];
    static int par[9][9];
    for (int j = 0; j <= z; j++) f[0][j] = (j == 0) ? 0 : LLONG_MIN;
    for (int k = 1; k <= K; k++) {
        for (int j = 0; j <= z; j++) {
            f[k][j] = f[k - 1][j]; par[k][j] = 0;               // crane k-1 unused
            if (j >= 1 && f[k - 1][j - 1] != LLONG_MIN) {
                long long cand = f[k - 1][j - 1] + P[k - 1] * Dz[j - 1];
                if (cand > f[k][j]) { f[k][j] = cand; par[k][j] = 1; }  // crane k-1 -> zone j-1
            }
        }
    }
    long long bestSum = f[K][z];
    long long idle = T * SUMP - bestSum;
    long long F = T + GAMMA * idle;
    if (fillAssign) {
        fillAssign->assign(z, -1);
        int k = K, j = z;
        while (k > 0) {
            if (par[k][j] == 1) { (*fillAssign)[j - 1] = k - 1; j--; }
            k--;
        }
    }
    return F;
}

// nearest clean gap to rank r (returns r if none / clamps)
static int snapClean(int r) {
    if (cleanGaps.empty()) return r;
    int lo = 0, hi = (int)cleanGaps.size() - 1, best = cleanGaps[0];
    long long bd = LLONG_MAX;
    // binary search
    int it = lower_bound(cleanGaps.begin(), cleanGaps.end(), r) - cleanGaps.begin();
    for (int t = max(0, it - 1); t <= min((int)cleanGaps.size() - 1, it); t++) {
        long long d = llabs((long long)cleanGaps[t] - r);
        if (d < bd) { bd = d; best = cleanGaps[t]; }
    }
    return best;
}

static void dedupSort(vector<int>& v) {
    sort(v.begin(), v.end());
    v.erase(unique(v.begin(), v.end()), v.end());
}

int main() {
    if (scanf("%d %d %lld %lld %lld", &K, &M, &S, &ALPHA, &GAMMA) != 5) return 0;
    P.resize(K); SUMP = 0;
    for (int c = 0; c < K; c++) { scanf("%lld", &P[c]); SUMP += P[c]; }
    vector<long long> ipos(M), iwork(M);
    for (int i = 0; i < M; i++) scanf("%lld %lld", &ipos[i], &iwork[i]);

    vector<int> ord(M);
    for (int i = 0; i < M; i++) ord[i] = i;
    sort(ord.begin(), ord.end(), [&](int a, int b){ return ipos[a] < ipos[b]; });
    pos.resize(M); work.resize(M);
    for (int r = 0; r < M; r++) { pos[r] = ipos[ord[r]]; work[r] = iwork[ord[r]]; }
    prefW.assign(M + 1, 0);
    for (int r = 0; r < M; r++) prefW[r + 1] = prefW[r] + work[r];
    long long total = prefW[M];
    for (int r = 1; r < M; r++) if (pos[r] - pos[r - 1] >= S) cleanGaps.push_back(r);

    vector<int> bestCut; long long bestF = LLONG_MAX;
    auto consider = [&](const vector<int>& cut) {
        long long f = evalCuts(cut, nullptr);
        if (f < bestF) { bestF = f; bestCut = cut; }
    };

    int maxZ = min(K, M);
    // ---- seed partitions ----
    for (int z = 1; z <= maxZ; z++) {
        // work-quantile
        vector<int> wq, cq, wqs;
        for (int k = 1; k < z; k++) {
            long long target = (long long)k * total / z;
            int r = (int)(lower_bound(prefW.begin(), prefW.end(), target) - prefW.begin());
            r = max(1, min(M - 1, r));
            wq.push_back(r);
            wqs.push_back(max(1, min(M - 1, snapClean(r))));
            cq.push_back(max(1, min(M - 1, (int)((long long)k * M / z))));
        }
        dedupSort(wq); dedupSort(cq); dedupSort(wqs);
        if ((int)wq.size()  == z - 1) consider(wq);
        if ((int)cq.size()  == z - 1) consider(cq);
        if ((int)wqs.size() == z - 1) consider(wqs);
    }

    // ---- hill climb from best, trying clean gaps + local nudges ----
    auto hillClimb = [&](vector<int> cut) {
        bool improved = true; int passes = 0;
        while (improved && passes < 40) {
            improved = false; passes++;
            for (int t = 0; t < (int)cut.size(); t++) {
                int orig = cut[t];
                long long curF = evalCuts(cut, nullptr);
                int bestPos = orig; long long bf = curF;
                // candidate positions: nearby clean gaps + small nudges
                vector<int> cands;
                for (int d = -2; d <= 2; d++) cands.push_back(orig + d);
                if (!cleanGaps.empty()) {
                    int it = lower_bound(cleanGaps.begin(), cleanGaps.end(), orig) - cleanGaps.begin();
                    for (int q = it - 3; q <= it + 3; q++)
                        if (q >= 0 && q < (int)cleanGaps.size()) cands.push_back(cleanGaps[q]);
                }
                int lo = (t == 0) ? 1 : cut[t - 1] + 1;
                int hi = (t + 1 == (int)cut.size()) ? M - 1 : cut[t + 1] - 1;
                for (int np : cands) {
                    if (np < lo || np > hi) continue;
                    cut[t] = np;
                    long long f = evalCuts(cut, nullptr);
                    if (f < bf) { bf = f; bestPos = np; }
                }
                cut[t] = bestPos;
                if (bestPos != orig) improved = true;
            }
        }
        consider(cut);
    };
    if (!bestCut.empty() || bestF != LLONG_MAX) hillClimb(bestCut);

    // ---- deterministic multistart over clean-gap subsets ----
    unsigned long long rng = 0x9e3779b97f4a7c15ULL ^ (unsigned long long)(M * 1000003 + K);
    auto nextR = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    int restarts = (int)min<long long>(3000, 200000LL / max(1, K));
    for (int it = 0; it < restarts; it++) {
        int z = 1 + (int)(nextR() % maxZ);
        vector<int> cut;
        for (int k = 0; k < z - 1; k++) {
            int r;
            if (!cleanGaps.empty() && (nextR() & 1))
                r = cleanGaps[nextR() % cleanGaps.size()];
            else
                r = 1 + (int)(nextR() % (M - 1));
            cut.push_back(max(1, min(M - 1, r)));
        }
        dedupSort(cut);
        if ((int)cut.size() != z - 1) continue;
        long long f = evalCuts(cut, nullptr);
        if (f < bestF * 12 / 10) hillClimb(cut);   // only refine promising starts
        else consider(cut);
    }

    // ---- reconstruct assignment + offsets for bestCut ----
    vector<int> zoneCrane;
    evalCuts(bestCut, &zoneCrane);
    int z = (int)bestCut.size() + 1;
    vector<int> bnd; bnd.push_back(0);
    for (int c : bestCut) bnd.push_back(c);
    bnd.push_back(M);

    vector<int> craneOfSorted(M);
    for (int j = 0; j < z; j++)
        for (int r = bnd[j]; r < bnd[j + 1]; r++) craneOfSorted[r] = zoneCrane[j];

    // per-crane D, min/max, offsets
    vector<long long> W(K,0), mn(K,LLONG_MAX), mx(K,LLONG_MIN), D(K,0);
    vector<char> ne(K,0);
    for (int r=0;r<M;r++){ int c=craneOfSorted[r]; ne[c]=1; W[c]+=work[r]; mn[c]=min(mn[c],pos[r]); mx[c]=max(mx[c],pos[r]); }
    for (int c=0;c<K;c++) if(ne[c]) D[c]=W[c]+ALPHA*(mx[c]-mn[c]);
    vector<int> order; for(int c=0;c<K;c++) if(ne[c]) order.push_back(c);
    vector<long long> s(K,0);
    long long runClock=0, prevMax=LLONG_MIN; bool first=true;
    for (int c : order) {
        long long gap = first ? (S+1) : (mn[c]-prevMax);
        if (first || gap>=S) { s[c]=0; runClock=D[c]; }
        else { s[c]=runClock; runClock+=D[c]; }
        prevMax=mx[c]; first=false;
    }

    // map back to input order
    vector<int> craneOfInput(M);
    for (int r = 0; r < M; r++) craneOfInput[ord[r]] = craneOfSorted[r];

    for (int i=0;i<M;i++) printf("%d%c", craneOfInput[i]+1, i+1<M?' ':'\n');
    for (int c=0;c<K;c++) printf("%lld%c", s[c], c+1<K?' ':'\n');
    return 0;
}
