// TIER: strong
// Insight: this is a common-pool problem in temperature space. Don't rank
// segments by raw temperature and drain each one hard -- for any candidate
// SET of installed segments, the right depths satisfy an equalized marginal
// value dE/dx = eta_i*(Tpost_i-Tsink)/Tpost_i = lambda (a shared shadow price)
// across every active unit. For a fixed lambda we can solve each unit's own
// outlet temperature in closed form (Tpost_i = Tsink/(1-lambda/eta_i)) and
// forward-simulate the whole chain (so downstream mixing correctly reflects
// upstream restraint); a ternary search over lambda finds the depth profile
// that maximizes THIS set's total output. We then greedily forward-select up
// to K segments from a generous candidate pool (local temperature peaks plus
// the hottest-by-raw-temperature segments), each time keeping whichever
// addition raises the properly-computed total the most. This deliberately
// leaves upstream units under-drained to preserve heat for the rest of the
// chain -- the opposite of the greedy trap.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static int T, K, Tsink;
static vector<ll> q, theta, eta, cap;
static vector<double> pre;

static double energyOf(double e, double F, double Tpre, double Tpost) {
    if (Tpost > Tpre) Tpost = Tpre;
    if (Tpost < Tsink) Tpost = Tsink;
    if (Tpost >= Tpre) return 0.0;
    return e * F * ((Tpre - Tpost) - Tsink * log(Tpre / Tpost));
}

// Forward-simulate the whole chain for a fixed active-set W at shadow price
// lambda. If outX != null, also record each active segment's extraction.
static double evalLambda(const vector<char>& inW, double lambda, vector<ll>* outX) {
    double F = 0.0, Tcur = 0.0, Etot = 0.0;
    for (int i = 1; i <= T; i++) {
        double Fnew = F + (double)q[i];
        double Tpre = (F == 0.0) ? (double)theta[i] : (F * Tcur + (double)q[i] * theta[i]) / Fnew;
        F = Fnew;
        if (inW[i]) {
            double e = eta[i] / 1000.0;
            double v0 = e * (Tpre - Tsink) / Tpre;
            double Tpost;
            if (v0 > lambda) {
                Tpost = Tsink / (1.0 - lambda / e);
                if (Tpost > Tpre) Tpost = Tpre;
                if (Tpost < Tsink) Tpost = Tsink;
            } else {
                Tpost = Tpre;
            }
            double xdes = F * (Tpre - Tpost);
            double feasMax = F * (Tpre - Tsink);
            double xcap = min(min(xdes, (double)cap[i]), feasMax);
            if (xcap < 0) xcap = 0;
            double TpostActual = Tpre - xcap / F;
            if (outX) (*outX)[i] = (ll)floor(xcap);
            Etot += energyOf(e, F, Tpre, TpostActual);
            Tcur = TpostActual;
        } else {
            Tcur = Tpre;
        }
    }
    return Etot;
}

static double bestForSet(const vector<char>& inW, vector<ll>* outX) {
    double lo = 0.0, hi = 1.0;
    for (int it = 0; it < 26; it++) {
        double m1 = lo + (hi - lo) / 3.0, m2 = hi - (hi - lo) / 3.0;
        double f1 = evalLambda(inW, m1, nullptr), f2 = evalLambda(inW, m2, nullptr);
        if (f1 < f2) lo = m1; else hi = m2;
    }
    double lam = (lo + hi) / 2.0;
    return evalLambda(inW, lam, outX);
}

int main() {
    scanf("%d %d %d", &T, &K, &Tsink);
    q.assign(T + 1, 0); theta.assign(T + 1, 0); eta.assign(T + 1, 0); cap.assign(T + 1, 0);
    for (int i = 1; i <= T; i++)
        scanf("%lld %lld %lld %lld", &q[i], &theta[i], &eta[i], &cap[i]);

    pre.assign(T + 1, 0.0);
    {
        double F = 0.0, Tcur = 0.0;
        for (int i = 1; i <= T; i++) {
            double Fnew = F + (double)q[i];
            double Tpre = (F == 0.0) ? (double)theta[i] : (F * Tcur + (double)q[i] * theta[i]) / Fnew;
            F = Fnew;
            pre[i] = Tpre;
            Tcur = Tpre;
        }
    }

    // candidate pool: local temperature peaks union top-by-temperature segments
    vector<char> inPool(T + 1, 0);
    for (int i = 1; i <= T; i++) {
        double left = (i > 1) ? pre[i - 1] : -1e18;
        double right = (i < T) ? pre[i + 1] : -1e18;
        if (pre[i] >= left && pre[i] >= right) inPool[i] = 1;
    }
    int M = min(T, max(4 * K, 24));
    vector<int> order(T);
    iota(order.begin(), order.end(), 1);
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (pre[a] != pre[b]) return pre[a] > pre[b];
        return a < b;
    });
    for (int i = 0; i < M; i++) inPool[order[i]] = 1;

    vector<int> pool;
    for (int i = 1; i <= T; i++) if (inPool[i]) pool.push_back(i);
    int poolCap = 6 * K + 40;
    if ((int)pool.size() > poolCap) {
        sort(pool.begin(), pool.end(), [&](int a, int b) { return pre[a] > pre[b]; });
        pool.resize(poolCap);
        sort(pool.begin(), pool.end());
    }

    vector<char> W(T + 1, 0);
    double bestF = 0.0;
    vector<ll> bestX(T + 1, -1);

    for (int round = 0; round < K; round++) {
        int bestCand = -1;
        double bestCandF = bestF;
        for (int cand : pool) {
            if (W[cand]) continue;
            W[cand] = 1;
            double F = bestForSet(W, nullptr);
            if (F > bestCandF + 1e-9) { bestCandF = F; bestCand = cand; }
            W[cand] = 0;
        }
        if (bestCand < 0) break;
        W[bestCand] = 1;
        bestF = bestCandF;
    }
    // one final pass over the chosen set to materialize the actual extractions
    bestF = bestForSet(W, &bestX);

    vector<pair<int,ll>> out;
    for (int i = 1; i <= T; i++) if (bestX[i] >= 0) out.push_back({i, bestX[i]});
    printf("%d\n", (int)out.size());
    for (auto& pr : out) printf("%d %lld\n", pr.first, pr.second);
    return 0;
}
